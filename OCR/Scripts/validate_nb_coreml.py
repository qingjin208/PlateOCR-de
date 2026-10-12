#!/usr/bin/env python3
"""Validate original NB parameters and exported Core ML graph semantics on CPU.

The NB graph is evaluated with PyTorch. The serialized Core ML graph is translated
independently into ONNX and evaluated with ONNX Runtime. This does not replace
compilation, prediction parity, or energy measurements with Apple's Core ML runtime.
"""

import argparse
import hashlib
import json
import pathlib

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
import torch
import torch.nn.functional as functional

from nb_model import load
from convert_nb_to_coreml import protobuf_model, padding


def run_nb(model, pixels):
    tensors = {name: torch.from_numpy(np.frombuffer(param["data"], dtype="<f4").copy().reshape(param["shape"]))
               for name, param in model["parameters"].items()}
    x = torch.from_numpy(pixels.copy())
    mean = torch.tensor([.485, .456, .406], dtype=torch.float32).reshape(1, 3, 1, 1)
    inv_std = (1 / torch.tensor([.229, .224, .225], dtype=torch.float32)).reshape(1, 3, 1, 1)
    normalized = (x / 255 - mean) * inv_std
    with torch.inference_mode():
        for operation in model["operations"]:
            kind, a = operation["type"], operation["attrs"]
            ins, outs = operation["inputs"], operation["outputs"]
            if kind == "feed": tensors[outs["Out"][0]] = normalized; continue
            if kind == "fetch": return tensors[ins["X"][0]].numpy()
            x = tensors[ins["Input" if kind == "conv2d" else "X"][0]]
            if kind == "conv2d":
                p = padding(a["paddings"])
                result = functional.conv2d(functional.pad(x, [p[2], p[3], p[0], p[1]]),
                    tensors[ins["Filter"][0]], stride=a["strides"], dilation=a["dilations"], groups=a["groups"])
            elif kind == "reshape2":
                result = x.reshape([x.shape[i] if v == 0 else v for i, v in enumerate(a["shape"])])
            elif kind == "elementwise_add": result = x + tensors[ins["Y"][0]]
            elif kind == "elementwise_sub": result = x - tensors[ins["Y"][0]]
            elif kind == "elementwise_mul": result = x * tensors[ins["Y"][0]]
            elif kind == "silu": result = functional.silu(x)
            elif kind == "sigmoid": result = torch.sigmoid(x)
            elif kind == "concat": result = torch.cat([tensors[name] for name in ins["X"]], dim=a["axis"])
            elif kind == "split":
                sections = a["sections"] or [x.shape[a["axis"]] // a["num"]] * a["num"]
                result = torch.split(x, sections, dim=a["axis"])
            elif kind == "pool2d":
                p = padding(a["paddings"])
                result = functional.max_pool2d(functional.pad(x, [p[2], p[3], p[0], p[1]], value=float("-inf")),
                    a["ksize"], stride=a["strides"])
            elif kind == "nearest_interp_v2": result = functional.interpolate(x, scale_factor=a["scale"], mode="nearest")
            elif kind == "transpose2": result = x.permute(a["axis"])
            elif kind == "softmax": result = torch.softmax(x, dim=a["axis"])
            elif kind == "scale": result = x * a["scale"] + a["bias"] if a["bias_after_scale"] else (x + a["bias"]) * a["scale"]
            else: raise ValueError(kind)
            names = outs.get("Out", outs.get("Output"))
            for name, value in zip(names, result if kind == "split" else [result]): tensors[name] = value
    raise ValueError("NB graph has no fetch output.")


def coreml_to_onnx(spec):
    nodes, initializers = [], []
    def constant(name, values):
        initializers.append(numpy_helper.from_array(np.asarray(values), name=name))
        return name
    for layer in spec.neuralNetwork.layers:
        kind = layer.WhichOneof("layer")
        inputs, outputs = list(layer.input), list(layer.output)
        attributes = {}
        if kind == "loadConstantND":
            constant(outputs[0], np.array(layer.loadConstantND.data.floatValue, dtype=np.float32).reshape(layer.loadConstantND.shape))
            continue
        if kind == "convolution":
            c = layer.convolution
            shape = [c.outputChannels, c.kernelChannels] + list(c.kernelSize)
            inputs.append(constant(layer.name + "_weights", np.array(c.weights.floatValue, dtype=np.float32).reshape(shape)))
            borders = c.valid.paddingAmounts.borderAmounts
            attributes = {"strides": list(c.stride), "dilations": list(c.dilationFactor), "group": c.nGroups,
                "pads": [borders[0].startEdgeSize, borders[1].startEdgeSize, borders[0].endEdgeSize, borders[1].endEdgeSize]}
            op = "Conv"
        elif kind == "pooling":
            p = layer.pooling; borders = p.valid.paddingAmounts.borderAmounts
            op = "MaxPool"; attributes = {"kernel_shape": list(p.kernelSize), "strides": list(p.stride),
                "pads": [borders[0].startEdgeSize, borders[1].startEdgeSize, borders[0].endEdgeSize, borders[1].endEdgeSize]}
        elif kind == "reshapeStatic":
            op = "Reshape"; inputs.append(constant(layer.name + "_shape", np.array(layer.reshapeStatic.targetShape, dtype=np.int64)))
        elif kind in ["addBroadcastable", "subtractBroadcastable", "multiplyBroadcastable"]:
            op = {"addBroadcastable": "Add", "subtractBroadcastable": "Sub", "multiplyBroadcastable": "Mul"}[kind]
        elif kind == "activation":
            activation = layer.activation.WhichOneof("NonlinearityType")
            if activation == "sigmoid": op = "Sigmoid"
            elif activation == "linear":
                alpha = constant(layer.name + "_alpha", np.array(layer.activation.linear.alpha, dtype=np.float32))
                beta = constant(layer.name + "_beta", np.array(layer.activation.linear.beta, dtype=np.float32))
                intermediate = layer.name + "_scaled"
                nodes.append(helper.make_node("Mul", [inputs[0], alpha], [intermediate]))
                inputs = [intermediate, beta]; op = "Add"
            else: raise ValueError(activation)
        elif kind == "concatND": op = "Concat"; attributes = {"axis": layer.concatND.axis}
        elif kind == "splitND":
            op = "Split"; attributes = {"axis": layer.splitND.axis}
            inputs.append(constant(layer.name + "_sections", np.array(layer.splitND.splitSizes, dtype=np.int64)))
        elif kind == "transpose": op = "Transpose"; attributes = {"perm": list(layer.transpose.axes)}
        elif kind == "softmaxND": op = "Softmax"; attributes = {"axis": layer.softmaxND.axis}
        elif kind == "upsample":
            op = "Resize"
            inputs = [inputs[0], "", constant(layer.name + "_scales", np.array([1, 1] + list(layer.upsample.scalingFactor), dtype=np.float32))]
            attributes = {"mode": "nearest", "coordinate_transformation_mode": "asymmetric", "nearest_mode": "floor"}
        elif kind == "copy": op = "Identity"
        else: raise ValueError(f"Unsupported Core ML validation layer: {kind}")
        nodes.append(helper.make_node(op, inputs, outputs, name=layer.name, **attributes))
    graph = helper.make_graph(nodes, "original_plate_detector",
        [helper.make_tensor_value_info("image", onnx.TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("predictions", onnx.TensorProto.FLOAT, [1, 5, 8400])], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def verify_weights(source, spec, report):
    layers = {layer.name: layer for layer in spec.neuralNetwork.layers}
    for name, location in report["parameter_mapping"].items():
        if location.endswith(".convolution.weights"):
            values = layers[location.split(".")[0]].convolution.weights.floatValue
        else:
            layer = next(layer for layer in spec.neuralNetwork.layers if location in layer.output)
            values = layer.loadConstantND.data.floatValue
        recovered = np.array(values, dtype="<f4").tobytes()
        if recovered != source["parameters"][name]["data"]:
            raise ValueError(f"Original FP32 weights were changed: {name}")
    if set(report["parameter_mapping"]) != set(source["parameters"]): raise ValueError("Parameter mapping is incomplete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--coreml", type=pathlib.Path, required=True)
    parser.add_argument("--provenance", type=pathlib.Path, required=True)
    parser.add_argument("--schema-dir", type=pathlib.Path)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    args = parser.parse_args()
    source = load(args.source)
    proto = protobuf_model(args.schema_dir); spec = proto.Model()
    payload = args.coreml.read_bytes(); spec.ParseFromString(payload)
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    if provenance["sha256"] != source["sha256"] or provenance["coreml_sha256"] != hashlib.sha256(payload).hexdigest():
        raise ValueError("The validation inputs do not match the provenance hashes.")
    verify_weights(source, spec, provenance)
    model = coreml_to_onnx(spec)
    options = ort.SessionOptions(); options.intra_op_num_threads = 2
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    torch.set_num_threads(2)
    rng = np.random.default_rng(1234)
    samples = {"black": np.zeros((1, 3, 640, 640), dtype=np.float32),
               "letterbox_gray": np.full((1, 3, 640, 640), 114, dtype=np.float32),
               "seeded_random": rng.integers(0, 256, (1, 3, 640, 640)).astype(np.float32)}
    cases = []
    for name, pixels in samples.items():
        original = run_nb(source, pixels)
        exported = session.run(["predictions"], {"image": pixels})[0]
        np.testing.assert_allclose(exported, original, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(exported[:, 4, :], original[:, 4, :], rtol=1e-4, atol=1e-5)
        case = {"name": name, "shape": list(exported.shape),
            "max_coordinate_absolute_error": float(np.abs(exported[:, :4, :] - original[:, :4, :]).max()),
            "max_score_absolute_error": float(np.abs(exported[:, 4, :] - original[:, 4, :]).max())}
        cases.append(case); print("PASS:", json.dumps(case))
    report = {"source_sha256": source["sha256"], "coreml_sha256": hashlib.sha256(payload).hexdigest(),
        "parameter_tensors_verified_bit_for_bit": len(source["parameters"]),
        "comparison": "NB graph evaluated with PyTorch vs serialized Core ML graph evaluated through ONNX Runtime",
        "cases": cases, "apple_runtime_verified": False,
        "limitations": "Synthetic graph equivalence only. Actual Paddle Lite and Core ML device predictions, accuracy, acceleration and energy remain unverified."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS: original weights and all synthetic graph-equivalence cases.")


if __name__ == "__main__":
    main()
