#!/usr/bin/env python3
"""Convert this project's Paddle Lite plate detector to Core ML with its original FP32 weights.

Requires numpy, protobuf and coremltools (only its protobuf schemas are used).
No Paddle runtime or Apple runtime is required to write the .mlmodel specification.
The emitted model still needs Core ML compilation and numerical validation on macOS.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import pathlib
import sys
import types

import numpy as np

from nb_model import load, summary

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "64dbb06ba96ee2b3552ad8bc4fa167cb0020e2b507f67e96c6427fc60441926a"


def protobuf_model(schema_dir=None):
    if schema_dir is None:
        schema_dir = importlib.metadata.distribution("coremltools").locate_file("coremltools/proto")
    schema_dir = pathlib.Path(schema_dir).resolve()
    if not (schema_dir / "Model_pb2.py").is_file():
        raise ValueError("Core ML protobuf schemas are missing.")
    # Load relative protobuf imports without initializing platform-specific Core ML proxies.
    namespace = "plateocr_coreml_schema"
    package = types.ModuleType(namespace)
    package.__path__ = [str(schema_dir)]
    sys.modules[namespace] = package
    return importlib.import_module(namespace + ".Model_pb2")


def padding(values):
    if len(values) == 2:
        return [values[0], values[0], values[1], values[1]]
    if len(values) == 4:
        return list(values)
    raise ValueError("Unsupported spatial padding.")


class Converter:
    def __init__(self, source, proto):
        self.source = source
        self.spec = proto.Model()
        self.spec.specificationVersion = 5
        self.net = self.spec.neuralNetwork
        self.net.arrayInputShapeMapping = 1  # EXACT_ARRAY_MAPPING
        self.net.imageInputShapeMapping = 1  # RANK4_IMAGE_MAPPING
        self.current = {}
        self.shapes = {}
        self.weights = {name: np.frombuffer(param["data"], dtype="<f4").reshape(param["shape"])
                        for name, param in source["parameters"].items()}
        self.parameter_layers = {}
        if any(param["dtype"] != 5 for param in source["parameters"].values()):
            raise ValueError("Only FP32 parameter storage is supported.")

    def layer(self, kind, inputs, shapes, label=None):
        layer = self.net.layers.add()
        layer.name = label or f"layer_{len(self.net.layers):04d}_{kind}"
        layer.input.extend(inputs)
        outputs = [layer.name + f"_{index}" for index in range(len(shapes))]
        layer.output.extend(outputs)
        getattr(layer, kind).SetInParent()
        for name in inputs:
            tensor = layer.inputTensor.add()
            tensor.rank = len(self.shapes[name]); tensor.dimValue.extend(self.shapes[name])
        for name, shape in zip(outputs, shapes):
            self.shapes[name] = list(shape)
            tensor = layer.outputTensor.add()
            tensor.rank = len(shape); tensor.dimValue.extend(shape)
        return layer, outputs

    def constant(self, value, label):
        value = np.asarray(value, dtype=np.float32)
        layer, outputs = self.layer("loadConstantND", [], [value.shape], label)
        layer.loadConstantND.shape.extend(value.shape)
        layer.loadConstantND.data.floatValue.extend(value.ravel())
        return outputs[0]

    def input(self, name):
        if name not in self.current:
            if name not in self.weights:
                raise ValueError(f"Unknown graph input: {name}")
            value = self.constant(self.weights[name], f"parameter_{len(self.parameter_layers):03d}")
            self.current[name] = value
            self.parameter_layers[name] = value
        return self.current[name]

    def normalize(self, name, shape):
        feature = self.spec.description.input.add(); feature.name = "image"
        feature.type.imageType.width = shape[3]; feature.type.imageType.height = shape[2]
        feature.type.imageType.colorSpace = 30  # BGR, matching the source iOS preprocessing.
        self.shapes["image"] = list(shape)
        divisor = self.constant(np.array([1 / 255], dtype=np.float32).reshape(1, 1, 1, 1), "pixel_scale")
        _, result = self.layer("multiplyBroadcastable", ["image", divisor], [shape], "normalize_pixels")
        mean = self.constant(np.array([.485, .456, .406], dtype=np.float32).reshape(1, 3, 1, 1), "pixel_mean")
        _, result = self.layer("subtractBroadcastable", [result[0], mean], [shape], "subtract_mean")
        deviation = self.constant((1 / np.array([.229, .224, .225], dtype=np.float32)).reshape(1, 3, 1, 1), "pixel_inv_std")
        _, result = self.layer("multiplyBroadcastable", [result[0], deviation], [shape], "normalize_std")
        self.current[name] = result[0]

    def convert(self):
        for operation in self.source["operations"]:
            kind, a = operation["type"], operation["attrs"]
            ins, outs = operation["inputs"], operation["outputs"]
            if kind == "feed":
                name = outs["Out"][0]
                self.normalize(name, self.source["variables"][name]["shape"])
                continue
            if kind == "fetch":
                output = self.input(ins["X"][0])
                layer, result = self.layer("copy", [output], [self.shapes[output]], "plate_output")
                layer.output[0] = "predictions"
                self.shapes["predictions"] = self.shapes[result[0]]
                feature = self.spec.description.output.add(); feature.name = "predictions"
                feature.type.multiArrayType.shape.extend(self.shapes[output])
                feature.type.multiArrayType.dataType = 65568  # FLOAT32
                continue
            input_key = "Input" if kind == "conv2d" else "X"
            x_names = [self.input(name) for name in ins[input_key]]
            x = x_names[0]; shape = self.shapes[x]
            if kind == "conv2d":
                if a["data_format"] != "NCHW" or a["padding_algorithm"] != "EXPLICIT":
                    raise ValueError("Unsupported convolution layout/padding.")
                weights = self.weights[ins["Filter"][0]]
                out_channels, in_channels, kh, kw = weights.shape
                pad = padding(a["paddings"])
                sh, sw = a["strides"]; dh, dw = a["dilations"]
                output_shape = [shape[0], out_channels,
                    (shape[2] + pad[0] + pad[1] - dh * (kh - 1) - 1) // sh + 1,
                    (shape[3] + pad[2] + pad[3] - dw * (kw - 1) - 1) // sw + 1]
                layer, result = self.layer("convolution", [x], [output_shape])
                conv = layer.convolution
                conv.outputChannels = out_channels; conv.kernelChannels = in_channels; conv.nGroups = a["groups"]
                conv.kernelSize.extend([kh, kw]); conv.stride.extend([sh, sw]); conv.dilationFactor.extend([dh, dw])
                for before, after in [(pad[0], pad[1]), (pad[2], pad[3])]:
                    edge = conv.valid.paddingAmounts.borderAmounts.add()
                    edge.startEdgeSize = before; edge.endEdgeSize = after
                conv.weights.floatValue.extend(weights.ravel())
                self.parameter_layers[ins["Filter"][0]] = layer.name + ".convolution.weights"
            elif kind == "reshape2":
                target = [shape[i] if size == 0 else size for i, size in enumerate(a["shape"])]
                if -1 in target:
                    target[target.index(-1)] = math.prod(shape) // math.prod(size for size in target if size != -1)
                if math.prod(shape) != math.prod(target):
                    raise ValueError("Reshape changes tensor element count.")
                layer, result = self.layer("reshapeStatic", [x], [target])
                layer.reshapeStatic.targetShape.extend(target)
            elif kind.startswith("elementwise_"):
                y = self.input(ins["Y"][0]); y_shape = self.shapes[y]
                if a.get("axis", -1) != -1:
                    raise ValueError("Only trailing-dimension elementwise broadcasts are supported.")
                target = list(np.broadcast_shapes(shape, y_shape))
                name = {"elementwise_add": "addBroadcastable", "elementwise_sub": "subtractBroadcastable",
                        "elementwise_mul": "multiplyBroadcastable"}[kind]
                layer, result = self.layer(name, [x, y], [target])
            elif kind in ("silu", "sigmoid"):
                layer, result = self.layer("activation", [x], [shape])
                layer.activation.sigmoid.SetInParent()
                if kind == "silu":
                    layer, result = self.layer("multiplyBroadcastable", [x, result[0]], [shape])
            elif kind == "concat":
                axis = a["axis"] % len(shape)
                target = list(shape); target[axis] = sum(self.shapes[name][axis] for name in x_names)
                layer, result = self.layer("concatND", x_names, [target])
                layer.concatND.axis = axis
            elif kind == "split":
                axis = a["axis"] % len(shape)
                sections = a["sections"] or [shape[axis] // a["num"]] * a["num"]
                if sum(sections) != shape[axis]: raise ValueError("Split sizes do not cover the source axis.")
                targets = [list(shape) for _ in sections]
                for target, count in zip(targets, sections): target[axis] = count
                layer, result = self.layer("splitND", [x], targets)
                layer.splitND.axis = axis; layer.splitND.splitSizes.extend(sections)
            elif kind == "pool2d":
                if a["pooling_type"] != "max" or a["ceil_mode"] or a["adaptive"] or a["global_pooling"]:
                    raise ValueError("Unsupported pooling configuration.")
                pad = padding(a["paddings"]); kh, kw = a["ksize"]; sh, sw = a["strides"]
                target = [shape[0], shape[1], (shape[2] + pad[0] + pad[1] - kh) // sh + 1,
                          (shape[3] + pad[2] + pad[3] - kw) // sw + 1]
                layer, result = self.layer("pooling", [x], [target])
                pool = layer.pooling; pool.type = 0; pool.kernelSize.extend([kh, kw]); pool.stride.extend([sh, sw])
                for before, after in [(pad[0], pad[1]), (pad[2], pad[3])]:
                    edge = pool.valid.paddingAmounts.borderAmounts.add()
                    edge.startEdgeSize = before; edge.endEdgeSize = after
            elif kind == "nearest_interp_v2":
                if a["align_corners"] or a["interp_method"] != "nearest" or a["data_layout"] != "NCHW":
                    raise ValueError("Unsupported nearest resize configuration.")
                factors = a["scale"]
                if any(value != int(value) for value in factors): raise ValueError("Expected integer resize factors.")
                target = shape[:2] + [shape[2] * int(factors[0]), shape[3] * int(factors[1])]
                layer, result = self.layer("upsample", [x], [target])
                layer.upsample.scalingFactor.extend(map(int, factors)); layer.upsample.mode = 0
            elif kind == "transpose2":
                axes = a["axis"]
                layer, result = self.layer("transpose", [x], [[shape[axis] for axis in axes]])
                layer.transpose.axes.extend(axes)
            elif kind == "softmax":
                layer, result = self.layer("softmaxND", [x], [shape]); layer.softmaxND.axis = a["axis"]
            elif kind == "scale":
                layer, result = self.layer("activation", [x], [shape])
                layer.activation.linear.alpha = a["scale"]
                layer.activation.linear.beta = a["bias"] if a["bias_after_scale"] else a["bias"] * a["scale"]
            else:
                raise ValueError(f"Unsupported operation: {kind}")
            output_names = outs.get("Out", outs.get("Output"))
            if output_names is None or len(output_names) != len(result): raise ValueError("Unsupported graph outputs.")
            # Optimized Lite graphs reuse blob names. SSA names prevent overwrite/liveness ambiguity.
            for original, unique in zip(output_names, result): self.current[original] = unique
        metadata = self.spec.description.metadata
        metadata.shortDescription = "Original VINRecognizeSDK yolo.nb plate detector, converted with identical FP32 parameters."
        metadata.userDefined["source_nb_sha256"] = self.source["sha256"]
        metadata.userDefined["source_optimizer"] = self.source["optimizer"]
        metadata.userDefined["coordinate_format"] = "xywh_center_pixels"
        metadata.userDefined["preprocessing"] = "BGR; (pixel/255 - [0.485,0.456,0.406]) * (1/[0.229,0.224,0.225])"
        metadata.userDefined["device_validation"] = "pending"
        if set(self.parameter_layers) != set(self.weights):
            raise ValueError("Some original weights were not incorporated into the Core ML graph.")
        return self.spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--schema-dir", type=pathlib.Path)
    parser.add_argument("--report", type=pathlib.Path)
    args = parser.parse_args()
    source = load(args.source)
    if source["sha256"] != SOURCE_SHA256: raise ValueError("Source is not the expected baseline plate detector.")
    converter = Converter(source, protobuf_model(args.schema_dir))
    spec = converter.convert()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = spec.SerializeToString(deterministic=True)
    args.output.write_bytes(payload)
    report = summary(source)
    report.update({"format": "Core ML neuralNetwork (specification version 5)", "coreml_sha256": hashlib.sha256(payload).hexdigest(),
        "source_main_commit": "9ec6f24bd75b5486fea4df176ae98e1108b75d70", "same_weights_as_nb": True,
        "coreml_layers": len(spec.neuralNetwork.layers), "input": {"name": "image", "shape": [1, 3, 640, 640], "color": "BGR"},
        "output": {"name": "predictions", "shape": [1, 5, 8400], "coordinates": "xywh_center_pixels"},
        "parameter_mapping": converter.parameter_layers, "coreml_compile_verified": False, "device_inference_verified": False})
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Converted {len(source['parameters'])} original parameter tensors into {len(spec.neuralNetwork.layers)} Core ML layers.")
    print(f"Model: {args.output}; SHA-256: {report['coreml_sha256']}")


if __name__ == "__main__":
    main()
