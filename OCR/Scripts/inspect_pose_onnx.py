"""
Inspect the trained German-plate YOLOv8-pose ONNX model and sanity-check the
decode + keypoint extraction logic. Used to validate the iOS Swift port before
writing code. Runtime-agnostic: only confirms tensor shapes and decode structure.
"""
import os
import numpy as np

import onnx
try:
    import onnxruntime as ort
    HAVE_ORT = True
except ImportError:
    HAVE_ORT = False
    print("(onnxruntime not available on this Python; shape-only inspection)")

MODEL = r"E:\NewFolder\german_vehicle_dataset\german_plate_pose_500\runs\pose\De-plate-pose-500\weights\De-plate-pose-500.onnx"

m = onnx.load(MODEL)
print("=== ONNX model ===")
for i, inp in enumerate(m.graph.input):
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"input[{i}] {inp.name}: {dims}")
for o in m.graph.output:
    dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(f"output {o.name}: {dims}")

# Run a dummy forward pass to confirm output layout.
if not HAVE_ORT:
    print("SKIP dummy forward pass (no onnxruntime).")
    raise SystemExit(0)
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
ishape = sess.get_inputs()[0].shape
print("session input:", iname, ishape)
# ishape like [1,3,640,640]
H = ishape[2] if ishape[2] is not None else 640
W = ishape[3] if ishape[3] is not None else 640
dummy = np.random.rand(1, 3, H, W).astype(np.float32).astype(np.float32)
out = sess.run(None, {iname: dummy})[0]
print("raw output shape:", out.shape)
# YOLOv8-pose: [1, 5 + nk*3, 8400] or [1, 8400, 5+nk*3]
shape = out.shape
if shape[1] < shape[2]:
    C, N = shape[1], shape[2]
    layout = "[1, C, N]"
else:
    N, C = shape[2], shape[1]
    layout = "[1, N, C]"
print("layout:", layout, "channels C =", C, "proposals N =", N)
nk = (C - 5) // 3
print("inferred keypoints nk =", nk, "(expected 4)")
print("expected channels for 4 kpts =", 5 + 4 * 3)

# Decode sanity for the [1,C,N] layout.
if layout == "[1, C, N]":
    arr = out[0]  # [C, N]
    # proposal i -> [x,y,w,h,conf, k0x,k0y,k0v, k1..., k2..., k3...]
    confs = arr[4, :]
    best = int(np.argmax(confs))
    proposal = arr[:, best]
    box = proposal[0:4]
    conf = proposal[4]
    kpts = proposal[5:5 + nk * 3].reshape(nk, 3)
    print("\n=== best dummy proposal (random input, not meaningful) ===")
    print("box (cx,cy,w,h):", np.round(box, 2), "conf:", float(conf))
    print("keypoints (x,y,vis):")
    names = ["left_top", "right_top", "right_bottom", "left_bottom"]
    for i in range(nk):
        print(f"  {names[i] if i < len(names) else i}: {np.round(kpts[i],2)}")
print("\nOK: model loads and decodes structurally.")
