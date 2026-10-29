"""
Empirical probe of the real De-plate-pose-500.onnx.

Goal: confirm the keypoint output range/order the iOS SDK must decode.
We run a forward pass (random input is fine — we only care about the
coordinate *range* the model emits, not a real detection) and print the
best proposal's box + 4 keypoints exactly as YOLOOutputDecoder would read them.
"""
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL = r"E:\Git\ITLogica\PlateOCR-de\OCR\LicensePlateOCR\LicensePlateSDK\Resources\De-plate-pose-500.onnx"

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
ishape = sess.get_inputs()[0].shape
H = ishape[2] if ishape[2] is not None else 640
W = ishape[3] if ishape[3] is not None else 640
print("input:", iname, [1, 3, H, W])

dummy = np.random.rand(1, 3, H, W).astype(np.float32)
out = sess.run(None, {iname: dummy})[0]
print("raw output shape:", out.shape)

shape = out.shape
if shape[1] < shape[2]:
    C, N = shape[1], shape[2]
    arr = out[0]            # [C, N]
    get = lambda i: arr[:, i]
    layout = "[1,C,N]"
else:
    N, C = shape[2], shape[1]
    arr = out[0].T          # [N, C]
    get = lambda i: arr[i]
    layout = "[1,N,C]"
print("layout:", layout, "C=", C, "N=", N, "inferred kpts=", (C - 5) // 3)

# best proposal by confidence
confs = np.array([get(i)[4] for i in range(N)])
best = int(np.argmax(confs))
row = get(best)
cx, cy, w, h, conf = row[0], row[1], row[2], row[3], row[4]
print(f"\nbest proposal idx={best} conf={conf:.4f}")
print(f"  box(cx,cy,w,h) 640-space: cx={cx:.2f} cy={cy:.2f} w={w:.2f} h={h:.2f}")
print(f"  box x-range: {cx-w/2:.2f}..{cx+w/2:.2f}  y-range: {cy-h/2:.2f}..{cy+h/2:.2f}")
nk = (C - 5) // 3
print(f"  keypoints (x,y,vis) 640-space:")
names = ["left_top", "right_top", "right_bottom", "left_bottom"]
for k in range(nk):
    kx, ky, kv = row[5 + k*3], row[6 + k*3], row[7 + k*3]
    print(f"    {names[k] if k < len(names) else k}: x={kx:.2f} y={ky:.2f} vis={kv:.4f}")

# Coordinate range diagnosis
allk = [row[5 + k*3:5 + k*3 + 3] for k in range(nk)]
xs = [v[0] for v in allk]; ys = [v[1] for v in allk]
print(f"\nkeypoint x range: {min(xs):.2f}..{max(xs):.2f}")
print(f"keypoint y range: {min(ys):.2f}..{max(ys):.2f}")
if max(xs) <= 1.0001 and max(ys) <= 1.0001:
    print("  >>> KEYPOINTS APPEAR NORMALIZED [0,1] — SDK would need /640 (MISMATCH if treated as px)")
elif max(xs) > 1.5:
    print("  >>> KEYPOINTS IN PIXEL SPACE (0..640) — matches SDK assumption (640-px, same as box)")
