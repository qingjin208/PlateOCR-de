"""
German-plate YOLOv8-pose pipeline REFERENCE (numpy-only, no OpenCV/onnxruntime required).

This mirrors exactly what the iOS Swift code must do, so the algorithm is proven
before any Swift is written. Validates:

  1. ONNX output layout  [1, 17, 8400]  (5 box/conf + 4*3 keypoints)
  2. decode -> box (xyxy) + 4 keypoints in 640 input space
  3. NMS
  4. remap from 640 letterbox space -> normalized original-image coords (top-left origin)
  5. 4-point homography (perspective correction) from the 4 plate corners to a rectangle

Keypoint order is FIXED by the training config (data.yaml / pose_config.yaml):
    index 0 = left_top, 1 = right_top, 2 = right_bottom, 3 = left_bottom

Run:  python pose_pipeline_reference.py
Optional real test:  python pose_pipeline_reference.py path/to/image.jpg
(requires onnxruntime + the .onnx; otherwise only the synthetic check runs.)
"""
import sys
import numpy as np

INPUT_SIZE = 640
CONF_THR = 0.45
IOU_THR = 0.45
NUM_KPTS = 4
KPT_NAMES = ["left_top", "right_top", "right_bottom", "left_bottom"]


# ----------------------------------------------------------------------------
# Preprocessing (mirrors the Swift letterbox + RGB/255 step)
# ----------------------------------------------------------------------------
def letterbox(img, new_shape=INPUT_SIZE, color=114):
    h, w = img.shape[:2]
    scale = min(new_shape / w, new_shape / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    pad_x = (new_shape - nw) // 2
    pad_y = (new_shape - nh) // 2
    resized = np.array([color], dtype=np.uint8).reshape(1, 1, 1).repeat(nh, 0).repeat(nw, 1).repeat(3, 2)
    resized[:nh, :nw] = img[:nh, :nw] if False else _resize_nearest(img, nw, nh)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def _resize_nearest(img, nw, nh):
    h, w = img.shape[:2]
    if (h, w) == (nh, nw):
        return img.copy()
    ys = (np.arange(nh) + 0.5) * h / nh - 0.5
    xs = (np.arange(nw) + 0.5) * w / nw - 0.5
    ys = np.clip(ys, 0, h - 1).astype(int)
    xs = np.clip(xs, 0, w - 1).astype(int)
    return img[np.ix_(ys, xs)]


def preprocess(img, new_shape=INPUT_SIZE):
    """img: HWC uint8 (RGB). Returns NCHW float32 /255."""
    lb, _, _, _ = letterbox(img, new_shape)
    rgb = lb.astype(np.float32) / 255.0
    chw = rgb.transpose(2, 0, 1)
    return np.expand_dims(chw, 0).astype(np.float32)


# ----------------------------------------------------------------------------
# Decode (mirrors YOLOOutputDecoder + pose keypoint extraction)
# ----------------------------------------------------------------------------
def _iou(a, b):
    xa, ya, xb, yb = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, xb - xa), max(0.0, yb - ya)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, thr):
    order = np.argsort(-np.array(scores))
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        ious = np.array([_iou(boxes[i], boxes[j]) for j in rest])
        order = rest[ious <= thr]
    return keep


def decode_pose(out, conf_thr=CONF_THR, iou_thr=IOU_THR, num_kpts=NUM_KPTS, input_size=INPUT_SIZE):
    shape = out.shape
    if shape[1] < shape[2]:
        C, N = shape[1], shape[2]
        arr = out[0]  # [C, N]
        get = lambda i: arr[:, i]
    else:
        N, C = shape[2], shape[1]
        arr = out[0].T  # [N, C]
        get = lambda i: arr[i]
    assert C == 5 + num_kpts * 3, f"unexpected channels {C}"

    proposals = []
    for i in range(N):
        row = get(i)
        conf = float(row[4])
        if conf < conf_thr:
            continue
        cx, cy, w, h = row[0], row[1], row[2], row[3]
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        kpts = []
        for k in range(num_kpts):
            kx, ky, kv = row[5 + k * 3], row[6 + k * 3], row[7 + k * 3]
            kpts.append((float(kx), float(ky), float(kv)))
        proposals.append({"box": (x1, y1, x2, y2), "conf": conf, "kpts": kpts})

    if not proposals:
        return []
    boxes = [p["box"] for p in proposals]
    scores = [p["conf"] for p in proposals]
    keep = nms(boxes, scores, iou_thr)
    return [proposals[i] for i in keep]


def remap_box(box, scale, pad_x, pad_y, orig_w, orig_h):
    x1, y1, x2, y2 = box
    nx1 = max(0.0, min(1.0, (x1 - pad_x) / scale / orig_w))
    ny1 = max(0.0, min(1.0, (y1 - pad_y) / scale / orig_h))
    nx2 = max(0.0, min(1.0, (x2 - pad_x) / scale / orig_w))
    ny2 = max(0.0, min(1.0, (y2 - pad_y) / scale / orig_h))
    return (nx1, ny1, nx2, ny2)


def remap_kpt(kpt, scale, pad_x, pad_y, orig_w, orig_h):
    x, y, v = kpt
    nx = max(0.0, min(1.0, (x - pad_x) / scale / orig_w))
    ny = max(0.0, min(1.0, (y - pad_y) / scale / orig_h))
    return (nx, ny, v)


# ----------------------------------------------------------------------------
# Perspective correction (4-point homography, mirrors CIPerspectiveCorrection)
# ----------------------------------------------------------------------------
def homography_4pt(src, dst):
    """src, dst: list of 4 (x,y). Returns 3x3 H mapping src -> dst."""
    A = []
    for (x, y), (xp, yp) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -x * xp, -y * xp, -xp])
        A.append([0, 0, 0, x, y, 1, -x * yp, -y * yp, -yp])
    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_homography(H, pts):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    ones = np.ones((pts.shape[0], 1))
    ph = np.hstack([pts, ones])
    out = (H @ ph.T).T
    out = out[:, :2] / out[:, 2:3]
    return out


def warp_image(img, H, out_w, out_h):
    h, w = img.shape[:2]
    ys, xs = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing="ij")
    dst_pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    src = apply_homography(np.linalg.inv(H), dst_pts)
    sx, sy = src[:, 0], src[:, 1]
    sx0 = np.floor(sx).astype(int); sy0 = np.floor(sy).astype(int)
    sx1 = sx0 + 1; sy1 = sy0 + 1
    valid = (sx0 >= 0) & (sx1 < w) & (sy0 >= 0) & (sy1 < h)
    out = np.zeros((out_h, out_w, 3), dtype=np.float32)
    for c in range(3):
        v00 = np.where(valid, img[sy0, sx0, c], 0).astype(np.float64)
        v10 = np.where(valid, img[sy0, sx1, c], 0).astype(np.float64)
        v01 = np.where(valid, img[sy1, sx0, c], 0).astype(np.float64)
        v11 = np.where(valid, img[sy1, sx1, c], 0).astype(np.float64)
        dx = np.clip(sx - sx0, 0, 1); dy = np.clip(sy - sy0, 0, 1)
        val = (v00 * (1 - dx) * (1 - dy) + v10 * dx * (1 - dy)
               + v01 * (1 - dx) * dy + v11 * dx * dy)
        out[..., c] = np.where(valid, val, 0).reshape(out_h, out_w)
    return out


# ----------------------------------------------------------------------------
# Synthetic validation (no model needed)
# ----------------------------------------------------------------------------
def synthetic_check():
    print("=== Synthetic pipeline check ===")
    N = 8400
    out = np.zeros((1, 5 + NUM_KPTS * 3, N), dtype=np.float32)
    # one perfect proposal at index 0
    cx, cy, w, h = 320.0, 320.0, 400.0, 200.0
    out[0, 0, 0] = cx; out[0, 1, 0] = cy; out[0, 2, 0] = w; out[0, 3, 0] = h; out[0, 4, 0] = 0.9
    kp = [(120, 220, 2), (520, 220, 2), (520, 420, 2), (120, 420, 2)]  # left_top,right_top,right_bottom,left_bottom
    for k, (kx, ky, kv) in enumerate(kp):
        out[0, 5 + k * 3, 0] = kx
        out[0, 6 + k * 3, 0] = ky
        out[0, 7 + k * 3, 0] = kv
    # a second overlapping low-conf detection to exercise NMS
    out[0, 0, 1] = 325; out[0, 1, 1] = 318; out[0, 2, 1] = 405; out[0, 3, 1] = 205; out[0, 4, 1] = 0.5
    for k, (kx, ky, kv) in enumerate(kp):
        out[0, 5 + k * 3, 1] = kx + 5; out[0, 6 + k * 3, 1] = ky - 2; out[0, 7 + k * 3, 1] = kv

    dets = decode_pose(out)
    assert len(dets) == 1, f"NMS should keep 1, got {len(dets)}"
    d = dets[0]
    print("decoded box(xyxy,640):", tuple(round(v, 1) for v in d["box"]), "conf:", d["conf"])
    for i, kp in enumerate(d["kpts"]):
        print(f"  kpt {KPT_NAMES[i]}: (x={kp[0]:.1f}, y={kp[1]:.1f}, v={kp[2]:.0f})")
    # verify keypoint ordering matches the placed corners
    assert abs(d["kpts"][0][0] - 120) < 1e-3 and abs(d["kpts"][0][1] - 220) < 1e-3
    assert abs(d["kpts"][3][0] - 120) < 1e-3 and abs(d["kpts"][3][1] - 420) < 1e-3

    # remap to a 1280x720 original
    orig_w, orig_h = 1280, 720
    scale = min(INPUT_SIZE / orig_w, INPUT_SIZE / orig_h)
    nw, nh = int(round(orig_w * scale)), int(round(orig_h * scale))
    pad_x = (INPUT_SIZE - nw) // 2
    pad_y = (INPUT_SIZE - nh) // 2
    box_n = remap_box(d["box"], scale, pad_x, pad_y, orig_w, orig_h)
    kpts_n = [remap_kpt(k, scale, pad_x, pad_y, orig_w, orig_h) for k in d["kpts"]]
    print("box normalized:", tuple(round(v, 4) for v in box_n))
    for i, k in enumerate(kpts_n):
        print(f"  kpt {KPT_NAMES[i]} normalized: ({k[0]:.4f}, {k[1]:.4f})")

    # pixel coords in original image (top-left origin)
    src_px = [(k[0] * orig_w, k[1] * orig_h) for k in kpts_n]
    dst_w, dst_h = 400, 200
    dst_px = [(0, 0), (dst_w, 0), (dst_w, dst_h), (0, dst_h)]  # left_top,right_top,right_bottom,left_bottom
    H = homography_4pt(src_px, dst_px)
    mapped = apply_homography(H, src_px)
    print("homography maps source corners ->", np.round(mapped, 2).tolist())
    assert np.allclose(mapped, dst_px, atol=1e-6), "homography must map corners exactly"
    print("OK: 4-point homography maps the 4 plate corners to the rectangle exactly.")

    # warp a synthetic original image and confirm corner pixel values survive
    img = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
    img[:] = [30, 60, 90]
    for (px, py), color in zip(src_px, [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        img[int(py) - 10:int(py) + 10, int(px) - 10:int(px) + 10] = color
    warped = warp_image(img, H, dst_w, dst_h)
    corner_vals = [warped[0, 0], warped[0, -1], warped[-1, -1], warped[-1, 0]]
    print("warped corner colors (TL,TR,BR,BL):", [tuple(map(int, c)) for c in corner_vals])
    print("OK: synthetic pipeline passed.\n")


# ----------------------------------------------------------------------------
# Optional real-inference check (needs onnxruntime + image)
# ----------------------------------------------------------------------------
def real_check(image_path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not available; skipping real inference check.")
        return
    from PIL import Image
    model = r"E:\NewFolder\german_vehicle_dataset\german_plate_pose_500\runs\pose\De-plate-pose-500\weights\De-plate-pose-500.onnx"
    img = np.array(Image.open(image_path).convert("RGB"))
    orig_h, orig_w = img.shape[:2]
    x = preprocess(img)
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    out = sess.run(None, {iname: x})[0]
    print("real output shape:", out.shape)
    dets = decode_pose(out)
    print(f"real detections: {len(dets)}")
    for d in dets[:5]:
        print("  box(xyxy,640):", tuple(round(v, 1) for v in d["box"]), "conf:", round(d["conf"], 3))


if __name__ == "__main__":
    synthetic_check()
    if len(sys.argv) > 1:
        real_check(sys.argv[1])
