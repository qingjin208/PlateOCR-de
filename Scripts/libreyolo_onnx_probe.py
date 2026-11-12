# -*- coding: utf-8 -*-
"""实证校验 LibreYOLO(YOLO-NAS-pose) ONNX 的输出契约:
用数据集 val 图 + GT 标签验证 box 格式、角点顺序、sigmoid 位置、BGR/127/bottom-right 预处理。
"""
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path

ROOT = Path(r"E:\NewFolder\german_vehicle_dataset\german_plate_pose_500_libreyolo")
MODEL = ROOT / "weights" / "best.onnx"
INPUT = 512
PAD = 127  # YOLO_NAS_POSE_PAD_VALUE

sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
print("inputs:", [(i.name, i.shape) for i in sess.get_inputs()])
print("outputs:", [(o.name, o.shape) for o in sess.get_outputs()])


def preprocess(img_rgb: np.ndarray):
    """libreyolo preprocess_pose_image 的 numpy 复刻: RGB->BGR, bottom-right pad 127, /255"""
    orig_h, orig_w = img_rgb.shape[:2]
    ratio = min(INPUT / orig_h, INPUT / orig_w)
    new_w, new_h = int(round(orig_w * ratio)), int(round(orig_h * ratio))
    bgr = np.ascontiguousarray(img_rgb[:, :, ::-1])
    resized = np.array(Image.fromarray(bgr).resize((new_w, new_h), Image.BILINEAR))
    padded = np.full((INPUT, INPUT, 3), PAD, dtype=np.uint8)
    padded[:new_h, :new_w] = resized          # bottom-right: 贴左上
    arr = padded.astype(np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None], ratio, (new_w, new_h)


def preprocess_wrong(img_rgb: np.ndarray):
    """对照组: 居中 pad + RGB(不翻转) —— 模拟现有 SDK 的 ultralytics 预处理"""
    orig_h, orig_w = img_rgb.shape[:2]
    ratio = min(INPUT / orig_h, INPUT / orig_w)
    new_w, new_h = int(round(orig_w * ratio)), int(round(orig_h * ratio))
    resized = np.array(Image.fromarray(img_rgb).resize((new_w, new_h), Image.BILINEAR))
    padded = np.full((INPUT, INPUT, 3), 114, dtype=np.uint8)
    ox, oy = (INPUT - new_w) // 2, (INPUT - new_h) // 2
    padded[oy:oy + new_h, ox:ox + new_w] = resized   # 居中
    arr = padded.astype(np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None], ratio, (new_w, new_h)


def load_gt(path: Path, w: int, h: int):
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) != 17:
            continue
        v = list(map(float, p[1:]))
        cx, cy, bw, bh = v[0] * w, v[1] * h, v[2] * w, v[3] * h
        box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
        kps = [(v[4 + i * 3] * w, v[5 + i * 3] * h) for i in range(4)]
        out.append((box, kps))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def run(img_path: Path, preprocess):
    img = np.array(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]
    x, ratio, (nw, nh) = preprocess(img)
    boxes, scores, kxy, kconf = sess.run(None, {"images": x.astype(np.float32)})
    b = boxes[0]; s = scores[0].reshape(-1); kxy = kxy[0]; kconf = kconf[0]
    print(f"\n--- {img_path.name} (orig {w}x{h}, resize {nw}x{nh}, ratio {ratio:.3f})")
    print(f"score range: min={s.min():.4f} max={s.max():.4f}  "
          f"kconf range: min={kconf.min():.4f} max={kconf.max():.4f}")
    # 坐标范围检查 (boxes 是否在 0..512 内 = 输入画布像素 xyxy)
    valid = s >= 0.25
    print(f"conf>=0.25: {valid.sum()} 个")
    gt = load_gt(ROOT / "labels" / "val" / (img_path.stem + ".txt"), w, h)
    print(f"GT: {len(gt)} 实例")
    for gi in np.argsort(-s)[:3]:
        bx = b[gi] / ratio            # bottom-right: 无 offset, 直接除 ratio
        kx = kxy[gi] / ratio
        print(f"  pred#{gi} score={s[gi]:.3f} box(canvas)={np.round(b[gi],1)} "
              f"box(orig)={np.round(bx,1)}")
        print(f"      kpts_xy(orig)={np.round(kx,1).tolist()}")
        print(f"      kpts_conf={np.round(kconf[gi],3).tolist()}")
        if gt:
            best = max(gt, key=lambda g: iou(bx, g[0]))
            print(f"      GT box(orig)={np.round(best[0],1)} IoU={iou(bx, best[0]):.3f}")
            print(f"      GT kpts(orig)={[[round(p,1) for p in k] for k in best[1]]}")


imgs = sorted((ROOT / "images" / "val").glob("*.jpg"))[:3]
for p in imgs:
    run(p, preprocess)        # 正确契约
print("\n===== 对照组: 居中pad+RGB (现有SDK的ultralytics式预处理) =====")
for p in imgs[:1]:
    run(p, preprocess_wrong)
