# -*- coding: utf-8 -*-
"""Swift 移动端 LibreYOLO 管线的离线复刻验证 —— 找"PC 能检到、手机检不到"的原因。

复刻的是 OCR/LicensePlateOCR/LicensePlateSDK 里的 ONNXPoseLicensePlateDetector +
YOLOOutputDecoder 全链路，逐条对应 Swift 实现：

  1. letterboxedUpright(size:512, flipY:true, bottomRightPad:true, padValue:127)
     —— BGRA 缓冲按 B,G,R 读出 → BGR 通道（feedsRGB=false 的 else 分支）
  2. inputTensorData → NCHW float32 /255
  3. decodeLibreYOLO: scores>=0.25, canvas xyxy 框, cornerOrder 质心重排
  4. suppress (NMS IoU 0.7)
  5. remap: /scale, padX=padY=0

如果这条链路在 PC 上能对 input_folder 的图检出车牌，而真机不能，
则问题不在契约本身，而在真机运行时（模型文件/ORT 会话/图像来源）；
如果这条链路也检不到，则契约有 bug，可在此定位。
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import cv2

BUNDLED = Path(r"E:/Git/ITLogica/PlateOCR-de/OCR/LicensePlateOCR/LicensePlateSDK/Resources/De-plate-pose.onnx")
SRC = Path(r"E:/Git/Neil/Project/pythonCode/模型测试/车牌关键点检测模型-TEST/input_folder/image")
INPUT = 512
PAD = 127
CONF = 0.25
IOU = 0.7

sess = ort.InferenceSession(str(BUNDLED), providers=["CPUExecutionProvider"])
print("inputs :", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])
IN_NAME = sess.get_inputs()[0].name
IN_SHAPE = sess.get_inputs()[0].shape

# ---- 输入张量布局检查：NCHW 还是 NHWC ----
nchw = IN_SHAPE == [1, 3, INPUT, INPUT]
nhwc = IN_SHAPE == [1, INPUT, INPUT, INPUT and INPUT] or (len(IN_SHAPE) == 4 and IN_SHAPE[3] == 3)
print(f"layout: {'NCHW' if nchw else ('NHWC' if nhwc else '???')}  shape={IN_SHAPE}")


def preprocess_mobile(img_rgb):
    """Swift letterboxedUpright + inputTensorData 的逐字节复刻（BGR/NCHW）。"""
    h, w = img_rgb.shape[:2]
    scale = min(INPUT / w, INPUT / h)
    rw, rh = int(round(w * scale)), int(round(h * scale))
    # Swift: BGRA 缓冲, CGContext draw(self, in: CGRect(0,0,rw,rh)) 顶行在前
    resized = cv2.resize(img_rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    canvas = np.full((INPUT, INPUT, 3), PAD, dtype=np.uint8)
    canvas[:rh, :rw] = bgr            # bottom-right pad → 贴左上
    arr = canvas.astype(np.float32) / 255.0
    x = arr.transpose(2, 0, 1)[None].astype(np.float32)
    return x, scale


def corner_order(pts):
    mean_x = np.mean([p[0] for p in pts]); mean_y = np.mean([p[1] for p in pts])
    slots = [-1, -1, -1, -1]
    for i, (x, y) in enumerate(pts):
        s = (0 if x < mean_x else 1) + (0 if y < mean_y else 2) * 0
        if x < mean_x and y < mean_y: slots[0] = i
        elif x >= mean_x and y < mean_y: slots[1] = i
        elif x >= mean_x and y >= mean_y: slots[2] = i
        else: slots[3] = i
    return slots if -1 not in slots else None


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0


def nms(det):
    det = sorted(det, key=lambda d: -d["conf"])
    kept = []
    for d in det:
        if not any(d["label"] == k["label"] and iou(d["box"], k["box"]) > IOU for k in kept):
            kept.append(d)
    return kept


def run_image(p):
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [skip] 无法读取 {p.name}")
        return
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Swift 端拿到的 CGImage 是 RGB
    x, scale = preprocess_mobile(rgb)
    boxes, scores, kxy, kconf = sess.run(None, {IN_NAME: x})
    s = scores.reshape(-1)
    n_kpt = 4
    dets = []
    for a in range(len(s)):
        if s[a] < CONF:
            continue
        x1, y1, x2, y2 = boxes[0][a]
        pts = [(kxy[0][a][k][0], kxy[0][a][k][1]) for k in range(n_kpt)]
        order = corner_order(pts)
        pts_o = [pts[i] for i in order] if order else pts
        dets.append({"conf": float(s[a]), "label": "plate",
                     "box": (x1/scale/w, y1/scale/h, x2/scale/w, y2/scale/h),
                     "kpts": [(px/scale/w, py/scale/h) for px, py in pts_o]})
    kept = nms(dets)
    print(f"\n{p.name}  ({w}x{h})  原始过阈 {len(dets)}  NMS 后 {len(kept)}")
    for d in kept[:5]:
        bx = d["box"]
        print(f"  conf={d['conf']:.3f} box_norm=({bx[0]:.3f},{bx[1]:.3f},{bx[2]:.3f},{bx[3]:.3f}) "
              f"kpts={[(round(px,3), round(py,3)) for px, py in d['kpts']]}")


imgs = sorted(SRC.glob("*.jpg"))
print(f"\n===== 用 打包模型 + 移动端契约 跑 {len(imgs)} 张 input_folder 图 =====")
for p in imgs:
    run_image(p)
