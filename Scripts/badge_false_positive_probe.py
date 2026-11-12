"""徽章误检探针 —— 逐条复刻 Swift PlateImageCleaner.detectSeals 的判据。

目的：回答「字符 8 / 0 / O 会不会被当成徽章涂白」。
方法：在 128px 高的尺度上跑与 Swift 完全一致的连通域 + 门控，
      把所有候选的度量值打印出来（不只是最终通过的），
      这样能看出字符离「通过」还有多远。

与 Swift 的对应关系（PlateImageCleaner.swift）：
  - 尺度：无条件缩放到高 128（与上一轮修正后的 PixelBuffer(image:targetHeight:) 一致）
  - 彩色路径：sat>=70, val>=60, 非蓝(190-270°) → 连通域
              aspect 0.6-1.7, fill>=0.5, dia 0.30-0.95*h, area>=30
  - 灰路径：  dark = gray < max(60, median*0.62), 5x5 椭圆闭运算 → 连通域
              aspect 0.55-1.5, fill>=0.55, dia 0.30-0.95*h, area>=40
  - 门控：    cx>=0.24*w; 圆内 0.6r 白色占比<=0.45
              彩色: 0<=cy<=0.9h;  灰: 0.18h<=cy<=0.86h 且 gapIsolation<=0.15
  - 排序：    (iso, white) 升序，取前 2

用法：python Scripts/badge_false_positive_probe.py
输出：Scripts/cleaner_calibration_out/badge_probe.png（带圈标注的对照图）
      stdout 打印每张图的候选表
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

SRC = Path(r"E:\Git\Neil\Project\pythonCode\模型测试\车牌关键点检测模型-TEST\liberyolo\output_folder\image")
DST = Path(r"E:\Git\ITLogica\PlateOCR-de\Scripts\cleaner_calibration_out")

TARGET_H = 128
BADGE_MIN_DIA = 0.30
BADGE_MAX_DIA = 0.95
MIN_CX_FRAC = 0.24
INTERIOR_WHITE_MAX = 0.45
ISO_MAX = 0.15
MAX_BADGES = 2


def imread_unicode(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def resize_to_h(img: np.ndarray, h: int = TARGET_H) -> np.ndarray:
    s = h / img.shape[0]
    return cv2.resize(img, (max(1, round(img.shape[1] * s)), h),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def circle_stats(gray: np.ndarray, cx: float, cy: float, r: float) -> dict:
    h, w = gray.shape
    y0, y1 = max(0, int(cy - r) - 1), min(h, int(cy + r) + 2)
    x0, x1 = max(0, int(cx - r) - 1), min(w, int(cx + r) + 2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    patch = gray[y0:y1, x0:x1]
    disc = d2 <= (r * 0.6) ** 2
    white = float((patch[disc] > 190).mean()) if disc.any() else 1.0
    ring = (d2 <= r ** 2) & (d2 >= (r * 0.7) ** 2)
    ring_dark = float((patch[ring] < 120).mean()) if ring.any() else 0.0
    return {"white": white, "ring_dark": ring_dark}


def gap_isolation(gray: np.ndarray, cx: float, cy: float, r: float) -> float:
    """与 Swift 一致：±1.45r 两条竖带的暗像素占比之和（0-2，Swift 直接比较 <=0.15）。"""
    h, w = gray.shape
    half = max(2, int(r * 0.25))
    total = 0.0
    for sx in (cx - r * 1.45, cx + r * 1.45):
        x = int(round(sx))
        if x - half < 0 or x + half >= w:
            return 2.0
        strip = gray[:, x - half:x + half + 1]
        total += float((strip < 130).mean())
    return total


def candidates(img: np.ndarray) -> list[dict]:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    min_dia, max_dia = BADGE_MIN_DIA * h, BADGE_MAX_DIA * h
    out: list[dict] = []

    sat, val, hue = hsv[:, :, 1], hsv[:, :, 2], hsv[:, :, 0]
    blue = ((hue >= 95) & (hue <= 135) & (sat >= 50) & (val >= 30))
    colored = ((sat >= 70) & (val >= 60) & ~blue).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(colored, 8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 30:
            continue
        out.append({"kind": "colored", "bbox": (x, y, bw, bh), "area": int(area),
                    "cx": x + bw / 2, "cy": y + bh / 2, "dia": max(bw, bh)})

    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    dark = (blur < max(60, float(np.median(blur)) * 0.62)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(dark, 8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 40:
            continue
        out.append({"kind": "gray", "bbox": (x, y, bw, bh), "area": int(area),
                    "cx": x + bw / 2, "cy": y + bh / 2, "dia": max(bw, bh)})

    for c in out:
        bw, bh = c["bbox"][2], c["bbox"][3]
        c["aspect"] = bw / max(1, bh)
        c["fill"] = c["area"] / max(1, bw * bh)
        c["aspect_ok"] = (0.6 <= c["aspect"] <= 1.7) if c["kind"] == "colored" else (0.55 <= c["aspect"] <= 1.5)
        c["fill_ok"] = c["fill"] >= (0.5 if c["kind"] == "colored" else 0.55)
        c["dia_ok"] = min_dia <= c["dia"] <= max_dia
        st = circle_stats(gray, c["cx"], c["cy"], c["dia"] / 2)
        c["white"] = st["white"]
        c["ring_dark"] = st["ring_dark"]
        c["iso"] = gap_isolation(gray, c["cx"], c["cy"], c["dia"] / 2)
        c["geo_ok"] = c["aspect_ok"] and c["fill_ok"] and c["dia_ok"]
        c["x_ok"] = c["cx"] >= MIN_CX_FRAC * w
        c["white_ok"] = c["white"] <= INTERIOR_WHITE_MAX
        c["cy_ok"] = (0 <= c["cy"] <= 0.9 * h) if c["kind"] == "colored" else (0.18 * h <= c["cy"] <= 0.86 * h)
        c["iso_ok"] = True if c["kind"] == "colored" else c["iso"] <= ISO_MAX
        c["passes"] = c["geo_ok"] and c["x_ok"] and c["white_ok"] and c["cy_ok"] and c["iso_ok"]
        c["x_frac"] = c["cx"] / w
    return sorted(out, key=lambda c: (c["iso"], c["white"]))


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*.jpg"))
    if not files:
        print(f"no samples under {SRC}")
        return 1

    rows = []
    for p in files:
        img = imread_unicode(p)
        if img is None:
            continue
        small = resize_to_h(img)
        cands = candidates(small)
        passing = [c for c in cands if c["passes"]][:MAX_BADGES]
        # 近失候选：只差一项，用来量化「字符离被误遮有多近」
        near = [c for c in cands if not c["passes"] and
                sum([c["aspect_ok"], c["fill_ok"], c["dia_ok"], c["white_ok"], c["cy_ok"], c["iso_ok"]]) >= 5][:3]

        print(f"\n=== {p.name}  ({small.shape[1]}x{small.shape[0]}) ===")
        for tag, group in (("MASK", passing), ("NEAR", near)):
            for c in group:
                print(f"  {tag} {c['kind']:7s} x={c['x_frac']:.2f}w cy={c['cy']:.0f} "
                      f"dia={c['dia']:.0f} aspect={c['aspect']:.2f}({'ok' if c['aspect_ok'] else 'NO'}) "
                      f"fill={c['fill']:.2f}({'ok' if c['fill_ok'] else 'NO'}) "
                      f"white={c['white']:.2f} iso={c['iso']:.2f}")

        vis = small.copy()
        for c in near:
            cv2.rectangle(vis, (c["bbox"][0], c["bbox"][1]),
                          (c["bbox"][0] + c["bbox"][2], c["bbox"][1] + c["bbox"][3]), (0, 165, 255), 1)
        for c in passing:
            cv2.circle(vis, (int(round(c["cx"])), int(round(c["cy"]))),
                       int(round(c["dia"] / 2 * 1.15)), (0, 255, 0), 1)
            cv2.circle(vis, (int(round(c["cx"])), int(round(c["cy"]))), 1, (0, 0, 255), -1)
        rows.append(vis)

    maxw = max(r.shape[1] for r in rows)
    canvas = np.full((sum(r.shape[0] + 8 for r in rows), maxw, 3), 35, dtype=np.uint8)
    y = 0
    for r in rows:
        canvas[y:y + r.shape[0], 0:r.shape[1]] = r
        y += r.shape[0] + 8
    out = DST / "badge_probe.png"
    cv2.imwrite(str(out), canvas)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
