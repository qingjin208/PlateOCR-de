"""锚点替代方案原型 —— 用「量出来的字高」代替「假设的图高」。

背景（实测数据，见 badge_scale_anchor_probe.py）：
    真徽章暗块 / 图高   = 0.20 - 0.47   ← 现有下界 0.30 直接切掉其中 5/14
    文字粘连大块 / 图高 = 0.71 - 0.85   ← 现有上界 0.95 挡不住（落在窗口内）
    真徽章暗块 / 字高   = 0.26 - 0.60
    文字粘连大块 / 字高 = 0.99 - 1.02   ← 恒等于 1，因为粘连块高度 ≡ 字高

所以把锚点从「图像高度」换成「量出来的字高」后，粘连块会稳定地落在 1.0，
用 0.85 的上界就能确定性排除；下界 0.20 则放回被切掉的真实徽章。

本原型同时把闭运算半径 2 → 1（半径 2 会把章和相邻字母桥接成一块）。

用法：python Scripts/badge_anchor_prototype.py
输出：Scripts/cleaner_calibration_out/badge_anchor_proto.png（绿圈=新判据遮盖）
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from badge_false_positive_probe import (  # noqa: E402
    SRC, DST, imread_unicode, resize_to_h, circle_stats, gap_isolation,
)

DIA_MIN_FRAC = 0.20      # × 字高
DIA_MAX_FRAC = 0.85      # × 字高（粘连块 ≡ 1.00）
CLOSE_RADIUS = 1         # 原为 2
WHITE_MAX = 0.45
ISO_MAX = 0.15
MAX_BADGES = 2


def char_metrics(small: np.ndarray):
    """返回 (charH, band_top, band_bottom, text_left, text_right)。全部从图像内容量出来。"""
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    thr = max(60, float(np.median(gray)) * 0.62)
    dark = (gray < thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    keep = []
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if not (0.30 * h <= bh <= 0.85 * h):
            continue
        if not (0.20 <= bw / max(1, bh) <= 1.20):
            continue
        if area < 0.006 * w * h:
            continue
        keep.append((x, y, bw, bh))
    if not keep:
        return 0.0, 0, h - 1, 0, w - 1
    ch = float(np.median([k[3] for k in keep]))
    top = int(np.median([k[1] for k in keep]))
    bottom = int(np.median([k[1] + k[3] for k in keep]))
    left = int(min(k[0] for k in keep))
    right = int(max(k[0] + k[2] for k in keep))
    return ch, top, bottom, left, right


def detect(small: np.ndarray):
    h, w = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    ch, top, bottom, tleft, tright = char_metrics(small)
    if ch <= 0:
        return [], ch, (top, bottom, tleft, tright)
    min_dia, max_dia = DIA_MIN_FRAC * ch, DIA_MAX_FRAC * ch

    sat, val, hue = hsv[:, :, 1], hsv[:, :, 2], hsv[:, :, 0]
    blue = ((hue >= 95) & (hue <= 135) & (sat >= 50) & (val >= 30))
    cands = []

    colored = ((sat >= 70) & (val >= 60) & ~blue).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(colored, 8)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area >= 30:
            cands.append(("colored", x, y, bw, bh, area))

    dark = (gray < max(60, float(np.median(gray)) * 0.62)).astype(np.uint8)
    if CLOSE_RADIUS > 0:
        k = 2 * CLOSE_RADIUS + 1
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, _, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area >= 40:
            cands.append(("gray", x, y, bw, bh, area))

    scored = []
    for kind, x, y, bw, bh, area in cands:
        aspect = bw / max(1, bh)
        if kind == "colored":
            if not (0.6 <= aspect <= 1.7):
                continue
            if area / max(1, bw * bh) < 0.5:
                continue
        else:
            if not (0.55 <= aspect <= 1.5):
                continue
            if area / max(1, bw * bh) < 0.55:
                continue
        dia = max(bw, bh)
        if not (min_dia <= dia <= max_dia):
            continue
        cx, cy, r = x + bw / 2, y + bh / 2, dia / 2
        # 垂直：相对量出来的文字带，而不是图像高度
        if not (top - 0.25 * ch <= cy <= bottom + 0.25 * ch):
            continue
        # 水平：限制在文字区内部（原 0.24w 假设裁剪边界==车牌边界）
        if not (tleft + 0.25 * ch <= cx <= tright - 0.25 * ch):
            continue
        st_ = circle_stats(gray, cx, cy, r)
        if st_["white"] > WHITE_MAX:
            continue
        iso = 0.0 if kind == "colored" else gap_isolation(gray, cx, cy, r)
        if iso > ISO_MAX:
            continue
        scored.append((iso, st_["white"], kind, cx, cy, r))

    scored.sort()
    return scored[:MAX_BADGES], ch, (top, bottom, tleft, tright)


def main() -> int:
    files = sorted(SRC.glob("*.jpg"))
    rows = []
    total = 0
    for p in files:
        img = imread_unicode(p)
        if img is None:
            continue
        small = resize_to_h(img)
        found, ch, band = detect(small)
        total += len(found)
        print(f"{p.name:44s} charH={ch:5.0f} band={band[1]}-{band[2]} masks={len(found)} "
              + " ".join(f"[{k} x={c/small.shape[1]:.2f} dia={d:.0f} iso={i:.2f}]"
                         for i, _, k, c, _, d in [(f[0], f[1], f[2], f[3], f[4], f[5] * 2) for f in found]))
        vis = small.copy()
        for _, _, _, cx, cy, r in found:
            cv2.circle(vis, (int(cx), int(cy)), int(r * 1.15), (0, 255, 0), 1)
            cv2.circle(vis, (int(cx), int(cy)), 1, (0, 0, 255), -1)
        rows.append(vis)
    maxw = max(r.shape[1] for r in rows)
    canvas = np.full((sum(r.shape[0] + 8 for r in rows), maxw, 3), 35, dtype=np.uint8)
    y = 0
    for r in rows:
        canvas[y:y + r.shape[0], 0:r.shape[1]] = r
        y += r.shape[0] + 8
    out = DST / "badge_anchor_proto.png"
    cv2.imwrite(str(out), canvas)
    print(f"\ntotal masks = {total} over {len(rows)} plates; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
