# -*- coding: utf-8 -*-
"""
车牌矫正图 OCR 前清理标定脚本（Windows/PC 端标定，结论移植到 Swift）
输入：libreyolo pose 模型输出的车牌裁剪图
步骤：①上采样到目标高度 ②EU 蓝条自适应遮盖 ③徽章圆形自适应遮盖
输出：四联对比图（原图 | 上采样 | +EU遮盖 | +徽章遮盖）与逐图诊断
"""
import cv2
import numpy as np
from pathlib import Path

SRC = Path(r"E:\Git\Neil\Project\pythonCode\模型测试\车牌关键点检测模型-TEST\liberyolo\output_folder\image")
DST = Path(r"E:\Git\ITLogica\PlateOCR-de\Scripts\cleaner_calibration_out")

# ---------- 参数（标定后移植 Swift） ----------
TARGET_HEIGHT = 128          # 上采样目标高度
EU_MAX_FRACTION = 0.22       # EU 条最宽不超过牌宽的比例
EU_BLUE_H = (95, 135)        # OpenCV H 0-180
EU_BLUE_S_MIN = 50
EU_BLUE_V_MIN = 30
EU_COL_RATIO = 0.35          # 列内蓝色像素占比阈值

BADGE_MAX_DIAMETER_FRAC = 0.95   # 徽章直径不超过图高比例
BADGE_MIN_DIAMETER_FRAC = 0.30
BADGE_X_RANGE = (0.15, 0.72)     # 徽章只在行政区码后间隙：收紧水平范围
BADGE_CY_RANGE = (0.18, 0.86)    # 垂直中心范围（叠章可上下偏）
GAP_STRIP_DARK_MAX = 0.15        # 圆两侧竖条带暗像素占比上限（两侧必须空白）
GAP_STRIP_OFFSET = 1.45          # 条带中心距圆心 = r * 该系数
GAP_STRIP_WIDTH_FRAC = 0.5       # 条带宽度 = r * 该系数
INTERIOR_WHITE_MAX = 0.55        # 圆内近白占比上限（超过=字符 O 的白底，不遮）
COLORED_S_MIN = 70               # 彩色徽章饱和度下限


def upscale(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(1.0, TARGET_HEIGHT / h)
    if scale <= 1.0:
        return img.copy()
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LANCZOS4)


def detect_eu_strip(img: np.ndarray) -> int:
    """返回 EU 条右边界 x（像素）；0 表示未检测到。基于左缘连续蓝色列。"""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = ((hsv[:, :, 0] >= EU_BLUE_H[0]) & (hsv[:, :, 0] <= EU_BLUE_H[1]) &
            (hsv[:, :, 1] >= EU_BLUE_S_MIN) & (hsv[:, :, 2] >= EU_BLUE_V_MIN))
    limit = max(1, int(w * EU_MAX_FRACTION))
    edge = 0
    run_broken = False
    for x in range(limit):
        ratio = blue[:, x].mean()
        if ratio >= EU_COL_RATIO and not run_broken:
            edge = x + 1
        elif x < 3 and ratio < EU_COL_RATIO:
            break  # 最左侧就不是蓝条
        elif ratio < EU_COL_RATIO:
            run_broken = True
    return edge


def interior_white_fraction(gray: np.ndarray, cx: float, cy: float, r: float) -> float:
    """圆盘内近白像素占比（字符 O 内部是白底，徽章内部是图案）。"""
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    if disc.sum() == 0:
        return 1.0
    white = gray[disc] > 190
    return float(white.mean())


def gap_isolation(gray: np.ndarray, cx: float, cy: float, r: float) -> float:
    """孤立度：圆左右两侧竖条带的暗像素占比之和。徽章孤立在空白间隙 → 低；
    字符紧邻其他字符 → 高。返回 (left_dark + right_dark)。"""
    h, w = gray.shape
    half_w = max(2, int(r * GAP_STRIP_WIDTH_FRAC / 2))
    total = 0.0
    for sx in (cx - r * GAP_STRIP_OFFSET, cx + r * GAP_STRIP_OFFSET):
        x = int(round(sx))
        if x - half_w < 0 or x + half_w >= w:
            total += 1.0  # 出界视为不孤立
            continue
        strip = gray[:, x - half_w:x + half_w]
        total += float((strip < 130).mean())
    return total


def circle_stats(gray: np.ndarray, cx: float, cy: float, r: float) -> dict:
    """候选圆诊断：不同内径的白底占比、暗像素占比。"""
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    out = {}
    for frac in (0.4, 0.6, 0.8):
        disc = d2 <= (r * frac) ** 2
        if disc.sum() == 0:
            out[f"white_{frac}"] = 1.0
        else:
            out[f"white_{frac}"] = float((gray[disc] > 190).mean())
    ring = (d2 <= r ** 2) & (d2 >= (r * 0.7) ** 2)
    out["ring_dark"] = float((gray[ring] < 120).mean()) if ring.sum() else 0.0
    disc = d2 <= (r * 0.8) ** 2
    out["disc_dark"] = float((gray[disc] < 160).mean()) if disc.sum() else 0.0
    return out


def detect_badges(img: np.ndarray) -> list:
    """返回 [(cx, cy, r)]，徽章圆列表。检测不到返回空。"""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    x0, x1 = int(w * BADGE_X_RANGE[0]), int(w * BADGE_X_RANGE[1])
    min_r = BADGE_MIN_DIAMETER_FRAC * h / 2
    max_r = BADGE_MAX_DIAMETER_FRAC * h / 2
    found = []
    cands = []

    # --- 1) 彩色徽章：高饱和度非蓝色圆形色块 ---
    sat = hsv[:, :, 1].astype(np.int32)
    hue = hsv[:, :, 0]
    val = hsv[:, :, 2]
    colored = ((sat >= COLORED_S_MIN) & (val >= 60) &
               ~((hue >= EU_BLUE_H[0]) & (hue <= EU_BLUE_H[1])))
    colored[:, :x0] = False
    colored[:, x1:] = False
    contours, _ = cv2.findContours(colored.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 30:
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (per * per) if per > 0 else 0
        (bx, by), r = cv2.minEnclosingCircle(c)
        bw, bh = 2 * r, 2 * r
        r = max(bw, bh) / 2
        if circ >= 0.65 and min_r <= r <= max_r and 0.6 <= bw / max(1, bh) <= 1.7:
            cands.append(("colored", bx, by, r))

    # --- 2) 灰色徽章：暗色连通域 + 实心度（Swift 可移植，无 Hough 依赖） ---
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    # 自适应暗阈值：低于全场中位数 - 余量
    median = float(np.median(blur))
    dark = (blur < max(60, median * 0.62)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 40:
            continue
        if not (0.55 <= bw / max(1, bh) <= 1.5):
            continue
        dia = max(bw, bh)
        if not (BADGE_MIN_DIAMETER_FRAC * h <= dia <= BADGE_MAX_DIAMETER_FRAC * h):
            continue
        fill = area / max(1, bw * bh)
        if fill < 0.55:  # 实心度：字符笔画稀疏，徽章圆盘致密
            continue
        cx, cy = x + bw / 2, y + bh / 2
        cands.append(("cc", cx, cy, dia / 2))

    # 诊断输出
    for kind, cx, cy, r in cands:
        st = circle_stats(gray, cx, cy, r)
        iso = gap_isolation(gray, cx, cy, r)
        print(f"    cand {kind} c=({cx:.0f},{cy:.0f}) r={r:.0f} "
              f"white@0.6={st['white_0.6']:.2f} disc_dark={st['disc_dark']:.2f} iso={iso:.2f}")

    # 判别：彩色路径本质安全（字符灰度无饱和度），只做几何+内部检查；
    # 灰色 CC 路径需全套约束（x≥0.24w + 孤立度 + 内部非白底）。
    scored = []
    for kind, cx, cy, r in cands:
        if cx < 0.24 * w:
            continue
        if kind == "colored":
            if not (0.0 <= cy <= 0.9 * h):
                continue
            iso, wf = 0.0, circle_stats(gray, cx, cy, r)["white_0.6"]
        else:
            if not (BADGE_CY_RANGE[0] * h <= cy <= BADGE_CY_RANGE[1] * h):
                continue
            iso = gap_isolation(gray, cx, cy, r)
            wf = circle_stats(gray, cx, cy, r)["white_0.6"]
        scored.append((iso, wf, kind, cx, cy, r))
    scored.sort()  # 孤立度最低（最像间隙中的徽章）优先
    for iso, wf, kind, cx, cy, r in scored[:2]:  # 每牌最多 2 枚（可叠放）
        if iso <= 0.15 and wf <= 0.45:
            found.append((cx, cy, r * 1.15))
    return found


def clean(img: np.ndarray):
    """返回 (清理图, eu_edge, badges, 上采样图)"""
    up = upscale(img)
    out = up.copy()
    eu = detect_eu_strip(out)
    if eu > 0:
        out[:, :eu] = 255
    badges = detect_badges(out)
    for cx, cy, r in badges:
        cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(r)), (255, 255, 255), -1)
    return out, eu, badges, up


def imread_unicode(path: Path) -> np.ndarray | None:
    """cv2.imread 在 Windows 非 ASCII 路径下失败，改用 imdecode。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def main():
    DST.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in SRC.glob("*.jpg"))
    report = []
    tiles = []
    for p in files:
        img = imread_unicode(p)
        if img is None:
            continue
        cleaned, eu, badges, up = clean(img)
        report.append(f"{p.name}: eu_edge={eu} badges={[(int(c), int(y), int(r)) for c, y, r in badges]}")
        # 四联图
        h_target = 128
        def fit(im):
            s = h_target / im.shape[0]
            return cv2.resize(im, (round(im.shape[1] * s), h_target), interpolation=cv2.INTER_NEAREST) if s != 1 else im
        row = np.hstack([fit(img), fit(up), fit(cleaned)])
        tiles.append((p.name, row))

    # 拼总览图
    maxw = max(r.shape[1] for _, r in tiles) + 160
    H = sum(r.shape[0] + 6 for _, r in tiles)
    canvas = np.full((H, maxw, 3), 40, dtype=np.uint8)
    y = 0
    for name, r in tiles:
        canvas[y:y + r.shape[0], 160:160 + r.shape[1]] = r
        cv2.putText(canvas, name[:26], (2, r.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        y += r.shape[0] + 6
    cv2.imwrite(str(DST / "overview.png"), canvas)
    print("\n".join(report))
    print(f"overview -> {DST / 'overview.png'}")


if __name__ == "__main__":
    main()
