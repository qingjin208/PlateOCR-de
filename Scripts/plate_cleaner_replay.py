#!/usr/bin/env python3
"""Replay the CURRENT PlateImageCleaner.detectSeals gates on real plate crops.

Mirrors Swift PlateImageCleaner.swift (probe height 128, HSV ranges, morph close
radius 2, 8-neighbour CC, every gate). Diagnostics only: prints the value and the
failing gates for each candidate. No decisions changed.

Updated 2026-11-17 to mirror the blue-seal channel + relaxed colored diameter
floor, both added after real-sample analysis (see .workbuddy/memory/2026-11-17.md).
"""
import sys
from PIL import Image, ImageDraw
import numpy as np

W_HUE = (190.0, 270.0)
TARGET_H = 128
GEOM = dict(gray_min_dia=0.30, colored_min_dia=0.12, blue_min_dia=0.12,
            max_dia=0.95, aspect_lo=0.6, aspect_hi=1.7,
            fill_lo=0.5, gray_aspect_lo=0.55, gray_aspect_hi=1.5,
            gray_fill_lo=0.55, min_area_col=30, min_area_gray=40)


def to_probe(path, box):
    im = Image.open(path).convert("RGB").crop(box)
    scale = TARGET_H / im.height
    im = im.resize((max(1, round(im.width * scale)), TARGET_H), Image.LANCZOS)
    return np.asarray(im).astype(np.float64)  # top-row-first, HxWx3


def hsv(px):
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn
    h = np.zeros_like(mx)
    nz = delta > 0
    idx = nz & (mx == r); h[idx] = 60 * (((g - b)[idx] / delta[idx]) % 6)
    idx = nz & (mx == g) & ~(mx == r); h[idx] = 60 * (((b - r)[idx] / delta[idx]) + 2)
    idx = nz & (mx == g) & (mx == r); h[idx] = 60 * (((b - r)[idx] / delta[idx]) + 2)
    idx = nz & (mx == b); h[idx] = 60 * (((r - g)[idx] / delta[idx]) + 4)
    h[h < 0] += 360
    s = np.where(mx == 0, 0, delta / np.maximum(mx, 1e-9) * 255)
    return h, s, mx


def colored_mask(px, sat_min=70, val_min=60):
    h, s, v = hsv(px)
    blue = (h >= W_HUE[0]) & (h <= W_HUE[1])
    return (s >= sat_min) & (v >= val_min) & (~blue)


def blue_mask(px, sat_min=70, val_min=60):
    h, s, v = hsv(px)
    return (s >= sat_min) & (v >= val_min) & (h >= W_HUE[0]) & (h <= W_HUE[1])


def dark_mask(px, radius=2):
    gray = px.mean(axis=2)
    median = np.median(gray)
    thr = max(60, median * 0.62)
    m = gray < thr
    return morph_close(m, radius)


def morph_close(mask, radius):
    return erode(dilate(mask, radius), radius)


def dilate(mask, radius):
    H, W = mask.shape
    out = mask.copy()
    ys, xs = np.nonzero(mask)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            y0, y1 = max(0, dy), min(H, H + dy)
            x0, x1 = max(0, dx), min(W, W + dx)
            out[y0:y1, x0:x1] |= mask[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def erode(mask, radius):
    H, W = mask.shape
    out = mask.copy()
    for y in range(H):
        for x in range(W):
            if not mask[y, x]:
                continue
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy > radius * radius:
                        continue
                    yy, xx = y + dy, x + dx
                    if yy < 0 or yy >= H or xx < 0 or xx >= W or not mask[yy, xx]:
                        out[y, x] = False
                        break
                else:
                    continue
                break
    return out


def connected(mask, min_area):
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sy in range(H):
        for sx in range(W):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]; seen[sy, sx] = True
            area = 0; minx, maxx, miny, maxy = W, -1, H, -1
            while stack:
                y, x = stack.pop(); area += 1
                minx, maxx = min(minx, x), max(maxx, x)
                miny, maxy = min(miny, y), max(maxy, y)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                            seen[yy, xx] = True; stack.append((yy, xx))
            if area >= min_area:
                comps.append(dict(area=area, minx=minx, maxx=maxx, miny=miny, maxy=maxy))
    return comps


def interior_white(px, cx, cy, rad):
    H, W = px.shape[:2]
    gray = px.mean(axis=2)
    y0, y1 = max(0, int(cy - rad)), min(H - 1, int(cy + rad))
    x0, x1 = max(0, int(cx - rad)), min(W - 1, int(cx + rad))
    white = total = 0
    for y in range(y0, max(y0, y1) + 1):
        for x in range(x0, max(x0, x1) + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= rad * rad:
                total += 1
                if gray[y, x] > 190:
                    white += 1
    return white / total if total else 1.0


def gap_isolation(px, cx, cy, r):
    H, W = px.shape[:2]
    gray = px.mean(axis=2)
    half = max(2, int(r * 0.25))
    total = 0.0
    for sx in (cx - r * 1.45, cx + r * 1.45):
        x = int(round(sx))
        if x - half < 0 or x + half >= W:
            return 1.0
        strip = gray[:, x - half:x + half + 1]
        total += float((strip < 130).sum()) / (H * (2 * half + 1))
    return total


def replay(path, box, tag):
    px = to_probe(path, box)
    H, W = px.shape[:2]
    h = H; w = W
    gray_dia, max_dia = GEOM['gray_min_dia'] * h, GEOM['max_dia'] * h
    col_dia = GEOM['colored_min_dia'] * h
    blue_dia = GEOM['blue_min_dia'] * h
    print(f"\n===== 样本{tag}: probe {w}x{h}  grayDia>={gray_dia:.0f} coloredDia>={col_dia:.0f} maxDia={max_dia:.0f} =====")

    # colored (non-blue) + blue seals — relaxed diameter, no isolation gate.
    # Assume a nominal EU-strip right edge of 15% width for the blue centerX gate
    # (Swift uses the detected euStripRightEdge; conservative proxy here).
    for kind, mask, lo_dia in [('colored', colored_mask(px), col_dia),
                               ('blue', blue_mask(px), blue_dia)]:
        for c in connected(mask, GEOM['min_area_col']):
            bw, bh = c['maxx'] - c['minx'] + 1, c['maxy'] - c['miny'] + 1
            aspect = bw / max(1, bh); fill = c['area'] / max(1, bw * bh); dia = max(bw, bh)
            cx = c['minx'] + bw / 2; cy = c['miny'] + bh / 2
            failed = []
            if aspect < GEOM['aspect_lo'] or aspect > GEOM['aspect_hi']: failed.append('aspect')
            if fill < GEOM['fill_lo']: failed.append('fill')
            if dia < lo_dia or dia > max_dia: failed.append('diameter')
            if cx < max(0.15 * w + bw / 2, 0.24 * w): failed.append('centerX')
            white = interior_white(px, cx, cy, dia / 2 * 0.6)
            if white > 0.45: failed.append('interiorWhite')
            if not (0 <= cy <= 0.9 * h): failed.append('centerY')
            print(f"  [{kind:7}] cx={cx:.0f} cy={cy:.0f} dia={dia:.0f} aspect={aspect:.2f} "
                  f"fill={fill:.2f} white={white:.2f} -> {'ACCEPT' if not failed else 'FAIL:' + '+'.join(failed)}")

    # gray seals — strict, unchanged
    dmask = dark_mask(px)
    for c in connected(dmask, GEOM['min_area_gray']):
        bw, bh = c['maxx'] - c['minx'] + 1, c['maxy'] - c['miny'] + 1
        aspect = bw / max(1, bh); dia = max(bw, bh); fill = c['area'] / max(1, bw * bh)
        cx = c['minx'] + bw / 2; cy = c['miny'] + bh / 2
        failed = []
        if aspect < GEOM['gray_aspect_lo'] or aspect > GEOM['gray_aspect_hi']: failed.append('aspect')
        if dia < gray_dia or dia > max_dia: failed.append('diameter')
        if fill < GEOM['gray_fill_lo']: failed.append('fill')
        if not (0.18 * h <= cy <= 0.86 * h): failed.append('centerY')
        white = interior_white(px, cx, cy, dia / 2 * 0.6)
        if white > 0.45: failed.append('interiorWhite')
        iso = gap_isolation(px, cx, cy, dia / 2)
        if iso > 0.15: failed.append('isolation')
        # Only print blobs roughly seal-like in size (avoid flooding with characters)
        if dia >= gray_dia * 0.7:
            print(f"  [gray]    cx={cx:.0f} cy={cy:.0f} dia={dia:.0f} aspect={aspect:.2f} "
                  f"fill={fill:.2f} white={white:.2f} iso={iso:.2f} -> {'ACCEPT' if not failed else 'FAIL:' + '+'.join(failed)}")

    # save probe + both seal masks + dark mask for viewing
    base = f'E:/Git/ITLogica/PlateOCR-de/.workbuddy/replay_{tag}'
    Image.fromarray(px.astype(np.uint8)).save(base + '_probe.png')
    Image.fromarray((colored_mask(px) * 255).astype(np.uint8)).save(base + '_colored.png')
    Image.fromarray((blue_mask(px) * 255).astype(np.uint8)).save(base + '_blue.png')
    Image.fromarray((dmask * 255).astype(np.uint8)).save(base + '_dark.png')


if __name__ == '__main__':
    BASE = 'C:/Users/Neil.Zhen/.workbuddy/clipboard-images/'
    # tight plate boxes (original image coords, from visual inspection)
    replay(BASE + 'clipboard-2026-11-17T01-49-23-173Z-0a809065.jpg', (762, 1473, 913, 1512), 1)
    replay(BASE + 'clipboard-2026-11-17T01-49-23-176Z-365e5e73.jpg', (299, 524, 494, 631), 2)
    replay(BASE + 'clipboard-2026-11-17T01-49-23-180Z-6eefbd3d.jpg', (313, 807, 554, 993), 3)
