"""徽章尺度锚点探针 —— 证明「按图像高度取比例」为什么会失效，并验证替代锚点。

当前 Swift 判据（PlateImageCleaner.swift）：
    minDia = 0.30 * 图像高度,  maxDia = 0.95 * 图像高度
    垂直范围 cy ∈ [0.18, 0.86] * 图像高度

它隐含一个假设：**矫正图高度 == 车牌高度**。
但矫正图来自 4 个关键点围成的四边形 —— 关键点略外扩、车牌带边框、
或检测框把邻车/背景包进来时，图像高度就不再等于车牌高度，所有比例随之漂移。

本脚本做两件事：
 1. 量出「真实徽章直径 / 图像高度」的分布，看 0.30–0.95 这个区间落在哪里；
 2. 换一个自标定锚点：**文字行高**（从行投影量出来，不依赖裁剪松紧），
    报告「徽章直径 / 文字行高」的分布，看它是否更稳定。

用法：python Scripts/badge_scale_anchor_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from badge_false_positive_probe import (  # noqa: E402
    SRC, imread_unicode, resize_to_h, candidates, TARGET_H,
)

# 人工标注：哪些样本里看得见徽章（来自 badge_probe.png 的目视核对）
HAS_VISIBLE_SEAL = {
    "de_vehicle_0400_7137193_plate1.jpg": True,
    "de_vehicle_0406_58886794_plate1.jpg": True,
    "de_vehicle_0407_17970867_plate1.jpg": True,
    "de_vehicle_0424_19081357_plate1.jpg": True,
    "de_vehicle_0431_77529554_plate1.jpg": True,
    "de_vehicle_0444_28603475_plate1.jpg": True,
    "de_vehicle_0451_8712051_plate1.jpg": True,
    "de_vehicle_0454_133683064_plate1.jpg": True,
    "de_vehicle_0465_11809336_plate1.jpg": True,
    "de_vehicle_0466_35913777_plate1.jpg": True,
    "de_vehicle_0484_164009375_plate1.jpg": True,
    "de_vehicle_0494_163671971_plate1.jpg": True,
    "Snipaste_2026-10-20_08-39-53_plate1.jpg": True,
    "Snipaste_2026-10-20_08-39-53_plate2.jpg": True,
}


def text_band(img: np.ndarray) -> tuple[int, int, int]:
    """返回 (band_top, band_bottom, height)。用行投影：暗像素数超过峰值 25% 的行算文字行。

    这个量不依赖裁剪松紧 —— 背景再多也不会把文字行撑高。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    thr = max(60, float(np.median(gray)) * 0.62)
    dark = (gray < thr).astype(np.uint8)
    rows = dark.sum(axis=1).astype(float)
    peak = rows.max() if rows.max() > 0 else 1.0
    idx = np.where(rows > peak * 0.25)[0]
    if len(idx) == 0:
        return 0, h - 1, h
    return int(idx[0]), int(idx[-1]), int(idx[-1] - idx[0] + 1)


def main() -> int:
    files = sorted(SRC.glob("*.jpg"))
    if not files:
        print(f"no samples under {SRC}")
        return 1

    print(f"{'sample':44s} {'imH':>4s} {'band':>10s} {'Htext':>5s}  "
          f"{'seal dia':>8s} {'dia/imH':>7s} {'dia/Htext':>9s}  verdict")
    dia_by_img: list[float] = []
    dia_by_text: list[float] = []
    merged_by_img: list[float] = []
    merged_by_text: list[float] = []

    for p in files:
        img = imread_unicode(p)
        if img is None:
            continue
        small = resize_to_h(img)
        h = small.shape[0]
        top, bottom, htext = text_band(small)

        cands = candidates(small)
        passing = [c for c in cands if c["passes"]]
        # 最像徽章的那个候选（排序已按 iso 升序）；没有通过的就看 iso 最低的
        seal = passing[0] if passing else (cands[0] if cands else None)
        verdict = []
        if seal is None:
            print(f"{p.name:44s} {h:4d} {top:4d}-{bottom:<4d} {htext:5d}  (no candidate)")
            continue
        r_img, r_txt = seal["dia"] / h, seal["dia"] / htext
        dia_by_img.append(r_img)
        dia_by_text.append(r_txt)
        ok_now = 0.30 <= r_img <= 0.95
        ok_new = 0.55 <= r_txt <= 1.10
        verdict.append("OLD:" + ("pass" if ok_now else "REJECT"))
        verdict.append("NEW:" + ("pass" if ok_new else "reject"))
        print(f"{p.name:44s} {h:4d} {top:4d}-{bottom:<4d} {htext:5d}  "
              f"{seal['dia']:8.0f} {r_img:7.2f} {r_txt:9.2f}  {' '.join(verdict)}")

        # 被闭运算粘连的大块（dia > 0.6*imH 且 iso 高）—— 现有上界挡不住它们
        for c in cands:
            if c["dia"] > 0.6 * h and c["iso"] > 0.3:
                merged_by_img.append(c["dia"] / h)
                merged_by_text.append(c["dia"] / htext)
                break

    def stat(name: str, v: list[float]) -> None:
        if not v:
            print(f"{name}: (n=0)")
            return
        a = np.array(v)
        print(f"{name}: n={len(a)} min={a.min():.2f} p25={np.percentile(a,25):.2f} "
              f"median={np.median(a):.2f} p75={np.percentile(a,75):.2f} max={a.max():.2f}")

    print("\n--- 分布 ---")
    stat("dia / 图像高度（现有锚点）", dia_by_img)
    stat("dia / 文字行高（建议锚点）", dia_by_text)
    stat("粘连大块 / 图像高度（现有上界 0.95 挡不住）", merged_by_img)
    stat("粘连大块 / 文字行高（新上界 1.10 可挡）", merged_by_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
