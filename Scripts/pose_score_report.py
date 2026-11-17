#!/usr/bin/env python3
"""Report what the plate-pose model actually scores on an image.

Why this exists: "the PC detects it but the phone does not" is usually a *scale*
difference, not a logic bug. This script runs the SAME ONNX model with the SAME
preprocessing the iOS SDK uses (BGR, 127 grey padding, top-left letterbox) and
prints the top detections with their boxes, aspect ratios and corner confidences.

IMPORTANT — read before drawing conclusions from ``--crop``:

  The graph outputs ``boxes [1, N, 4]`` as **xyxy corner coordinates in 512-canvas
  pixels** (see ``YOLOOutputDecoder.decodeLibreYOLO``). An earlier version of this
  script decoded them as ``cx, cy, w, h``; every box and aspect ratio it printed was
  wrong. It is fixed below.

  Also, feeding a *tight* plate crop is NOT representative: the model was trained on
  whole scenes, so an isolated plate enlarged into a 512x512 letterbox is out of
  distribution and scores ~0.03 even for a perfectly sharp plate. A low score on a
  tight crop says nothing about whether the model sees that plate in a real frame.
  Use ``--tiled`` (what the album path actually runs) or ``--crop ... --context 1.5``
  to keep surrounding scene context.

Usage:
    py -3 Scripts/pose_score_report.py --image photo.jpg
    py -3 Scripts/pose_score_report.py --image photo.jpg --tiled
    py -3 Scripts/pose_score_report.py --image photo.jpg --crop 900 1100 160 170 --context 4.5

``--crop x y w h`` is in source-image pixels. ``--tiled`` reproduces the SDK's
``detectTiled`` (1024 px tiles, 30 % overlap) instead of a single full-frame pass.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    sys.exit("onnxruntime is missing: pip install onnxruntime")

INPUT = 512
PAD = 127
TILE = 1024
OVERLAP = 0.3


def load(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def infer(session: ort.InferenceSession, image: Image.Image, input_size: int = INPUT):
    """Letterbox `image` into a square canvas the way `letterboxedUpright` does.

    Returns detections in SOURCE-IMAGE pixels, with xyxy boxes already mapped back.
    The model's four outputs are (boxes xyxy, scores, keypoints_xy, keypoints_conf).
    """
    width, height = image.size
    scale = min(input_size / width, input_size / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    bgr = np.ascontiguousarray(np.array(image.convert("RGB"))[:, :, ::-1])
    resized = np.array(Image.fromarray(bgr).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.full((input_size, input_size, 3), PAD, np.uint8)
    canvas[:new_h, :new_w] = resized
    tensor = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    boxes = outputs[0].reshape(-1, 4)
    scores = outputs[1].reshape(-1)
    corner_conf = outputs[3].reshape(-1, 4)

    scale_back_w = width / new_w
    scale_back_h = height / new_h
    results = []
    for index in np.argsort(-scores)[:400]:
        x0, y0, x1, y1 = boxes[index]
        results.append({
            "score": float(scores[index]),
            "x": float(x0) * scale_back_w,
            "y": float(y0) * scale_back_h,
            "w": (float(x1) - float(x0)) * scale_back_w,
            "h": (float(y1) - float(y0)) * scale_back_h,
            "kp_min": float(corner_conf[index].min()),
            "kp_mean": float(corner_conf[index].mean()),
        })
    return results


def nms(rows: list[dict], iou_threshold: float = 0.7) -> list[dict]:
    """Same suppression the SDK applies after decoding, so duplicates don't flood the report."""
    kept: list[dict] = []
    for row in sorted(rows, key=lambda r: -r["score"]):
        ax, ay, aw, ah = row["x"], row["y"], row["w"], row["h"]
        overlap = False
        for other in kept:
            bx, by, bw, bh = other["x"], other["y"], other["w"], other["h"]
            ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
            iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
            inter = ix * iy
            union = aw * ah + bw * bh - inter
            if union > 0 and inter / union > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(row)
    return kept


def tile_offsets(length: float, tile: float) -> list[float]:
    """Mirrors `LicensePlateDetector.detectTiled`: step by (1-overlap)*tile, then pin
    the last tile to the far edge. Order matters — do not sort or de-duplicate."""
    step = tile * (1 - OVERLAP)
    result: list[float] = []
    offset = 0.0
    while offset + tile < length - 1:
        result.append(offset)
        offset += step
    result.append(length - tile)
    return result


def infer_tiled(session: ort.InferenceSession, image: Image.Image) -> list[dict]:
    width, height = image.size
    tile_w, tile_h = min(TILE, width), min(TILE, height)
    if max(width, height) <= max(tile_w, tile_h):
        return infer(session, image)
    rows: list[dict] = []
    for ty in tile_offsets(height, tile_h):
        for tx in tile_offsets(width, tile_w):
            crop = image.crop((int(tx), int(ty), int(tx + tile_w), int(ty + tile_h)))
            for det in infer(session, crop):
                det["x"] += tx
                det["y"] += ty
                det["tile"] = (int(tx), int(ty))
                rows.append(det)
    return nms(rows)


def show(label: str, rows: list[dict], top: int) -> None:
    print(label)
    for row in nms(rows)[:top]:
        ar = row["w"] / max(row["h"], 1e-6)
        tile = (" tile=(%d,%d)" % row["tile"]) if "tile" in row else ""
        print("  score=%.3f box=(x%.0f y%.0f w%.0f h%.0f) aspect=%.2f "
              "kp(min/mean)=%.2f/%.2f%s"
              % (row["score"], row["x"], row["y"], row["w"], row["h"], ar,
                 row["kp_min"], row["kp_mean"], tile))
    print("  (car plate ~4.4:1; two-line motorcycle plate ~0.9-1.4:1)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="OCR/LicensePlateOCR/LicensePlateSDK/"
                                         "Resources/De-plate-pose.onnx")
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--tiled", action="store_true",
                        help="reproduce the album's 1024px tiled detection pass")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--context", type=float, default=1.0,
                        help="expand --crop by this factor around its centre (default 1.0)")
    args = parser.parse_args()

    session = load(args.model)
    image = Image.open(args.image).convert("RGB")
    print("image : %s  %dx%d" % (args.image, image.width, image.height))
    print("model : %s" % args.model)

    show("whole picture:", infer(session, image), args.top)
    if args.tiled:
        show("\ntiled pass (album path, 1024px tiles / 30% overlap):",
             infer_tiled(session, image), args.top)

    if args.crop:
        x, y, w, h = args.crop
        factor = max(1.0, args.context)
        nw, nh = w * factor, h * factor
        cx, cy = x + w / 2, y + h / 2
        left, top = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
        right = min(image.width, int(cx + nw / 2))
        bottom = min(image.height, int(cy + nh / 2))
        crop = image.crop((left, top, right, bottom))
        print("\ncrop x=%d y=%d w=%d h=%d  (context x%.1f -> %dx%d at %d,%d):"
              % (x, y, w, h, factor, crop.width, crop.height, left, top))
        show("", infer(session, crop), args.top)
        print("  NOTE: an isolated tight plate crop is out of distribution for this "
              "model; a low score here does NOT mean the plate is invisible in a scene. "
              "Use --context 4..5 (a ~1024px tile) or --tiled to reproduce real framing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
