#!/usr/bin/env python3
"""Report what the plate-pose model actually scores on an image.

Why this exists: "the PC detects it but the phone does not" is almost always a
scale/preprocessing difference, not a logic bug. This script runs the SAME ONNX
model, with the SAME preprocessing the iOS SDK uses (BGR, 127 grey padding,
top-left letterbox), and prints:

  1. the top-N detections on the whole picture (score, box, aspect ratio), and
  2. the best score inside a tight crop around a region you point at,

so you can tell "the model sees it but the app filters it" apart from
"the model never sees it at all" (in which case no client-side change can help).

Usage:
    python Scripts/pose_score_report.py --image photo.jpg
    python Scripts/pose_score_report.py --image photo.jpg --model path/to/best.onnx
    python Scripts/pose_score_report.py --image photo.jpg --crop 440 620 100 140

`--crop x y w h` is in source-image pixels and is where you believe the plate is;
the script reports the model's best score there at several zoom levels.
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


def load(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def infer(session: ort.InferenceSession, image: Image.Image, input_size: int = INPUT):
    """Letterbox `image` into a square canvas the way `prepare()` does on device."""
    width, height = image.size
    scale = min(input_size / width, input_size / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    bgr = np.ascontiguousarray(np.array(image.convert("RGB"))[:, :, ::-1])
    resized = np.array(Image.fromarray(bgr).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.full((input_size, input_size, 3), PAD, np.uint8)
    canvas[:new_h, :new_w] = resized
    tensor = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    # Export order for these pose models is (boxes, scores, keypoints_xy, keypoints_conf).
    scores = outputs[1].reshape(-1)
    boxes = outputs[0].reshape(-1, 4)
    # Upscale-normalised boxes back to source pixels.
    scale_back_w = width / new_w
    scale_back_h = height / new_h
    results = []
    for index in np.argsort(-scores)[:10]:
        cx, cy, bw, bh = boxes[index]
        results.append({
            "score": float(scores[index]),
            "x": (cx - bw / 2) * scale_back_w,
            "y": (cy - bh / 2) * scale_back_h,
            "w": bw * scale_back_w,
            "h": bh * scale_back_h,
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="OCR/LicensePlateOCR/LicensePlateSDK/"
                                         "Resources/De-plate-pose.onnx")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="region you believe holds the plate, in source pixels")
    args = parser.parse_args()

    session = load(args.model)
    image = Image.open(args.image).convert("RGB")
    print(f"image : {args.image}  {image.width}x{image.height}")
    print(f"model : {args.model}")
    print("whole picture:")
    for row in infer(session, image)[: args.top]:
        aspect = row["w"] / max(row["h"], 1e-6)
        print("  score=%.3f box=(x%.0f y%.0f w%.0f h%.0f) aspect=%.2f"
              % (row["score"], row["x"], row["y"], row["w"], row["h"], aspect))
    print("  (a real plate is ~4.4:1 for a car, ~1.6-2.5:1 for a two-line bike plate)")

    if args.crop:
        x, y, w, h = args.crop
        print(f"\ntight crop x={x} y={y} w={w} h={h} at several zoom levels:")
        for zoom in (1, 2, 3, 4):
            crop = image.crop((x, y, x + w, y + h))
            if zoom > 1:
                crop = crop.resize((w * zoom, h * zoom), Image.LANCZOS)
            best = infer(session, crop)[0]
            print("  x%d: best score=%.3f (box w%.0f h%.0f inside the crop)"
                  % (zoom, best["score"], best["w"], best["h"]))
        print("  -> all scores below the 0.25 gate means the model cannot see this "
              "plate; cropping, thresholds and post-filters cannot fix that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
