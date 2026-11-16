#!/usr/bin/env python3
"""
PC reference / verification script for the on-device PaddleOCR recognizer.

It mirrors `LicensePlateSDK/PaddleOCRTextRecognizer.swift` step for step:

  CGImage / image file → (resizedW, 48) BGR → (x/255 - 0.5)/0.5
  → right-pad to W with zeros → ORT [1,3,48,W] → [1,T,18385] CTC probs
  → greedy decode + German-plate-constrained CTC prefix beam search

Use it to confirm that a rectified plate crop is read correctly on the PC before
blaming the Swift implementation, and to re-fetch the ONNX model + dictionary:

    python Scripts/paddleocr_rec_reference.py --self-test
    python Scripts/paddleocr_rec_reference.py rectified_plate.png --mode accurate
    python Scripts/paddleocr_rec_reference.py --fetch-models

Model provenance: `imgocr` 0.2.4 (Apache-2.0) ships Paddle2ONNX conversions of
PaddleOCR's PP-OCRv5 mobile models. PaddleOCR itself is Apache-2.0. The wheel is
~25 MB; only the 16.5 MB recognition graph and the 73 KB dictionary are copied.

Dependencies: numpy, onnxruntime, opencv-python (or Pillow), Pillow for --self-test.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESOURCES = REPO / "OCR" / "LicensePlateOCR" / "LicensePlateSDK" / "Resources"
MODEL_NAME = "PP-OCRv5_rec_mobile_infer.onnx"
DICT_NAME = "ppocrv5_dict.txt"
PLATE_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-ÄÖÜ ")

IMAGE_HEIGHT = 48
BASE_WIDTH = 320
BLANK = 0
NEG_INF = float("-inf")


# --------------------------------------------------------------------------- #
# Model acquisition
# --------------------------------------------------------------------------- #

def fetch_models() -> None:
    """Re-download the ONNX graph + dictionary from the pinned imgocr wheel."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [sys.executable, "-m", "pip", "download", "imgocr==0.2.4",
             "--no-deps", "-d", tmp],
            check=True,
        )
        wheels = list(Path(tmp).glob("imgocr-*.whl"))
        if not wheels:
            raise RuntimeError("pip download did not produce an imgocr wheel")
        with zipfile.ZipFile(wheels[0]) as archive:
            for member in ("imgocr/models/" + MODEL_NAME, "imgocr/models/" + DICT_NAME):
                RESOURCES.mkdir(parents=True, exist_ok=True)
                target = RESOURCES / Path(member).name
                target.write_bytes(archive.read(member))
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                print(f"wrote {target} ({target.stat().st_size:,} bytes, sha256 {digest[:16]}…)")


# --------------------------------------------------------------------------- #
# Alphabet / preprocessing
# --------------------------------------------------------------------------- #

def load_alphabet(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lines = [line[:-1] if line.endswith("\r") else line for line in lines]
    lines.append(" ")                 # use_space_char=True in imgocr
    return ["blank"] + lines          # CTC blank at index 0


def allowed_classes(alphabet: list[str]) -> list[int]:
    return [BLANK] + [i for i, char in enumerate(alphabet) if char in PLATE_CHARS]


def prepare(image_bgr, image_height: int = IMAGE_HEIGHT, base_width: int = BASE_WIDTH):
    """Returns the [1,3,48,W] float32 tensor and the resized width."""
    import cv2
    import numpy as np

    height, width = image_bgr.shape[:2]
    ratio = width / float(height)
    max_wh_ratio = max(base_width / image_height, ratio)
    img_w = int(image_height * max_wh_ratio)
    resized_w = min(int(math.ceil(image_height * ratio)), img_w)
    resized = cv2.resize(image_bgr, (resized_w, image_height)).astype(np.float32)
    x = resized.transpose((2, 0, 1)) / 255.0
    x = (x - 0.5) / 0.5
    canvas = np.zeros((3, image_height, img_w), dtype=np.float32)
    canvas[:, :, 0:resized_w] = x
    return canvas[None], img_w, resized_w


def read_bgr(path: Path):
    """Reads an image as BGR uint8, matching cv2.imread in PaddleOCR."""
    try:
        import cv2
        data = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if data is not None:
            return data
    except ImportError:
        pass
    from PIL import Image
    import numpy as np
    return np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


def resize_bgr(image_bgr, scale_x: float):
    import cv2
    height, width = image_bgr.shape[:2]
    return cv2.resize(image_bgr, (max(16, int(width * scale_x)), height))


# --------------------------------------------------------------------------- #
# CTC decoding (same log-domain prefix beam as the Swift implementation)
# --------------------------------------------------------------------------- #

def log_add(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    high = max(a, b)
    return high + math.log1p(math.exp(min(a, b) - high))


def greedy_decode(probs, alphabet: list[str]) -> tuple[str, float]:
    indices = probs.argmax(axis=1)
    values = probs.max(axis=1)
    text, total, count, previous = "", 0.0, 0, BLANK
    for index, value in zip(indices, values):
        if index != BLANK and index != previous:
            text += alphabet[int(index)]
            total += float(value)
            count += 1
        previous = index
    return text, (total / count if count else 0.0)


def beam_decode(probs, alphabet: list[str], allowed: list[int],
                beam_width: int = 8, limit: int = 6) -> list[tuple[str, float]]:
    # (p_blank, p_nonblank), both in log space
    beams: dict[str, tuple[float, float]] = {"": (0.0, NEG_INF)}

    def merge(target: dict, key: str, blank: float, non_blank: float) -> None:
        old_blank, old_non_blank = target.get(key, (NEG_INF, NEG_INF))
        target[key] = (log_add(old_blank, blank), log_add(old_non_blank, non_blank))

    for row in probs:
        nxt: dict[str, tuple[float, float]] = {}
        for prefix, (p_blank, p_non_blank) in beams.items():
            total = log_add(p_blank, p_non_blank)
            merge(nxt, prefix, total + math.log(max(float(row[BLANK]), 1e-12)), NEG_INF)
            last = prefix[-1] if prefix else None
            for cls in allowed:
                if cls == BLANK:
                    continue
                character = alphabet[cls]
                if not character:
                    continue
                probability = math.log(max(float(row[cls]), 1e-12))
                if character == last:
                    # repeated character needs an intervening blank
                    merge(nxt, prefix, NEG_INF, p_non_blank + probability)
                    merge(nxt, prefix + character, NEG_INF, p_blank + probability)
                else:
                    merge(nxt, prefix + character, NEG_INF, total + probability)
        beams = dict(sorted(nxt.items(),
                            key=lambda item: log_add(*item[1]),
                            reverse=True)[:beam_width])

    ranked = [(text, log_add(*state)) for text, state in beams.items() if text]
    return sorted(ranked, key=lambda item: -item[1])[:limit]


def normalize_probabilities(probs):
    """PP-OCRv5 exports already emit softmax probabilities; tolerate raw logits."""
    import numpy as np
    if probs.min() >= 0 and abs(float(probs.sum(axis=1).mean()) - 1.0) < 0.05:
        return probs.astype(np.float64)
    shifted = probs - probs.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float64)


# --------------------------------------------------------------------------- #
# End-to-end recognition
# --------------------------------------------------------------------------- #

def recognize(image_bgr, session, input_name: str, output_name: str,
              alphabet: list[str], allowed: list[int], mode: str = "fast"):
    ranked: dict[str, tuple[float, float]] = {}

    def record(text: str, score: float, confidence: float) -> None:
        if not text:
            return
        old = ranked.get(text)
        ranked[text] = (max(score, old[0]) if old else score,
                        max(confidence, old[1]) if old else confidence)

    variants = [image_bgr]
    if mode == "accurate":
        inset = max(1, int(image_bgr.shape[0] * 0.06))
        if image_bgr.shape[0] - 2 * inset >= 16:
            variants.append(image_bgr[inset:-inset, :])
        variants.append(resize_bgr(image_bgr, 0.92))

    beam_width = 16 if mode == "accurate" else 8
    limit = 12 if mode == "accurate" else 6

    for variant in variants:
        tensor, img_w, resized_w = prepare(variant)
        raw = session.run([output_name], {input_name: tensor})[0][0]
        probs = normalize_probabilities(raw)
        greedy_text, confidence = greedy_decode(probs, alphabet)
        beams = beam_decode(probs, alphabet, allowed, beam_width, limit)
        top_score = beams[0][1] if beams else -60.0
        for text, score in beams:
            record(text, score, confidence)
        record(greedy_text, top_score - 2, confidence)

    ordered = sorted(ranked.items(), key=lambda item: -item[1][0])
    if not ordered:
        return "", 0.0, []
    primary, (top_score, primary_confidence) = ordered[0]
    alternatives = []
    for text, (score, confidence) in ordered[1:10]:
        relative = math.exp(min(0, score - top_score))
        alternatives.append((text, max(0.01, min(1, confidence) * relative * 0.98)))
    return primary, min(1, max(0, primary_confidence)), alternatives


def load_session():
    import onnxruntime as ort
    model = RESOURCES / MODEL_NAME
    dictionary = RESOURCES / DICT_NAME
    if not model.exists() or not dictionary.exists():
        raise SystemExit(
            f"缺少模型资源：{model} / {dictionary}\n"
            "先运行： python Scripts/paddleocr_rec_reference.py --fetch-models")
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    alphabet = load_alphabet(dictionary)
    classes = session.get_outputs()[0].shape[-1]
    if isinstance(classes, int) and classes != len(alphabet):
        raise SystemExit(f"字典 {len(alphabet)} 类与模型输出 {classes} 类不一致")
    return (session, session.get_inputs()[0].name, session.get_outputs()[0].name,
            alphabet, allowed_classes(alphabet))


def self_test() -> None:
    from PIL import Image, ImageDraw, ImageFont

    session, input_name, output_name, alphabet, allowed = load_session()
    font_path = None
    for candidate in ("C:/Windows/Fonts/arialbd.ttf",
                      "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(candidate).exists():
            font_path = candidate
            break
    print(f"model classes={len(alphabet)}, constrained beam classes={len(allowed)}")
    for text in ("B-AB 1234", "S AB 1234E", "HH-AB 123", "M XY 987H"):
        image = Image.new("RGB", (460, 100), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, 62) if font_path else ImageFont.load_default()
        box = draw.textbbox((0, 0), text, font=font)
        draw.text(((460 - (box[2] - box[0])) / 2 - box[0],
                   (100 - (box[3] - box[1])) / 2 - box[1]),
                  text, font=font, fill="black")
        bgr = read_bgr_from_pil(image)
        primary, confidence, alternatives = recognize(
            bgr, session, input_name, output_name, alphabet, allowed, mode="fast")
        ok = primary.replace(" ", "").replace("-", "").upper() == text.replace(" ", "").replace("-", "").upper()
        print(f"  {'OK ' if ok else 'DIFF'} {text!r:14} → {primary!r:16} conf={confidence:.3f}")
        print(f"       candidates: {[t for t, _ in alternatives[:4]]}")


def read_bgr_from_pil(image):
    import numpy as np
    return np.array(image.convert("RGB"))[:, :, ::-1].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", type=Path,
                        help="rectified plate crop (PNG/JPG) to recognize")
    parser.add_argument("--mode", choices=("fast", "accurate"), default="fast")
    parser.add_argument("--fetch-models", action="store_true",
                        help="download imgocr 0.2.4 and extract model + dictionary")
    parser.add_argument("--self-test", action="store_true",
                        help="recognize four synthetic German plates")
    args = parser.parse_args()

    if args.fetch_models:
        fetch_models()
        return
    if args.self_test:
        self_test()
        return
    if args.image is None:
        parser.print_help()
        return
    if not args.image.exists():
        raise SystemExit(f"找不到图片：{args.image}")

    session, input_name, output_name, alphabet, allowed = load_session()
    primary, confidence, alternatives = recognize(
        read_bgr(args.image), session, input_name, output_name,
        alphabet, allowed, mode=args.mode)
    print(f"primary    : {primary!r}  confidence={confidence:.3f}  mode={args.mode}")
    for text, score in alternatives:
        print(f"candidate  : {text!r}  confidence={score:.3f}")


if __name__ == "__main__":
    main()
