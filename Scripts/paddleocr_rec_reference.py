#!/usr/bin/env python3
"""
PC reference / verification script for the on-device PaddleOCR recognizer.

It mirrors `LicensePlateSDK/PaddleOCRTextRecognizer.swift` step for step:

  CGImage / image file → (resizedW, 48) BGR → (x/255 - 0.5)/0.5
  → right-pad to W with zeros → ORT [1,3,48,W] → [1,T,18385] CTC probs
  → German-plate-constrained CTC prefix beam search + full-dictionary greedy
    reading (a cleaning-quality DIAGNOSTIC, never a recognition candidate)

The 41 plate characters are a DECODER CONSTRAINT, not a dictionary. The bundled
`ppocrv5_dict.txt` is an index→character map bound to the model's fixed 18,385
output classes (`B` is class 16180) and must never be truncated.

Use it to confirm that a rectified plate crop is read correctly on the PC before
blaming the Swift implementation, to batch-score a real sample set, and to
re-fetch the ONNX model + dictionary:

    python Scripts/paddleocr_rec_reference.py --self-test
    python Scripts/paddleocr_rec_reference.py rectified_plate.png --mode accurate
    python Scripts/paddleocr_rec_reference.py --eval samples/ --labels samples/labels.tsv
    python Scripts/paddleocr_rec_reference.py --fetch-models

`--eval` reports accuracy and character error rate. Truth comes from the TSV
(filename<TAB>expected) or, when omitted, from each image's filename stem. Run
the same crops through Apple Vision on device to quantify the engine swap.

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


def is_plausible_prefix(text: str) -> bool:
    """Conservative FZV-shaped gate, mirroring Swift `isPlausiblePrefix`.

    Rejects only transitions that can never appear on a German plate, so a real
    reading must never be pruned. Umlauts are legal in the 1-3 letter city code
    but never in the identification number; after the digit block only a terminal
    E/H suffix may appear.
    """
    core = text.replace("-", "").replace(" ", "")
    if not core:
        return True
    if len(core) > 10:
        return False
    saw_digit = False
    letter_run = 0
    for offset, character in enumerate(core):
        if character.isdigit():
            saw_digit = True
            letter_run = 0
        elif character in "ÄÖÜ":
            if saw_digit:
                return False
            letter_run += 1
        else:
            letter_run += 1
            if saw_digit and (character not in "EH" or offset != len(core) - 1):
                return False
        if letter_run > 6:
            return False
    return True


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
                    candidate = prefix + character
                    if is_plausible_prefix(candidate):
                        merge(nxt, candidate, NEG_INF, p_blank + probability)
                else:
                    candidate = prefix + character
                    if is_plausible_prefix(candidate):
                        merge(nxt, candidate, NEG_INF, total + probability)
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
    """Returns (primary, confidence, alternatives, diagnostics).

    Mirrors the Swift recognizer: the full-dictionary greedy reading is a
    CLEANING-QUALITY DIAGNOSTIC, not a recognition candidate, and every
    alternative is graded by its own CTC path score relative to the winner.
    """
    ranked: dict[str, float] = {}
    greedy_readings: list[str] = []
    best_greedy_confidence = 0.0

    def record(text: str, score: float) -> None:
        if text:
            ranked[text] = max(score, ranked.get(text, NEG_INF))

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
        for text, score in beams:
            record(text, score)
        if greedy_text:
            greedy_readings.append(greedy_text)
        best_greedy_confidence = max(best_greedy_confidence, confidence)

    if not ranked:
        # No legal CTC path survived; emit a character-filtered greedy reading.
        fallback = canonical(greedy_readings[0] if greedy_readings else "")
        record(fallback, -60.0)

    stray = set("".join(greedy_readings)) - PLATE_CHARS
    diagnostics = {"greedy": greedy_readings[0] if greedy_readings else "",
                   "stray": "".join(sorted(stray))}

    ordered = sorted(ranked.items(), key=lambda item: -item[1])
    if not ordered:
        return "", 0.0, [], diagnostics
    primary, top_score = ordered[0]
    # Absolute anchor for the vote aggregator's provisionalConfidence threshold.
    primary_confidence = min(1.0, max(0.01, best_greedy_confidence))
    alternatives = [
        (text, max(0.01, primary_confidence * math.exp(min(0, score - top_score)) * 0.98))
        for text, score in ordered[1:10]
    ]
    return primary, primary_confidence, alternatives, diagnostics


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
        primary, confidence, alternatives, _ = recognize(
            bgr, session, input_name, output_name, alphabet, allowed, mode="fast")
        ok = primary.replace(" ", "").replace("-", "").upper() == text.replace(" ", "").replace("-", "").upper()
        print(f"  {'OK ' if ok else 'DIFF'} {text!r:14} → {primary!r:16} conf={confidence:.3f}")
        print(f"       candidates: {[t for t, _ in alternatives[:4]]}")


def read_bgr_from_pil(image):
    import numpy as np
    return np.array(image.convert("RGB"))[:, :, ::-1].copy()


def canonical(text: str) -> str:
    """Uppercase plate text with separators removed, used only for scoring."""
    return "".join(c for c in text.upper() if c in PLATE_CHARS and c not in "- ")


def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def evaluate(directory: Path, labels: Path | None, mode: str) -> None:
    """Batch harness: score a folder of rectified plate crops.

    Ground truth comes from `--labels` (TSV: filename<TAB>expected) when given,
    otherwise from each filename stem. Run this on the SAME crops the on-device
    A/B test uses, then compare accuracy / CER against Apple Vision.
    """
    session, input_name, output_name, alphabet, allowed = load_session()
    expected_map: dict[str, str] = {}
    if labels and labels.exists():
        for line in labels.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t", 1) if "\t" in line else line.split(",", 1)
            if len(parts) == 2:
                expected_map[parts[0].strip()] = parts[1].strip()
    images = sorted(p for p in directory.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"})
    if not images:
        raise SystemExit(f"目录中没有图片：{directory}")

    hits = 0
    edits = 0
    chars = 0
    print(f"mode={mode}  images={len(images)}  labels={labels or '(filename stem)'}")
    print(f"{'file':<34}{'expected':<16}{'primary':<16}{'ok':<6}conf")
    for path in images:
        truth = canonical(expected_map.get(path.name, path.stem))
        primary, confidence, _, diagnostics = recognize(
            read_bgr(path), session, input_name, output_name,
            alphabet, allowed, mode=mode)
        prediction = canonical(primary)
        ok = prediction == truth
        hits += int(ok)
        edits += levenshtein(prediction, truth)
        chars += len(truth)
        note = f"  异常字符 {diagnostics['stray']}" if diagnostics["stray"] else ""
        print(f"{path.name[:33]:<34}{truth[:15]:<16}{prediction[:15]:<16}"
              f"{'OK' if ok else 'ERR':<6}{confidence:.3f}{note}")
    print(f"\naccuracy = {hits}/{len(images)} = {hits / len(images):.1%}")
    print(f"CER      = {edits}/{chars} = {edits / max(1, chars):.1%}")
    print("提示：用同一批矫正图在真机切换 Apple Vision 跑一次，即可量化替换收益。")


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
    parser.add_argument("--eval", type=Path, metavar="DIR",
                        help="batch-evaluate a folder of rectified plate crops")
    parser.add_argument("--labels", type=Path, metavar="TSV",
                        help="for --eval: filename<TAB>expected_plate, one per line")
    args = parser.parse_args()

    if args.fetch_models:
        fetch_models()
        return
    if args.self_test:
        self_test()
        return
    if args.eval:
        if not args.eval.is_dir():
            raise SystemExit(f"不是目录：{args.eval}")
        evaluate(args.eval, args.labels, args.mode)
        return
    if args.image is None:
        parser.print_help()
        return
    if not args.image.exists():
        raise SystemExit(f"找不到图片：{args.image}")

    session, input_name, output_name, alphabet, allowed = load_session()
    primary, confidence, alternatives, diagnostics = recognize(
        read_bgr(args.image), session, input_name, output_name,
        alphabet, allowed, mode=args.mode)
    print(f"primary    : {primary!r}  confidence={confidence:.3f}  mode={args.mode}")
    if diagnostics["greedy"]:
        print(f"greedy(诊断): {diagnostics['greedy']!r}"
              + (f"  异常字符 {diagnostics['stray']}" if diagnostics["stray"] else ""))
    for text, score in alternatives:
        print(f"candidate  : {text!r}  confidence={score:.3f}")


if __name__ == "__main__":
    main()
