#!/usr/bin/env python3
"""
Generate the official Ultralytics `yolov8n` (COCO) Core ML model and drop it into
the SDK's Resources folder. This model is used as a *placeholder detector* in the demo:
it has no license-plate class, so the app detects `car` (COCO id 2) and then OCRs each
vehicle region to extract German plates (see `YOLOConfig.cocoCar`).

Run on a Mac with internet access:

    python3 -m venv .venv && . .venv/bin/activate
    pip install ultralytics coremltools
    python3 Scripts/build_yolov8n_coreml.py

The script downloads `yolov8n.pt` (auto, via ultralytics), exports it to Core ML
(no NMS, 640x640) and copies `yolov8n.mlpackage` into
`LicensePlateOCR/LicensePlateSDK/Resources/`. Open the Xcode project and run.

To use YOUR OWN German-plate model instead, export it the same way
(`yolo export model=best.pt format=coreml imgsz=640 nms=False`), name it
`license_plate_yolo.mlpackage`, place it in Resources/, and switch the demo's
recognizer to `LicensePlateRecognizer()` (the default `.germanPlate` config).
"""

import pathlib
import shutil
import sys

import ultralytics  # noqa: F401  (ensures the package is importable)
from ultralytics import YOLO

HERE = pathlib.Path(__file__).resolve().parent
RESOURCES = HERE.parent / "LicensePlateOCR" / "LicensePlateSDK" / "Resources"


def main() -> int:
    print("[1/3] Loading yolov8n (downloads COCO weights on first run) ...")
    model = YOLO("yolov8n.pt")

    print("[2/3] Exporting to Core ML (raw output, 640x640, no NMS) ...")
    out = model.export(format="coreml", nms=False, int8=False, imgsz=640)
    src = pathlib.Path(out)
    if not src.exists():
        print(f"ERROR: export did not produce '{src}'", file=sys.stderr)
        return 1

    dest = RESOURCES / src.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    print(f"[3/3] Copied model -> {dest}")
    print("Done. Open LicensePlateOCR.xcodeproj in Xcode and run (choose 衿青 Team).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
