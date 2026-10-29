#!/usr/bin/env python3
"""Export the German plate pose model (.pt) to the Core ML .mlpackage that the
official Ultralytics YOLO iOS app expects in YOLOiOSApp/Models/Pose/.

This mirrors ultralytics/yolo-ios-app scripts/export-models.py for the pose task:
    format="coreml", imgsz=640, quantize=8, nms=False

Run this ON A MAC (Core ML export needs macOS). Example:

    # one-time environment
    uv venv --python 3.13 .venv
    uv pip install "ultralytics[export-coreml]>=8.4.142"
    source .venv/bin/activate

    # convert your weights and copy the package straight into a cloned official app
    python export_pose_coreml.py \
        --weights De-plate-pose-500.pt \
        --copy-to-app /path/to/yolo-ios-app

    # (optional) keep higher precision if int8 hurts small keypoints
    python export_pose_coreml.py --weights De-plate-pose-500.pt --quantize 16

Why this script exists: the official app never commits .mlpackage files (they are
gitignored and downloaded at build time). So we commit the .pt + this script, and
you produce the .mlpackage locally on your Mac.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=Path("De-plate-pose-500.pt"),
                        help="Path to the pose .pt weights (default: ./De-plate-pose-500.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Export input size (official pose uses 640)")
    parser.add_argument("--quantize", type=int, default=8, choices=[0, 8, 16, 32],
                        help="Core ML precision: 8=INT8 (official default), 16=FP16, 0/32=FP32. "
                             "Use 16 if INT8 degrades the small 4-point keypoints.")
    parser.add_argument("--copy-to-app", type=Path, default=None,
                        help="Path to a cloned yolo-ios-app repo root. If given, the exported "
                             ".mlpackage is copied into YOLOiOSApp/Models/Pose/.")
    parser.add_argument("--out", type=Path, default=Path("exports"),
                        help="Directory where the export is written (default: ./exports)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: "
              'uv pip install "ultralytics[export-coreml]>=8.4.142"', file=sys.stderr)
        return 2

    weights = args.weights.resolve()
    if not weights.exists():
        print(f"ERROR: weights not found: {weights}", file=sys.stderr)
        return 2

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {weights}")
    model = YOLO(str(weights))

    print(f"Exporting Core ML: imgsz={args.imgsz}, quantize={args.quantize}, nms=False")
    export_kwargs = dict(format="coreml", imgsz=args.imgsz, quantize=args.quantize, nms=False)
    try:
        package = model.export(**export_kwargs)
    except TypeError:
        # Older ultralytics builds may not accept nms= for pose; the raw head is the
        # default for pose anyway, so retry without it.
        print("Note: this ultralytics build rejected nms=False; retrying without it "
              "(pose Core ML exports a raw head by default).")
        export_kwargs.pop("nms", None)
        package = model.export(**export_kwargs)

    package = Path(package).resolve() if package else None
    if package is None or not package.exists():
        print("ERROR: export did not produce a .mlpackage", file=sys.stderr)
        return 3
    print(f"Exported: {package}")

    # ---- Verify the contract the official app relies on -------------------
    try:
        import coremltools as ct
        spec = ct.utils.load_spec(str(package))
        md = dict(spec.description.metadata.userDefined)
        print(f"  metadata: task={md.get('task')!r} imgsz={md.get('imgsz')!r} "
              f"names={md.get('names')!r} nms={md.get('nms')!r}")
        if md.get("task") not in (None, "pose"):
            print("  WARNING: metadata task is not 'pose'. The app still forces .pose by "
                  "folder (Models/Pose), so this is usually harmless.", file=sys.stderr)

        # Confirm the keypoint-bearing output shape and count.
        for out in spec.description.output:
            if out.type.HasField("multiArrayType"):
                shape = list(out.type.multiArrayType.shape)
                if len(shape) == 3:
                    c = shape[1]
                    if c > 5 and (c - 5) % 3 == 0:
                        print(f"  output '{out.name}' shape={shape} -> {(c - 5) // 3} keypoints "
                              "(traditional [1, C, N] head; official PostProcessPose handles this)")
                    else:
                        print(f"  output '{out.name}' shape={shape} (check this matches the "
                              "official pose decoder)")
    except Exception as exc:  # verification is best-effort
        print(f"  (metadata verification skipped: {exc})")

    # ---- Copy into the official app if requested --------------------------
    if args.copy_to_app:
        repo = args.copy_to_app.resolve()
        destination = repo / "YOLOiOSApp" / "Models" / "Pose" / package.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(package, destination)
        print(f"Copied -> {destination}")
        print("Now rebuild the YOLOiOSApp target in Xcode. The app discovers models in "
              "Models/Pose automatically; select Pose in the model picker.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
