#!/usr/bin/env python3
"""One-command setup: take the official Ultralytics YOLO iOS app and turn it into a
German plate OCR app running YOUR 4-keypoint pose model.

What it does (all on your Mac):
  1. git clone https://github.com/ultralytics/yolo-ios-app.git   (unless --app-dir given)
  2. export De-plate-pose-500.pt -> Core ML .mlpackage           (unless --model given)
  3. copy the .mlpackage into YOLOiOSApp/Models/Pose/            (auto-discovered by the app)
  4. INJECT the plate OCR into YOLOiOSApp/YOLOiOSApp/ViewController.swift
     - imports (Vision / CoreImage) at the top
     - one call at the top of yoloView(_:didReceiveResult:)
     - the full OCR implementation appended at the end of the file

No new source files are added, so project.pbxproj is never touched.
The script is idempotent: re-running it detects an already-patched file and skips.

Usage:
    python setup_official_app.py                       # everything, defaults
    python setup_official_app.py --app-dir /path/to/yolo-ios-app
    python setup_official_app.py --model exports/De-plate-pose-500.mlpackage
    python setup_official_app.py --dry-run             # show what would change
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_URL = "https://github.com/ultralytics/yolo-ios-app.git"
DEFAULT_CLONE_DIR = Path("yolo-ios-app")
DEFAULT_WEIGHTS = SCRIPT_DIR / "De-plate-pose-500.pt"
DEFAULT_MODEL_NAME = "De-plate-pose-500.mlpackage"

VC_RELATIVE = Path("YOLOiOSApp") / "YOLOiOSApp" / "ViewController.swift"
MODELS_RELATIVE = Path("YOLOiOSApp") / "Models" / "Pose"

# --------------------------------------------------------------------------
# Swift: the OCR call injected at the top of the existing delegate method.
# --------------------------------------------------------------------------
INJECTED_CALL = "        recognizePlateIfNeeded(result: result, in: view)"

# The exact signature we anchor on (whitespace-tolerant).
DID_RECEIVE_RE = re.compile(
    r"(func\s+yoloView\s*\(\s*_\s+view\s*:\s*YOLOView\s*,\s*didReceiveResult\s+result\s*:\s*YOLOResult\s*\)\s*\{)"
)

# --------------------------------------------------------------------------
# Swift: force-load our bundled model at startup.
#
# Why: the official picker hard-filters its list to official remote model names:
#     let names = Set(remoteModelsInfo[task.name]!.map { $0.modelName.lowercased() })
#     makeModelEntries(for: task.name).filter { names.contains($0.displayName.lowercased()) }
# So a locally bundled .mlpackage is in the bundle (getModelFiles finds it, it is
# appended to currentModels) but is NEVER shown in the UI and can never be selected.
# We therefore load it directly instead of asking the user to pick it.
# BasePredictor compiles an un-bundled .mlpackage itself (MLModel.compileModel), so
# shipping the raw .mlpackage is fine - no manual compile step needed.
# --------------------------------------------------------------------------
STARTUP_ANCHOR_RE = re.compile(r"(func\s+loadSavedModelOrFirstOfficial\s*\(\s*\)\s*\{)")
STARTUP_CALL = "        if plateLoadPoseModel() { return }"

STARTUP_SWIFT_BLOCK = r'''
// MARK: - Force-load the bundled German plate pose model (injected by PlateOCR-de)
// The official picker only lists official remote model names, so a locally bundled
// model never appears in the UI. Load it directly: BasePredictor compiles an
// uncompiled .mlpackage itself, so no manual compile step is needed.

private let _platePoseModelName = "De-plate-pose-500"

extension ViewController {

    /// Loads the bundled pose model at startup.
    /// Returns true when it took over startup loading (so the official model is skipped).
    @discardableResult
    func plateLoadPoseModel() -> Bool {
        guard let url = plateBundledModelURL(named: _platePoseModelName) else {
            print("Plate model not found in bundle: \(_platePoseModelName)")
            return false
        }
        yoloView.setModel(modelPathOrName: url.path, task: .pose) { [weak self] result in
            guard let self else { return }
            switch result {
            case .success:
                self.labelName.text = _platePoseModelName
                self.yoloView.setInferenceFlag(ok: true)
                print("Plate model loaded: \(url.lastPathComponent)")
            case .failure(let error):
                print("Plate model failed to load: \(error)")
            }
        }
        return true
    }
}

/// Locates the bundled model: Models/Pose first, then Models, then the bundle root.
private func plateBundledModelURL(named name: String) -> URL? {
    let fm = FileManager.default
    for folder in ["Models/Pose", "Models"] {
        guard let folderURL = Bundle.main.url(forResource: folder, withExtension: nil) else {
            continue
        }
        for ext in ["mlmodelc", "mlpackage"] {
            let candidate = folderURL.appendingPathComponent(name).appendingPathExtension(ext)
            if fm.fileExists(atPath: candidate.path) { return candidate }
        }
    }
    for ext in ["mlmodelc", "mlpackage"] {
        if let url = Bundle.main.url(forResource: name, withExtension: ext) { return url }
    }
    return nil
}
'''

# --------------------------------------------------------------------------
# Swift: everything appended to the end of ViewController.swift.
# --------------------------------------------------------------------------
SWIFT_BLOCK = r'''
// MARK: - Plate OCR (injected by PlateOCR-de)
// Uses the pose model's 4 keypoints (plate corners) for perspective correction,
// then Apple Vision OCR, then German plate validation.
// Keypoint order from the model: leftTop, rightTop, rightBottom, leftBottom
// (re-ordered defensively by centroid below).

private let _plateOCRService = PlateOCRService(throttle: 0.3)
private let _plateLabel = UILabel()

/// Perspective-corrects the plate quad and runs Apple Vision OCR.
private final class PlateOCRService {
    private let ciContext = CIContext(options: nil)
    private let queue = DispatchQueue(label: "plate.ocr", qos: .userInitiated)
    private var lastRun = Date.distantPast
    private let throttle: TimeInterval

    init(throttle: TimeInterval) { self.throttle = throttle }

    func recognize(frame: UIImage, corners: [CGPoint], completion: @escaping (String?) -> Void) {
        guard corners.count == 4 else { completion(nil); return }
        let now = Date()
        guard now.timeIntervalSince(lastRun) >= throttle else { completion(nil); return }
        lastRun = now
        queue.async { [weak self] in
            guard let self else { return }
            guard let corrected = self.perspectiveCorrect(frame: frame, corners: corners) else {
                completion(nil)
                return
            }
            self.runVisionOCR(on: corrected, completion: completion)
        }
    }

    private func perspectiveCorrect(frame: UIImage, corners: [CGPoint]) -> UIImage? {
        guard let cg = frame.cgImage else { return nil }
        let ci = CIImage(cgImage: cg)
        let height = ci.extent.height

        // UIKit top-left origin -> Core Image bottom-left origin.
        func ciPoint(_ p: CGPoint) -> CIVector { CIVector(x: p.x, y: height - p.y) }

        guard let filter = CIFilter(name: "CIPerspectiveCorrection") else { return nil }
        filter.setValue(ci, forKey: kCIInputImageKey)
        filter.setValue(ciPoint(corners[0]), forKey: "inputTopLeft")
        filter.setValue(ciPoint(corners[1]), forKey: "inputTopRight")
        filter.setValue(ciPoint(corners[3]), forKey: "inputBottomLeft")
        filter.setValue(ciPoint(corners[2]), forKey: "inputBottomRight")

        guard let output = filter.outputImage else { return nil }
        let extent = output.extent
        // Reject degenerate quads instead of warping garbage.
        guard extent.width >= 24, extent.height >= 8 else { return nil }

        // Upscale small crops so Vision has enough pixels.
        let scale = max(1.0, 320.0 / extent.width)
        let scaled = output.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let cgOut = ciContext.createCGImage(scaled, from: scaled.extent) else { return nil }
        return UIImage(cgImage: cgOut)
    }

    private func runVisionOCR(on image: UIImage, completion: @escaping (String?) -> Void) {
        guard let cg = image.cgImage else { completion(nil); return }
        let request = VNRecognizeTextRequest { request, _ in
            guard let observations = request.results as? [VNRecognizedTextObservation],
                  !observations.isEmpty
            else { completion(nil); return }
            let candidates = observations.compactMap { $0.topCandidates(1).first }
            let joined = candidates.map(\.string).joined(separator: " ")
            completion(joined.isEmpty ? nil : joined)
        }
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = false
        request.recognitionLanguages = ["de-DE", "en-US"]
        request.minimumTextHeight = 0.05

        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        do { try handler.perform([request]) } catch { completion(nil) }
    }
}

/// [city 1-3 letters] - [1-2 letters][1-4 digits][optional E/H]
private enum GermanPlateValidator {
    private static let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ"
    private static let allowed = letters + "0123456789-"

    static func normalize(_ raw: String) -> String? {
        var s = raw.uppercased()
        s = s.replacingOccurrences(of: " ", with: "")
        s = String(s.filter { allowed.contains($0) })
        guard !s.isEmpty else { return nil }

        if s.contains("-") {
            let parts = s.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false).map(String.init)
            if parts.count == 2, isValid(city: parts[0], rest: parts[1]) { return "\(parts[0])-\(parts[1])" }
        }

        let compact = s.replacingOccurrences(of: "-", with: "")
        guard compact.count >= 3 else { return nil }
        for cityLen in 1...min(3, compact.count - 2) {
            let idx = compact.index(compact.startIndex, offsetBy: cityLen)
            let city = String(compact[..<idx])
            let rest = String(compact[idx...])
            if isValid(city: city, rest: rest) { return "\(city)-\(rest)" }
        }
        return nil
    }

    private static func isValid(city: String, rest: String) -> Bool {
        guard (1...3).contains(city.count),
              city.allSatisfy({ letters.contains($0) }),
              !rest.isEmpty,
              rest.range(of: "^[A-ZÄÖÜ]{1,2}[0-9]{1,4}[EH]?$", options: .regularExpression) != nil
        else { return false }
        // FZV: city + identification (excluding the E/H suffix) <= 8 characters.
        let hasSuffix = (rest.last == "E" || rest.last == "H")
        let restCore = hasSuffix ? String(rest.dropLast()) : rest
        return city.count + restCore.count <= 8
    }
}

extension ViewController {

    /// Injected into yoloView(_:didReceiveResult:). No-op unless the pose model found 4 keypoints.
    func recognizePlateIfNeeded(result: YOLOResult, in view: YOLOView) {
        guard let keypoints = result.keypointsList.first, keypoints.xyn.count >= 4 else { return }

        let start: (UIImage) -> Void = { frame in
            guard let cg = frame.cgImage else { return }
            let w = CGFloat(cg.width), h = CGFloat(cg.height)
            let corners = keypoints.xyn.prefix(4).map {
                CGPoint(x: CGFloat($0.x) * w, y: CGFloat($0.y) * h)
            }
            _plateOCRService.recognize(frame: frame, corners: plateOrderCorners(corners)) { text in
                guard let raw = text, let plate = GermanPlateValidator.normalize(raw) else { return }
                DispatchQueue.main.async { plateShow(plate, in: view) }
            }
        }

        if let frame = result.originalImage {
            start(frame)
        } else {
            view.capturePhoto { image in
                if let image { start(image) }
            }
        }
    }
}

/// Reorders 4 points to [leftTop, rightTop, rightBottom, leftBottom] by centroid quadrant.
private func plateOrderCorners(_ pts: [CGPoint]) -> [CGPoint] {
    guard pts.count == 4 else { return pts }
    let cx = pts.map(\.x).reduce(0, +) / 4
    let cy = pts.map(\.y).reduce(0, +) / 4
    var lt = CGPoint.zero, rt = CGPoint.zero, rb = CGPoint.zero, lb = CGPoint.zero
    for p in pts {
        switch (p.x < cx, p.y < cy) {
        case (true, true): lt = p
        case (false, true): rt = p
        case (false, false): rb = p
        case (true, false): lb = p
        }
    }
    return [lt, rt, rb, lb]
}

/// Shows the recognised plate on top of the camera view.
private func plateShow(_ text: String, in view: YOLOView) {
    if _plateLabel.superview == nil {
        _plateLabel.textColor = .white
        _plateLabel.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        _plateLabel.font = .monospacedSystemFont(ofSize: 28, weight: .semibold)
        _plateLabel.textAlignment = .center
        _plateLabel.layer.cornerRadius = 8
        _plateLabel.layer.masksToBounds = true
        _plateLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(_plateLabel)
        NSLayoutConstraint.activate([
            _plateLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            _plateLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 96),
            _plateLabel.heightAnchor.constraint(equalToConstant: 48),
            _plateLabel.widthAnchor.constraint(lessThanOrEqualTo: view.widthAnchor, constant: -40),
        ])
    }
    _plateLabel.text = "  \(text)  "
    _plateLabel.isHidden = false
}
'''


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], cwd: Path | None = None) -> int:
    log("  $ " + " ".join(str(c) for c in cmd))
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


def ensure_app(app_dir: Path | None, clone_to: Path) -> Path:
    if app_dir:
        app_dir = app_dir.resolve()
        if not (app_dir / VC_RELATIVE).exists():
            raise SystemExit(f"ERROR: {app_dir / VC_RELATIVE} not found. "
                             "Is --app-dir pointing at a cloned yolo-ios-app?")
        log(f"[1/5] Using existing app at {app_dir}")
        return app_dir

    if not clone_to.exists():
        log(f"[1/5] Cloning official app into {clone_to} ...")
        if run(["git", "clone", "--depth", "1", REPO_URL, str(clone_to)]) != 0:
            raise SystemExit("ERROR: git clone failed.")
    else:
        log(f"[1/5] Reusing existing clone at {clone_to}")
    app_dir = clone_to.resolve()
    if not (app_dir / VC_RELATIVE).exists():
        raise SystemExit(f"ERROR: {VC_RELATIVE} missing after clone.")
    return app_dir


def python_has_module(module: str) -> bool:
    return subprocess.call([sys.executable, "-c", f"import {module}"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def ensure_ultralytics() -> None:
    """Make sure ultralytics (+ coremltools) is importable by the running python."""
    if python_has_module("ultralytics") and python_has_module("coremltools"):
        return

    pkg = '"ultralytics[export-coreml]>=8.4.142"'
    log(f"  ultralytics/coremltools missing for {sys.executable}")
    log(f"  $ {sys.executable} -m pip install {pkg}")
    if run([sys.executable, "-m", "pip", "install", pkg]) != 0:
        # pip may itself be missing (e.g. bare Xcode Command Line Tools python3).
        if not python_has_module("pip"):
            log("  pip itself is missing. Try one of:")
            log(f"    {sys.executable} -m ensurepip --upgrade")
            log("    brew install python3           # then: python3 -m pip install " + pkg)
            log('    uv pip install "ultralytics[export-coreml]>=8.4.142"')
        raise SystemExit(
            f"ERROR: could not install ultralytics automatically.\n"
            f"       Install it manually into the SAME python you use to run this script:\n"
            f"         {sys.executable} -m pip install {pkg}\n"
            f"       then re-run setup_official_app.py")
    if not (python_has_module("ultralytics") and python_has_module("coremltools")):
        raise SystemExit("ERROR: ultralytics installed but still not importable. "
                         "Check your python environment.")


def ensure_model(model: Path | None, weights: Path, out_dir: Path) -> Path:
    if model:
        model = model.resolve()
        if not model.exists():
            raise SystemExit(f"ERROR: model not found: {model}")
        log(f"[2/5] Using provided model {model}")
        return model

    # Look for an already-exported package: --out first, then next to the
    # weights / script dir (older ultralytics writes it beside the .pt).
    search_dirs = [out_dir, weights.parent, SCRIPT_DIR]
    for directory in search_dirs:
        existing = sorted(directory.glob("*.mlpackage")) if directory.exists() else []
        if existing:
            log(f"[2/5] Found exported model {existing[0]}")
            return existing[0].resolve()

    log("[2/5] Exporting Core ML from weights (needs ultralytics + macOS) ...")
    ensure_ultralytics()
    export_script = SCRIPT_DIR / "export_pose_coreml.py"
    if run([sys.executable, str(export_script), "--weights", str(weights),
            "--out", str(out_dir)]) != 0:
        raise SystemExit("ERROR: Core ML export failed. See export_pose_coreml.py output.")
    existing = sorted(out_dir.glob("*.mlpackage"))
    if not existing:
        raise SystemExit("ERROR: export produced no .mlpackage")
    return existing[0].resolve()


def install_model(model: Path, app_dir: Path, dry: bool) -> None:
    destination_dir = app_dir / MODELS_RELATIVE
    destination = destination_dir / model.name
    log(f"[3/5] Installing model -> {destination}")
    if dry:
        log("  (dry-run, skipped)")
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(model, destination)
    log("  done. The app auto-discovers models in Models/Pose (no registry edit needed).")


# The official app ships YOLOiOSApp.entitlements containing ONLY associated-domains
# (applinks:platform.ultralytics.com, for the Ultralytics HUB). Company provisioning
# profiles rarely include that capability, and signing fails with
# "Provisioning profile doesn't include Associated Domains". Empty the file (keep the
# file itself so CODE_SIGN_ENTITLEMENTS in project.pbxproj stays valid) - nothing in
# the trial depends on universal links.
ENTITLEMENTS_RELATIVE = Path("YOLOiOSApp") / "YOLOiOSApp" / "YOLOiOSApp.entitlements"
ENTITLEMENTS_PLACEHOLDER = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
    '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
    '<plist version="1.0"><dict/>\n</plist>\n'
)


def fix_entitlements(app_dir: Path, dry: bool) -> None:
    entitlements = app_dir / ENTITLEMENTS_RELATIVE
    if not entitlements.exists():
        log(f"[4/5] No entitlements file at {ENTITLEMENTS_RELATIVE} - nothing to fix")
        return
    content = entitlements.read_text(encoding="utf-8")
    if "associated-domains" not in content:
        log(f"[4/5] {ENTITLEMENTS_RELATIVE} has no associated-domains - skipping")
        return
    log(f"[4/5] Stripping associated-domains from {ENTITLEMENTS_RELATIVE} "
        "(company profiles usually lack this capability; the trial does not use it)")
    if dry:
        log("  (dry-run, file not written)")
        return
    entitlements.write_text(ENTITLEMENTS_PLACEHOLDER, encoding="utf-8")
    log("  written. If Xcode still complains, also remove the 'Associated Domains' "
        "capability on the Signing & Capabilities tab.")


def patch_view_controller(vc: Path, dry: bool) -> None:
    log(f"[5/5] Patching {vc}")
    source = vc.read_text(encoding="utf-8")

    if "recognizePlateIfNeeded" in source:
        log("  already patched (recognizePlateIfNeeded present) - skipping")
        return

    # 4a. imports
    lines = source.splitlines(keepends=True)
    last_import = max((i for i, line in enumerate(lines) if line.lstrip().startswith("import ")),
                      default=-1)
    needed = [name for name in ("Vision", "CoreImage")
              if not any(f"import {name}" in line for line in lines)]
    if needed and last_import >= 0:
        insert = "".join(f"import {n}\n" for n in needed)
        lines.insert(last_import + 1, insert)
        source = "".join(lines)
        log(f"  added imports: {', '.join(needed)}")
    elif needed:
        source = "".join(f"import {n}\n" for n in needed) + source
        log(f"  prepended imports: {', '.join(needed)}")

    # 4b. inject the call at the top of didReceiveResult
    match = DID_RECEIVE_RE.search(source)
    if not match:
        raise SystemExit(
            "ERROR: could not find yoloView(_:didReceiveResult:) in ViewController.swift.\n"
            "The official source may have changed. Add this line manually at the top of that\n"
            "method:\n\n" + INJECTED_CALL + "\n")
    source = (source[:match.end(1)] + "\n" + INJECTED_CALL + "\n" + source[match.end(1):])
    log("  injected OCR call into yoloView(_:didReceiveResult:)")

    # 4c. force-load our model at startup
    startup = STARTUP_ANCHOR_RE.search(source)
    if startup:
        source = (source[:startup.end(1)] + "\n" + STARTUP_CALL + "\n"
                  + source[startup.end(1):])
        log("  injected startup force-load into loadSavedModelOrFirstOfficial()")
    else:
        log("  WARNING: loadSavedModelOrFirstOfficial() not found - the app will keep")
        log("           loading the official pose model. Add this line manually as the")
        log("           first statement of that method:")
        log("    " + STARTUP_CALL.strip())

    # 4d. append the implementations
    if not source.endswith("\n"):
        source += "\n"
    source += STARTUP_SWIFT_BLOCK
    source += SWIFT_BLOCK
    log("  appended model force-load + OCR implementation")

    if dry:
        log("  (dry-run, file not written)")
        return
    vc.write_text(source, encoding="utf-8")
    log("  written.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app-dir", type=Path, default=None,
                        help="Path to an existing yolo-ios-app clone (default: clone it)")
    parser.add_argument("--clone-to", type=Path, default=DEFAULT_CLONE_DIR,
                        help="Where to clone the official app (default: ./yolo-ios-app)")
    parser.add_argument("--model", type=Path, default=None,
                        help="Already-exported .mlpackage (default: export from --weights)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="Pose .pt weights used when --model is not given")
    parser.add_argument("--out", type=Path, default=Path("exports"),
                        help="Export output directory (default: ./exports)")
    parser.add_argument("--no-export", action="store_true",
                        help="Do not attempt the Core ML export; fail if no --model given")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing anything")
    args = parser.parse_args()

    log("=== Official YOLO app + German plate OCR setup ===")
    app_dir = ensure_app(args.app_dir, args.clone_to)

    if args.no_export and not args.model:
        raise SystemExit("ERROR: --no-export requires --model")
    model = ensure_model(args.model, args.weights, args.out)

    install_model(model, app_dir, args.dry_run)
    fix_entitlements(app_dir, args.dry_run)
    patch_view_controller(app_dir / VC_RELATIVE, args.dry_run)

    log("\n=== Done ===")
    log(f"App:   {app_dir}")
    log(f"Model: {app_dir / MODELS_RELATIVE / model.name}")
    log("\nNext steps:")
    log(f"  1. open {app_dir / 'YOLOiOSApp' / 'YOLOiOSApp.xcodeproj'}")
    log("  2. Select your iPhone as the run target and press Run")
    log("  3. In the app, choose task Pose, then select De-plate-pose-500")
    log("  4. Point at a plate: 4 coloured dots should sit on the plate corners and the")
    log("     recognised plate text appears at the top of the screen.")
    log("\nNote: nothing was added to the Xcode project file, so no build settings changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
