"""Local validation of setup_official_app.py's patch logic.

We cannot clone the real yolo-ios-app here (network blocked), so we exercise the
patching against a synthetic ViewController.swift that mirrors the official structure
(imports block + YOLOViewDelegate extension with yoloView(_:didReceiveResult:)).
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import setup_official_app as setup

FAKE_VC = """import Foundation
import UIKit
import UltralyticsYOLO

final class ViewController: UIViewController {
    @IBOutlet weak var yoloView: YOLOView!
    var currentTask: String = "Pose"
    var breakdownText: String = ""
    var fpsText: String = ""

    override func viewDidLoad() {
        super.viewDidLoad()
        yoloView.delegate = self
    }

    func renderPerformanceLabel() {}

    func reloadModelEntriesAndLoadFirst(for taskName: String) {}

    private func loadSavedModelOrFirstOfficial() {
        reloadModelEntriesAndLoadFirst(for: currentTask)
    }
}

// MARK: - YOLOViewDelegate
extension ViewController {

    func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {}

    func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
        let breakdownTotal = result.preMs + result.inferenceMs + result.postMs
        let text =
            breakdownTotal > 0
            ? String(format: "%.1f pre", result.preMs)
            : ""
        DispatchQueue.main.async { [weak self] in
            self?.breakdownText = text
            self?.renderPerformanceLabel()
        }
    }
}
"""


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    vc = tmp / "ViewController.swift"
    vc.write_text(FAKE_VC, encoding="utf-8")

    print("--- patching synthetic ViewController.swift ---")
    setup.patch_view_controller(vc, dry=False)
    result = vc.read_text(encoding="utf-8")

    checks = {
        "Vision import added": "import Vision" in result,
        "CoreImage import added": "import CoreImage" in result,
        "imports after existing imports (not top of file)": result.index("import Vision") > result.index("import Foundation"),
        "OCR call injected into didReceiveResult": "recognizePlateIfNeeded(result: result, in: view)" in result,
        "injected inside the method body": "func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {\n" in result,
        "PlateOCRService appended": "final class PlateOCRService" in result,
        "GermanPlateValidator appended": "enum GermanPlateValidator" in result,
        "orderCorners appended": "func plateOrderCorners" in result,
        "extension hook appended": "func recognizePlateIfNeeded(result: YOLOResult, in view: YOLOView)" in result,
        "original content preserved": "@IBOutlet weak var yoloView: YOLOView!" in result,
        "didUpdatePerformance untouched": "func yoloView(_ view: YOLOView, didUpdatePerformance" in result,
        "startup force-load injected": "if plateLoadPoseModel() { return }" in result,
        "startup call is first statement of the method": (
            "private func loadSavedModelOrFirstOfficial() {\n"
            "        if plateLoadPoseModel() { return }\n" in result),
        "force-load implementation appended": "func plateLoadPoseModel() -> Bool" in result,
        "bundle lookup helper appended": "func plateBundledModelURL(named name: String) -> URL?" in result,
        "model name constant appended": '_platePoseModelName = "De-plate-pose-500"' in result,
    }

    print()
    ok = True
    for name, passed in checks.items():
        print(("  PASS  " if passed else "  FAIL  ") + name)
        ok = ok and passed

    # Show the patched delegate method.
    print("\n--- patched didReceiveResult ---")
    start = result.index("func yoloView(_ view: YOLOView, didReceiveResult")
    end = result.index("\n    }\n", start) + len("\n    }\n")
    print(result[start:end])

    # Idempotency.
    print("\n--- idempotency check (re-run) ---")
    before = vc.read_text(encoding="utf-8")
    setup.patch_view_controller(vc, dry=False)
    after = vc.read_text(encoding="utf-8")
    idem = before == after
    print(("  PASS  " if idem else "  FAIL  ") + "second run leaves file unchanged")
    ok = ok and idem

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
