/*
 License Plate OCR SDK — PaddleOCR bundle resources

 Abstract:
 Locates the PP-OCRv5 recognition model and its CTC character dictionary in the app
 bundle (or in `Bundle.module` for the SwiftPM target). Kept free of any ONNX Runtime
 import so `OCREngine.isAvailable` can be evaluated in builds that do not link ORT.

 Model provenance
 ----------------
 `PP-OCRv5_rec_mobile_infer.onnx` (16.5 MB) and `ppocrv5_dict.txt` (18,383 characters)
 ship with the Apache-2.0 `imgocr` 0.2.4 distribution, which converts PaddleOCR's
 PP-OCRv5 mobile weights with Paddle2ONNX. PaddleOCR itself is Apache-2.0 licensed.
 Re-fetch with `Scripts/paddleocr_rec_reference.py --fetch-models`.

 Why the lookup is defensive
 ---------------------------
 The Xcode target uses `fileSystemSynchronizedGroups` and `PBXResourcesBuildPhase.files`
 is empty, so the relative path Xcode preserves for the `LicensePlateSDK/Resources`
folder is
 not guaranteed. A flat `url(forResource:)` returning nil used to make
 `OCREngine.defaultEngine` silently fall back to Apple Vision — the App then claimed
 nothing was wrong while the entire PaddleOCR branch was dead. We now probe several
 layouts and fall back to a bounded recursive scan; if both resources still cannot be
 found, `OCREngine.makeRecognizer` returns a recognizer that throws an explicit error
 instead of quietly running Vision.
 */

import Foundation

public enum PaddleOCRResources {

    public static let recognitionModelName = "PP-OCRv5_rec_mobile_infer"
    public static let characterDictName = "ppocrv5_dict"

    /// Bundle holding the SDK resources. Xcode 16 synchronized folders copy
    /// folder contents into `Bundle.main`; the SwiftPM target uses
    /// `Bundle.module` (`resources: [.process("Resources")]` in `Package.swift`).
    public static var resourceBundle: Bundle {
        #if SWIFT_PACKAGE
        return Bundle.module
        #else
        return Bundle.main
        #endif
    }

    /// Relative directories Xcode has been observed to preserve for synchronized
    /// folder groups. `nil` is the bundle root.
    private static let candidateDirectories: [String?] = [
        nil,
        "Resources",
        "LicensePlateSDK/Resources",
        "LicensePlateSDK",
        "Contents/Resources"
    ]

    private static func locate(_ name: String, extension ext: String) -> URL? {
        let bundle = resourceBundle
        for directory in candidateDirectories {
            if let url = bundle.url(forResource: name, withExtension: ext,
                                    subdirectory: directory) {
                return url
            }
        }

        // Last resort: Xcode may keep an arbitrary relative path inside a
        // synchronized group. Apple documents `subdirectory` as a TOP-LEVEL bundle
        // directory, so `url(forResource:)` cannot find a resource nested several
        // levels deep. Scan only the resource directory (not the whole bundle),
        // skip code / framework / localization containers, and BOUND the work.
        // This is the exceptional path — normal builds hit a candidate above — and
        // an unbounded enumeration on the launch path would violate Apple's
        // launch-time guidance. The resolved URL is cached below.
        let filename = "\(name).\(ext)"
        let root = bundle.resourceURL ?? bundle.bundleURL
        guard let enumerator = FileManager.default.enumerator(
                at: root,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles, .skipsPackageDescendants]) else { return nil }
        let skipped: Set<String> = ["Frameworks", "PlugIns", "Watch",
                                    "_CodeSignature", "SC_Info"]
        let rootDepth = root.pathComponents.count
        var inspected = 0
        for case let candidate as URL in enumerator {
            let relative = candidate.pathComponents.dropFirst(rootDepth)
            if relative.contains(where: { skipped.contains($0) || $0.hasSuffix(".lproj") }) {
                continue
            }
            if candidate.lastPathComponent == filename { return candidate }
            inspected += 1
            if inspected >= 2_000 { break }
        }
        return nil
    }

    /// Resolved once, lazily and thread-safely (Swift static `let`).
    private static let resolvedModel = locate(recognitionModelName, extension: "onnx")
    private static let resolvedDictionary = locate(characterDictName, extension: "txt")

    public static var modelURL: URL? { resolvedModel }
    public static var dictionaryURL: URL? { resolvedDictionary }

    /// `true` when both the ONNX graph and the CTC dictionary are packaged. The App
    /// uses this to hide the PaddleOCR option instead of claiming an engine that would
    /// silently fall back to Apple Vision.
    public static var areBundled: Bool { modelURL != nil && dictionaryURL != nil }

    /// Human-readable engine state for the diagnostics panel. Naming the bundle makes
    /// a failed lookup actionable on device instead of a silent downgrade.
    public static var statusDescription: String {
        let bundleName = resourceBundle.bundleURL.lastPathComponent
        guard modelURL != nil else {
            return "PaddleOCR 识别模型未打包（\(recognitionModelName).onnx；bundle: \(bundleName)）"
        }
        guard dictionaryURL != nil else {
            return "PaddleOCR 字典未打包（\(characterDictName).txt；bundle: \(bundleName)）"
        }
        return "PaddleOCR PP-OCRv5 mobile · ONNX Runtime · 车牌字符约束 CTC"
    }
}
