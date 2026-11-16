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
 */

import Foundation

public enum PaddleOCRResources {

    public static let recognitionModelName = "PP-OCRv5_rec_mobile_infer"
    public static let characterDictName = "ppocrv5_dict"

    /// Bundle holding the SDK resources. Xcode 16 synchronized folders copy
    /// `LicensePlateSDK/Resources/*` into `Bundle.main`; the SwiftPM target uses
    /// `Bundle.module` (`resources: [.process("Resources")]` in `Package.swift`).
    public static var resourceBundle: Bundle {
        #if SWIFT_PACKAGE
        return Bundle.module
        #else
        return Bundle.main
        #endif
    }

    /// Resolves a resource both at the bundle root and inside a `Resources/`
    /// subdirectory (folder references preserve the relative path).
    private static func url(for name: String, extension ext: String) -> URL? {
        let bundle = resourceBundle
        if let url = bundle.url(forResource: name, withExtension: ext) { return url }
        return bundle.url(forResource: name, withExtension: ext, subdirectory: "Resources")
    }

    public static var modelURL: URL? { url(for: recognitionModelName, extension: "onnx") }
    public static var dictionaryURL: URL? { url(for: characterDictName, extension: "txt") }

    /// `true` when both the ONNX graph and the CTC dictionary are packaged. The App
    /// uses this to hide the PaddleOCR option instead of claiming an engine that would
    /// silently fall back to Apple Vision.
    public static var areBundled: Bool { modelURL != nil && dictionaryURL != nil }

    /// Human-readable engine state for the diagnostics panel.
    public static var statusDescription: String {
        guard modelURL != nil else { return "PaddleOCR 识别模型未打包（PP-OCRv5_rec_mobile_infer.onnx）" }
        guard dictionaryURL != nil else { return "PaddleOCR 字典未打包（ppocrv5_dict.txt）" }
        return "PaddleOCR PP-OCRv5 mobile · ONNX Runtime · 车牌字符约束 CTC"
    }
}
