/*
 License Plate OCR SDK — OCR engine selection

 Abstract:
 The App can choose the OCR engine ("Apple Vision" or "PaddleOCR") and the SDK
 switches engines at runtime. Both are `PlateTextRecognizing` implementations, so
 the detection + perspective-correction + validation pipeline is identical.

 PaddleOCR here is NOT a separate iOS framework: it reuses the already-linked
 ONNX Runtime to execute PaddleOCR's PP-OCRv5 mobile **recognition** graph
 (`PP-OCRv5_rec_mobile_infer.onnx` + `ppocrv5_dict.txt`). The plate keypoints
 already deliver a rectified text line, so PaddleOCR's DB text detector is not
 used. Apple Vision remains selectable for A/B comparison on device.
 */

import Foundation
import CoreGraphics

/// OCR engine used to read the plate characters.
public enum OCREngine: String, CaseIterable, Sendable {
    case appleVision
    case paddleOCR

    /// User-facing label for the in-App picker.
    public var displayName: String {
        switch self {
        case .appleVision: return "Apple Vision"
        case .paddleOCR: return "PaddleOCR"
        }
    }

    /// The engine the App starts with. PaddleOCR is the default once its model and
    /// dictionary are packaged; if they are missing (e.g. a build that skipped the
    /// Resources folder) the App falls back to Vision instead of binding the picker
    /// to an engine that is not linked.
    public static var defaultEngine: OCREngine {
        paddleOCR.isAvailable ? .paddleOCR : .appleVision
    }

    /// Builds the recognizer backing this engine.
    ///
    /// `.paddleOCR` returns a real ONNX Runtime recognizer when ORT is linked and
    /// both resources are bundled. Otherwise it returns Apple Vision — but
    /// `isAvailable` is `false` in that case, so the App hides the option and never
    /// claims PaddleOCR is running when it is not.
    public func makeRecognizer(languages: [Locale.Language] = [Locale.Language(identifier: "de-DE")]) -> PlateTextRecognizing {
        switch self {
        case .appleVision:
            return PlateTextRecognizer(languages: languages)
        case .paddleOCR:
            #if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc) || canImport(onnxruntime_objc)
            if PaddleOCRResources.areBundled { return PaddleOCRTextRecognizer() }
            #endif
            return PlateTextRecognizer(languages: languages)
        }
    }

    /// Whether this engine is actually usable in the current build.
    public var isAvailable: Bool {
        switch self {
        case .appleVision:
            return true
        case .paddleOCR:
            #if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc) || canImport(onnxruntime_objc)
            return PaddleOCRResources.areBundled
            #else
            return false
            #endif
        }
    }
}
