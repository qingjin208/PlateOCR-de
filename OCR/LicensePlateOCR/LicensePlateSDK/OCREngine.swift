/*
 License Plate OCR SDK — OCR engine selection

 Abstract:
 The App lets the user choose the OCR engine ("Apple Vision" or "PaddleOCR") and the
 SDK switches engines at runtime. Each engine is a `PlateTextRecognizing` implementation,
 so the detection + correction pipeline is identical regardless of which engine reads
 the (rectified) plate characters.
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

    /// Builds the recognizer backing this engine. With `PaddleOCR-iOS` not linked
    /// (default build), `.paddleOCR` gracefully falls back to Apple Vision — check
    /// `isAvailable` first if the caller needs to know whether it really got PaddleOCR,
    /// because a silent fallback makes the UI claim an engine that is not running.
    public func makeRecognizer(languages: [Locale.Language] = [Locale.Language(identifier: "de-DE")]) -> PlateTextRecognizing {
        switch self {
        case .appleVision:
            return PlateTextRecognizer(languages: languages)
        case .paddleOCR:
            #if canImport(PaddleOCR)
            return PaddleOCRTextRecognizer()
            #else
            return PlateTextRecognizer(languages: languages)
            #endif
        }
    }

    /// Whether this engine is actually linked into the current build. `false` means
    /// `makeRecognizer` returns an Apple Vision recognizer instead.
    public var isAvailable: Bool {
        switch self {
        case .appleVision:
            return true
        case .paddleOCR:
            #if canImport(PaddleOCR)
            return true
            #else
            return false
            #endif
        }
    }
}
