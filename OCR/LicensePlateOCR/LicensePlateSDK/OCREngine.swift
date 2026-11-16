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
import os

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
    /// - Note: `languages` is an Apple Vision option only. PaddleOCR's PP-OCRv5 rec
    ///   head has no runtime language switch — its character set is fixed by the ONNX
    ///   graph and dictionary. The German constraint is applied by the CTC decoder
    ///   and `GermanPlateValidator`, not by a locale.
    ///
    /// When `.paddleOCR` is requested but its resources are missing, this returns a
    /// stub that throws `paddleOCRUnavailable` on the first frame. It deliberately
    /// does NOT fall back to Vision: a silent downgrade made the whole branch look
    /// healthy while Apple Vision was doing the reading. `defaultEngine` still starts
    /// on Vision when resources are absent, and the diagnostics row explains why.
    public func makeRecognizer(languages: [Locale.Language] = [Locale.Language(identifier: "de-DE")]) -> PlateTextRecognizing {
        switch self {
        case .appleVision:
            return PlateTextRecognizer(languages: languages)
        case .paddleOCR:
            #if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc) || canImport(onnxruntime_objc)
            if PaddleOCRResources.areBundled { return PaddleOCRTextRecognizer() }
            #endif
            return PaddleOCRUnavailableRecognizer(reason: PaddleOCRResources.statusDescription)
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

    /// Warms the selected engine's model AFTER the first frame is on screen.
    ///
    /// Apple's "Reducing your app's launch time" requires the main thread to stay
    /// available for UIKit's first frame and recommends deferring expensive
    /// initialization to first use. Call this from the view's `.task`, not `init`.
    public static func prewarm(_ engine: OCREngine) async {
        #if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc) || canImport(onnxruntime_objc)
        if engine == .paddleOCR { await PaddleOCRTextRecognizer.prewarm() }
        #else
        _ = engine
        #endif
    }
}

/// Returned when `.paddleOCR` is selected in a build without its model/dictionary.
/// Fails loudly instead of silently running Apple Vision under the PaddleOCR label.
private struct PaddleOCRUnavailableRecognizer: PlateTextRecognizing {
    let reason: String

    func recognize(in image: CGImage, mode: RecognitionMode) async throws -> [PlateObservation] {
        throw LicensePlateError.paddleOCRUnavailable(reason)
    }
}

/// The German plate character set used to constrain CTC decoding.
///
/// This is a DECODER CONSTRAINT, not a dictionary. `ppocrv5_dict.txt` is an
/// index→character lookup table bound to the model's fixed 18,385-class output
/// head and must never be truncated to these 41 characters — that shifts every
/// class index (`B` is class 16180) and breaks decoding outright. Physically
/// trimming the ONNX output head is a separate, model-editing operation.
public enum PlateOCRAlphabet {
    /// A-Z, 0-9, the three umlauts that occur in city codes, the canonical
    /// separator and the space the model may emit. `ß` is excluded (FZV writes it
    /// as SS); the `E`/`H` suffixes need no extra entry because their position is
    /// validated by `GermanPlateValidator`.
    public static let characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-ÄÖÜ "

    public static let allowedSet: Set<String> = Set(characters.map { String($0) })
}

/// Thread-safe, last-frame diagnostics for the on-device debug panel.
///
/// The full 18,385-class greedy reading is deliberately NOT a recognition
/// candidate: when it contains characters outside `PlateOCRAlphabet`, that is
/// evidence the cleaned plate crop still carries interference (EU strip, seal,
/// border). The constrained beam would silently swallow those characters and
/// return a confident but wrong string, so we keep this signal separately.
public enum PaddleOCRDiagnostics {
    private struct State {
        var readings: [String] = []
        var strayCharacters: String?
    }

    /// `OSAllocatedUnfairLock` is Apple's supported Swift wrapper for
    /// `os_unfair_lock`; unlike a manual `NSLock.lock()/unlock()` pair it cannot
    /// accidentally be held across an `await`.
    private final class Box: @unchecked Sendable {
        private let state = OSAllocatedUnfairLock(initialState: State())

        func record(_ newReadings: [String]) {
            state.withLock { current in
                current.readings = newReadings
                let stray = Set(newReadings.joined().map { String($0) })
                    .subtracting(PlateOCRAlphabet.allowedSet)
                current.strayCharacters = stray.isEmpty ? nil : stray.sorted().joined()
            }
        }

        func summary() -> String? {
            state.withLock { current -> String? in
                guard let reading = current.readings.first, !reading.isEmpty else { return nil }
                if let stray = current.strayCharacters {
                    return "OCR 原始读数 \(reading) · 异常字符 \(stray)（清洁未彻底？）"
                }
                return "OCR 原始读数 \(reading)"
            }
        }
    }

    private static let box = Box()

    public static func record(greedyReadings: [String]) { box.record(greedyReadings) }

    /// `nil` before the first PaddleOCR frame. The caller decides whether to show it.
    public static var summary: String? { box.summary() }
}
