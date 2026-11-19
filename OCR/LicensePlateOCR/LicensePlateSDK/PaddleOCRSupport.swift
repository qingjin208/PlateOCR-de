/*
 License Plate OCR SDK — PaddleOCR support

 Abstract:
 The SDK uses PaddleOCR exclusively for plate text recognition. PaddleOCR here is
 NOT a separate iOS framework: it reuses the linked ONNX Runtime to execute the
 PP-OCRv5 mobile **recognition** graph
 (`PP-OCRv5_rec_mobile_infer.onnx` + `ppocrv5_dict.txt`). The plate keypoints
 already deliver a rectified text line, so PaddleOCR's DB text detector is not used.

 This file holds the single-engine factory plus the German CTC decoder alphabet
 and the last-frame diagnostic box. The engine-agnostic value types (observation,
 protocol, quality mode) live in `OCRTypes.swift`.
 */

import Foundation
import CoreGraphics
import os

/// PaddleOCR is the SDK's only text-recognition engine. The factory keeps the
/// resource check and the fail-loud stub so a build missing its model/dictionary
/// reports the reason instead of producing silently wrong readings.
public enum PaddleOCR {
    /// True when ONNX Runtime is linked and both the model and dictionary are bundled.
    public static var isAvailable: Bool {
        #if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc) || canImport(onnxruntime_objc)
        return PaddleOCRResources.areBundled
        #else
        return false
        #endif
    }

    /// Builds the recognizer. Missing resources yield a stub that throws
    /// `paddleOCRUnavailable` on the first frame — no silent downgrade.
    public static func makeRecognizer() -> PlateTextRecognizing {
        #if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc) || canImport(onnxruntime_objc)
        if PaddleOCRResources.areBundled { return PaddleOCRTextRecognizer() }
        #endif
        return PaddleOCRUnavailableRecognizer(reason: PaddleOCRResources.statusDescription)
    }

    /// Warms the model AFTER the first frame is on screen. Apple's "Reducing your
    /// app's launch time" keeps the main thread free for UIKit's first frame and
    /// defers expensive initialization to first use. Call from the view's `.task`.
    public static func prewarm() async {
        #if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc) || canImport(onnxruntime_objc)
        await PaddleOCRTextRecognizer.prewarm()
        #endif
    }

    /// Stable label for the on-device diagnostics line.
    public static let displayName = "PaddleOCR"
}

/// Returned when PaddleOCR's model/dictionary are missing in the current build.
/// Fails loudly instead of running a silent fallback under the PaddleOCR label.
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
        /// Last two-row decision: split row, row readings and why a split may have
        /// been rejected. Read-only mirror; it never affects recognition.
        var twoRow: String?
    }

    /// `OSAllocatedUnfairLock` is Apple's supported Swift wrapper for
    /// `os_unfair_lock`; unlike a manual `NSLock.lock()/unlock()` pair it cannot
    /// accidentally be held across an `await`.
    private final class Box: @unchecked Sendable {
        private let state = OSAllocatedUnfairLock(initialState: State())

        func record(_ newReadings: [String], _ twoRow: String?) {
            state.withLock { current in
                current.readings = newReadings
                let stray = Set(newReadings.joined().map { String($0) })
                    .subtracting(PlateOCRAlphabet.allowedSet)
                current.strayCharacters = stray.isEmpty ? nil : stray.sorted().joined()
                current.twoRow = twoRow
            }
        }

        func summary() -> String? {
            state.withLock { current -> String? in
                guard let reading = current.readings.first, !reading.isEmpty else { return nil }
                var line = "OCR 原始读数 \(reading)"
                if let stray = current.strayCharacters {
                    line += " · 异常字符 \(stray)（清洁未彻底？）"
                }
                if let twoRow = current.twoRow { line += " · 两行 \(twoRow)" }
                return line
            }
        }
    }

    private static let box = Box()

    public static func record(greedyReadings: [String], twoRow: String? = nil) {
        box.record(greedyReadings, twoRow)
    }

    /// `nil` before the first PaddleOCR frame. The caller decides whether to show it.
    public static var summary: String? { box.summary() }
}
