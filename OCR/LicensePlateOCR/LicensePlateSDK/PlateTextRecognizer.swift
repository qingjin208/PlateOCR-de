/*
 License Plate OCR SDK
 
 Abstract:
 Wraps Apple's Vision `RecognizeTextRequest` — the same engine used by Apple's
 "Locating and displaying recognized text" sample — to read the characters inside a
 (already cropped) license-plate region. Supports the `fast` / `accurate` levels the
 user can pick, defaulting to `fast`.
 */

import Foundation
import CoreGraphics
import Vision

/// A recognition candidate produced by Vision for a single observation.
public struct PlateObservation: Sendable {
    /// Top recognized string for the observation.
    public let string: String
    /// Confidence of the top string (0...1).
    public let confidence: Float
    /// Up to 10 alternative candidates `(string, confidence)`.
    public let candidates: [(text: String, confidence: Float)]
}

/// Abstraction over the text recognizer so the engine can be swapped/tested.
public protocol PlateTextRecognizing: Sendable {
    /// Recognizes text in the given image region using the requested mode.
    func recognize(in image: CGImage, mode: RecognitionMode) async throws -> [PlateObservation]
}

/// The recognition quality the user can choose. Defaults to `.fast`.
public enum RecognitionMode: String, CaseIterable, Sendable {
    case fast
    case accurate

    /// Maps to Vision's `RecognitionLevel`.
    var visionLevel: RecognizeTextRequest.RecognitionLevel {
        switch self {
        case .fast: return .fast
        case .accurate: return .accurate
        }
    }
}

/// Apple Vision `RecognizeTextRequest` backed recognizer.
public final class PlateTextRecognizer: PlateTextRecognizing {
    private let languages: [Locale.Language]

    /// - Parameter languages: Recognition languages, in priority order.
    ///   Defaults to German (`de-DE`), which covers EU license-plate glyphs best.
    public init(languages: [Locale.Language] = [Locale.Language(identifier: "de-DE")]) {
        self.languages = languages
    }

    public func recognize(in image: CGImage, mode: RecognitionMode) async throws -> [PlateObservation] {
        var request = try RecognizeTextRequest()
        request.recognitionLevel = mode.visionLevel

        // Only keep languages Vision actually supports on this device.
        let supported = request.supportedRecognitionLanguages
        let chosen = languages.filter { supported.contains($0) }
        if !chosen.isEmpty { request.recognitionLanguages = chosen }

        // Plates are not natural language — disable correction so it never rewrites glyphs.
        request.usesLanguageCorrection = false

        let observations: [RecognizedTextObservation]
        do {
            observations = try await request.perform(on: image)
        } catch {
            throw LicensePlateError.recognitionFailed(error.localizedDescription)
        }

        return observations.map { obs in
            let top = obs.topCandidates(10)
            let primary = top.first
            let candidates = top.map { (text: $0.string, confidence: Float($0.confidence)) }
            return PlateObservation(
                string: primary?.string ?? "",
                confidence: Float(primary?.confidence ?? 0),
                candidates: candidates
            )
        }
    }
}
