/*
 License Plate OCR SDK — shared recognition types

 Abstract:
 Types shared by the text-recognition pipeline and the App UI. Kept free of any
 OCR-engine framework import: the SDK uses PaddleOCR exclusively, but these value
 types (observation, protocol, quality mode) are engine-agnostic and are used by
 the detector pipeline, the UI picker and the tests.
 */

import Foundation
import CoreGraphics

/// A recognition candidate produced for a single observation.
public struct PlateObservation: Sendable {
    /// Bottom-left normalized box within the cropped plate image.
    public let boundingBox: CGRect
    /// Top recognized string for the observation.
    public let string: String
    /// Confidence of the top string (0...1).
    public let confidence: Float
    /// Alternative candidates `(string, confidence)`.
    public let candidates: [(text: String, confidence: Float)]
}

/// Abstraction over the text recognizer so it can be swapped/tested.
public protocol PlateTextRecognizing: Sendable {
    /// Recognizes text in the given image region using the requested mode.
    func recognize(in image: CGImage, mode: RecognitionMode) async throws -> [PlateObservation]
}

/// The recognition quality the user can choose. Defaults to `.fast`.
///
/// PaddleOCR implements both levels itself (no engine switch):
/// - `.fast`: one inference, beam width 8.
/// - `.accurate`: three crops + beam width 16 (~4.7x cost).
public enum RecognitionMode: String, CaseIterable, Sendable {
    case fast
    case accurate
}
