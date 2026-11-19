/*
 License Plate OCR SDK
 
 Abstract:
 Shared error types for the license plate detection + recognition pipeline.
 */

import Foundation

/// Errors that can be thrown by the license plate SDK.
public enum LicensePlateError: LocalizedError {
    /// The YOLO ONNX pose model could not be found in the app bundle.
    /// Keep `De-plate-pose.onnx` in `LicensePlateOCR/LicensePlateSDK/Resources/`
    /// and rebuild (Xcode 16's synchronized folders copy it automatically).
    case modelNotFound(String)
    /// A model output tensor (pose detector or PaddleOCR rec) could not be parsed.
    case invalidModelOutput(String)
    /// Text recognition failed.
    case recognitionFailed(String)
    /// PaddleOCR could not be initialized (missing ONNX model/dictionary, or ORT
    /// session creation failed). Reported explicitly instead of being swallowed.
    case paddleOCRUnavailable(String)
    /// The provided media could not be read.
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Plate detection model \(name) not found. Make sure the ONNX model under LicensePlateOCR/LicensePlateSDK/Resources/ is bundled with the app (Xcode 16 synchronized folders copy it automatically)."
        case .invalidModelOutput(let detail):
            return "Could not parse model output: \(detail)"
        case .recognitionFailed(let detail):
            return "Text recognition failed: \(detail)"
        case .paddleOCRUnavailable(let detail):
            return "PaddleOCR engine unavailable: \(detail). Make sure PP-OCRv5_rec_mobile_infer.onnx and ppocrv5_dict.txt are bundled with the app."
        case .invalidInput(let detail):
            return "Invalid input media: \(detail)"
        }
    }
}
