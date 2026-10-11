/*
 License Plate OCR SDK
 
 Abstract:
 The public entry point of the SDK. Locates German license plates with a YOLO model,
 then reads each plate with Apple's Vision OCR engine (the same one Apple demonstrates in
 "Locating and displaying recognized text"), and cleans the result with German-plate
 post-processing. Works on a single image or on a video.
 */

import Foundation
import CoreGraphics
import AVFoundation

/// A recognized plate.
public struct PlateResult: Identifiable, Sendable {
    public let id = UUID()
    /// Canonical, format-validated plate text, e.g. `B-AB 1234`.
    public let text: String
    /// Confidence of the chosen plate (0...1).
    public let confidence: Float
    /// Detection bounding box, normalized (top-left origin, 0...1).
    public let boundingBox: CGRect
    /// Size of the source image the box refers to.
    public let imageSize: CGSize
    /// Top alternative readings (raw OCR candidates, cleaned).
    public let candidates: [PlateCandidate]
}

/// A single alternative reading of a plate.
public struct PlateCandidate: Sendable, Identifiable {
    public let id = UUID()
    public let text: String
    public let confidence: Float
}

/// The full recognition pipeline: detect -> crop -> recognize -> post-process.
public protocol LicensePlateRecognizing: Sendable {
    /// Recognizes plates in a still image.
    func recognize(in image: CGImage, mode: RecognitionMode) async throws -> [PlateResult]
    /// Recognizes plates across a video, sampling frames over time.
    func recognizeVideo(at url: URL, mode: RecognitionMode, maxSamples: Int,
                        progress: (@Sendable (Double) -> Void)?) async throws -> [PlateResult]
}

/// Default implementation wiring together the detector, recognizer and post-processor.
public final class LicensePlateRecognizer: LicensePlateRecognizing {

    private let detector: LicensePlateDetecting
    private let recognizer: PlateTextRecognizing
    private let postProcessor: PlatePostProcessor

    public init(detector: LicensePlateDetecting,
                recognizer: PlateTextRecognizing,
                postProcessor: PlatePostProcessor = PlatePostProcessor()) {
        self.detector = detector
        self.recognizer = recognizer
        self.postProcessor = postProcessor
    }

    /// Convenience initializer for the common setup: Core ML YOLO detector + Vision OCR,
    /// German (`de-DE`) recognition.
    /// - Parameters:
    ///   - modelName: Compiled YOLO model resource name (default `"license_plate_yolo"`).
    ///   - config: YOLO output layout (default `.germanPlate`).
    ///   - recognitionLanguages: OCR languages (default `[de-DE]`).
    public convenience init(modelName: String = "license_plate_yolo",
                            config: YOLOConfig = .germanPlate,
                            recognitionLanguages: [Locale.Language] = [Locale.Language(identifier: "de-DE")]) {
        self.init(detector: (try? CoreMLLicensePlateDetector(modelName: modelName, config: config))
                    ?? CoreMLLicensePlateDetectorFailing(),
                  recognizer: PlateTextRecognizer(languages: recognitionLanguages),
                  postProcessor: PlatePostProcessor())
    }

    public func recognize(in image: CGImage, mode: RecognitionMode = .fast) async throws -> [PlateResult] {
        let detections = try await detector.detect(in: image)
        var results: [PlateResult] = []
        results.reserveCapacity(detections.count)

        for detection in detections {
            // 1. Crop the plate region (with a little margin to include the full plate).
            let crop = image.cropRegion(detection.rect, padding: 0.05)
            // 2. Reduce interference (screws, borders, reflections) before OCR.
            let enhanced = crop.enhancedForOCR()
            // 3. Read the characters with Vision.
            let observations = try await recognizer.recognize(in: enhanced, mode: mode)
            // 4. Validate + clean + pick the best candidate.
            if let result = postProcessor.process(observations: observations,
                                                   region: detection.rect,
                                                   imageSize: CGSize(width: image.width, height: image.height)) {
                results.append(result)
            }
        }
        return results
    }

    public func recognizeVideo(at url: URL, mode: RecognitionMode = .fast,
                               maxSamples: Int = 30,
                               progress: (@Sendable (Double) -> Void)? = nil) async throws -> [PlateResult] {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        guard duration > 0 else { throw LicensePlateError.invalidInput("视频时长为 0") }

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        let step = max(0.1, duration / Double(max(1, maxSamples)))
        var collected: [PlateResult] = []
        var seen = 0

        for t in stride(from: 0.0, to: duration, by: step) {
            let time = CMTime(seconds: t, preferredTimescale: 600)
            let frame: CGImage
            do {
                let (cg, _) = try await generator.image(at: time)
                frame = cg
            } catch {
                continue
            }
            let frameResults = try await recognize(in: frame, mode: mode)
            collected.append(contentsOf: frameResults)
            seen += 1
            progress?(Double(seen) / Double(max(1, Int(duration / step))))
        }
        return PlatePostProcessor.deduplicate(collected)
    }
}

/// Placeholder used when the model is missing, so the convenience initializer never
/// crashes at init — failures surface as a clear error when detection runs.
private final class CoreMLLicensePlateDetectorFailing: LicensePlateDetecting {
    func detect(in image: CGImage) async throws -> [PlateDetection] {
        throw LicensePlateError.modelNotFound("license_plate_yolo")
    }
}
