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
import CoreImage
import ImageIO

/// Boxes and validated OCR results for one oriented frame. Coordinates start at the top-left.
public struct PlateFrameResult: Sendable {
    public let detections: [PlateDetection]
    public let plates: [PlateResult]
    public let imageSize: CGSize
}

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
    /// Accepts caller-owned BGRA/NV12 frames without creating a capture session.
    func recognizeFrame(_ pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation,
                        mode: RecognitionMode,
                        onDetections: (@Sendable ([PlateDetection], CGSize) -> Void)?) async throws -> PlateFrameResult
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
    private let context = CIContext(options: [.cacheIntermediates: false])

    /// The default pipeline uses the original plate model converted to Core ML and Apple OCR.
    public convenience init() {
        let detector: LicensePlateDetecting
        do { detector = try CoreMLLicensePlateDetector() }
        catch { detector = CoreMLLicensePlateDetectorFailing(error: error) }
        self.init(detector: detector, recognizer: PlateTextRecognizer())
    }

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
    ///   - modelName: Resource name for an explicitly supplied Core ML YOLO model.
    ///   - config: YOLO output layout (default `.germanPlate`).
    ///   - recognitionLanguages: OCR languages (default `[de-DE]`).
    public convenience init(modelName: String,
                            config: YOLOConfig = .germanPlate,
                            recognitionLanguages: [Locale.Language] = [Locale.Language(identifier: "de-DE")]) {
        let detector: LicensePlateDetecting
        do { detector = try CoreMLLicensePlateDetector(modelName: modelName, config: config) }
        catch { detector = CoreMLLicensePlateDetectorFailing(error: error) }
        self.init(detector: detector,
                  recognizer: PlateTextRecognizer(languages: recognitionLanguages),
                  postProcessor: PlatePostProcessor())
    }

    public func recognize(in image: CGImage, mode: RecognitionMode = .fast) async throws -> [PlateResult] {
        // Still images enter the same external-frame pipeline as video and live capture.
        let buffer = try makePixelBuffer(from: CIImage(cgImage: image))
        return try await recognizeFrame(buffer, mode: mode).plates
    }

    /// Converts an upright still image to an external BGRA frame, scaling oversized images.
    public func prepareFrame(from image: CGImage) throws -> CVPixelBuffer {
        try makePixelBuffer(from: CIImage(cgImage: image))
    }

    public func recognizeFrame(_ pixelBuffer: CVPixelBuffer,
                               orientation: CGImagePropertyOrientation = .up,
                               mode: RecognitionMode = .fast,
                               onDetections: (@Sendable ([PlateDetection], CGSize) -> Void)? = nil) async throws -> PlateFrameResult {
        let width = CVPixelBufferGetWidth(pixelBuffer), height = CVPixelBufferGetHeight(pixelBuffer)
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        guard width > 0, height > 0, max(width, height) <= 3840, min(width, height) <= 2160,
              [kCVPixelFormatType_32BGRA, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
               kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange].contains(format) else {
            throw LicensePlateError.invalidInput("Expected BGRA or NV12, up to 3840 x 2160 (or portrait equivalent).")
        }
        try Task.checkCancellation()
        let ci = CIImage(cvPixelBuffer: pixelBuffer).oriented(orientation)
        guard let image = context.createCGImage(ci, from: ci.extent) else {
            throw LicensePlateError.invalidInput("Could not render the input frame.")
        }
        let imageSize = CGSize(width: image.width, height: image.height)
        let detections = try await detector.detect(in: image)
        onDetections?(detections, imageSize)
        var results: [PlateResult] = []
        results.reserveCapacity(detections.count)

        for detection in detections {
            try Task.checkCancellation()
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
        return PlateFrameResult(detections: detections, plates: results, imageSize: imageSize)
    }

    private func makePixelBuffer(from image: CIImage) throws -> CVPixelBuffer {
        // Scale oversized photos down while preserving their aspect ratio.
        let extent = image.extent
        let scale = min(1, 3840 / max(extent.width, extent.height), 2160 / min(extent.width, extent.height))
        let normalized = image.transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, max(1, Int(normalized.extent.width)),
            max(1, Int(normalized.extent.height)), kCVPixelFormatType_32BGRA,
            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer)
        guard status == kCVReturnSuccess, let buffer else {
            throw LicensePlateError.invalidInput("Could not allocate a media frame.")
        }
        context.render(normalized, to: buffer)
        return buffer
    }

    public func recognizeVideo(at url: URL, mode: RecognitionMode = .fast,
                               maxSamples: Int = 30,
                               progress: (@Sendable (Double) -> Void)? = nil) async throws -> [PlateResult] {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        guard duration.isFinite, duration > 0 else { throw LicensePlateError.invalidInput("视频时长无效") }

        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw LicensePlateError.invalidInput("No video track.")
        }
        let transform = try await track.load(.preferredTransform)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw LicensePlateError.invalidInput("Cannot read video frames.") }
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? LicensePlateError.invalidInput("Cannot start video reader.") }
        defer { reader.cancelReading() }
        let sampleLimit = max(1, maxSamples)
        let step = duration / Double(sampleLimit)
        var collected: [PlateResult] = []
        var seen = 0
        var nextTime = 0.0
        while seen < sampleLimit, let sample = autoreleasepool(invoking: { output.copyNextSampleBuffer() }) {
            try Task.checkCancellation()
            let time = CMSampleBufferGetPresentationTimeStamp(sample).seconds
            guard time >= nextTime, let pixel = CMSampleBufferGetImageBuffer(sample) else { continue }
            let buffer = try autoreleasepool {
                try makePixelBuffer(from: CIImage(cvPixelBuffer: pixel).transformed(by: transform))
            }
            collected.append(contentsOf: try await recognizeFrame(buffer, mode: mode).plates)
            seen += 1
            nextTime += step
            progress?(min(1, time / duration))
        }
        if reader.status == .failed { throw reader.error ?? LicensePlateError.invalidInput("Video decoding failed.") }
        progress?(1)
        return PlatePostProcessor.deduplicate(collected)
    }
}

/// Preserves initialization failures so the UI can display the actual cause.
private final class CoreMLLicensePlateDetectorFailing: LicensePlateDetecting {
    private let error: Error
    init(error: Error) { self.error = error }
    func detect(in image: CGImage) async throws -> [PlateDetection] {
        throw error
    }
}
