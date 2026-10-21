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

/// One sampled frame of a video, kept so a player overlay can show the plates that were
/// found at *that* moment (the aggregate result list spans many frames and therefore
/// cannot be drawn onto a single frame's coordinate space).
public struct PlateVideoSample: Sendable, Identifiable {
    public let id: Int
    /// Presentation timestamp of the sampled frame, in seconds.
    public let time: Double
    public let imageSize: CGSize
    public let detections: [PlateDetection]
    public let plates: [PlateResult]

    public init(id: Int, time: Double, imageSize: CGSize,
                detections: [PlateDetection], plates: [PlateResult]) {
        self.id = id
        self.time = time
        self.imageSize = imageSize
        self.detections = detections
        self.plates = plates
    }
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
    /// The rectified plate image actually fed to the OCR engine (perspective-corrected
    /// from the 4 corner keypoints). `nil` when the detector fell back to a box crop.
    public let correctedPlateImage: CGImage?
    /// The plate outline (normalized, top-left origin) taken from the detection's pose
    /// keypoints. `nil` for sub-boxes carved out of a multi-plate crop, where the corners
    /// of the shared region would be wrong.
    public let corners: [CGPoint]?

    /// - Note: Declared explicitly (instead of relying on the synthesized memberwise
    ///   initializer) because Swift does not synthesize a parameter for an immutable
    ///   (`let`) property that carries a default value — `correctedPlateImage` would
    ///   otherwise be unusable by callers.
    public init(text: String,
                confidence: Float,
                boundingBox: CGRect,
                imageSize: CGSize,
                candidates: [PlateCandidate],
                correctedPlateImage: CGImage? = nil,
                corners: [CGPoint]? = nil) {
        self.text = text
        self.confidence = confidence
        self.boundingBox = boundingBox
        self.imageSize = imageSize
        self.candidates = candidates
        self.correctedPlateImage = correctedPlateImage
        self.corners = corners
    }
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
    private var recognizer: PlateTextRecognizing
    private let postProcessor: PlatePostProcessor
    private let corrector: PerspectivePlateCorrector
    private let recognitionLanguages: [Locale.Language]
    private let context = CIContext(options: [.cacheIntermediates: false])

    /// The OCR engine currently in use. Setting it rebuilds the underlying recognizer,
    /// so the App can let the user switch between Apple Vision and PaddleOCR at runtime.
    public var ocrEngine: OCREngine {
        didSet {
            guard ocrEngine != oldValue else { return }
            recognizer = ocrEngine.makeRecognizer(languages: recognitionLanguages)
        }
    }

    /// Channel order fed to the ONNX detector (RGB vs BGR). Defaults to RGB because that
    /// is what Ultralytics trains on and what the PC reference implementation feeds
    /// (`Scripts/pose_pipeline_reference.py`: `Image.open(...).convert("RGB")` → `/255`).
    /// Exposed so the Demo app can flip it at runtime if a differently trained export
    /// needs BGR.
    public var useRGB: Bool = true {
        didSet {
            #if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc)
            if let det = detector as? ONNXPoseLicensePlateDetector { det.useRGB = useRGB }
            #endif
        }
    }

    /// Human-readable description of the active detector (on-device diagnostics).
    public var detectorInfo: String { String(describing: type(of: detector)) }

    /// Opt-in strengthening of the validation / overlay stage. See
    /// `RecognitionEnhancements`. Toggling it is cheap: the value is read on every
    /// recognition pass, so nothing has to be rebuilt or reloaded.
    public var enhancements: RecognitionEnhancements = .off

    /// The default pipeline uses the original plate model converted to Core ML and Apple OCR.
    public convenience init() {
        let detector: LicensePlateDetecting
        do { detector = try CoreMLLicensePlateDetector() }
        catch { detector = CoreMLLicensePlateDetectorFailing(error: error) }
        self.init(detector: detector,
                  recognizer: OCREngine.appleVision.makeRecognizer(),
                  recognitionLanguages: [Locale.Language(identifier: "de-DE")])
    }

    public init(detector: LicensePlateDetecting,
                recognizer: PlateTextRecognizing,
                recognitionLanguages: [Locale.Language] = [Locale.Language(identifier: "de-DE")],
                ocrEngine: OCREngine = .appleVision,
                postProcessor: PlatePostProcessor = PlatePostProcessor(),
                enhancements: RecognitionEnhancements = .off) {
        self.detector = detector
        self.recognizer = recognizer
        self.recognitionLanguages = recognitionLanguages
        self.ocrEngine = ocrEngine
        self.postProcessor = postProcessor
        self.corrector = PerspectivePlateCorrector()
        self.enhancements = enhancements
    }

    /// Convenience initializer for the common setup: Core ML YOLO detector + Vision OCR,
    /// German (`de-DE`) recognition.
    /// - Parameters:
    ///   - modelName: Resource name for an explicitly supplied Core ML YOLO model.
    ///   - config: YOLO output layout (default `.germanPlate`).
    ///   - recognitionLanguages: OCR languages (default `[de-DE]`).
    ///   - enhancements: Opt-in strengthening (default `.off` = legacy behaviour).
    public convenience init(modelName: String,
                            config: YOLOConfig = .germanPlate,
                            recognitionLanguages: [Locale.Language] = [Locale.Language(identifier: "de-DE")],
                            enhancements: RecognitionEnhancements = .off) {
        let detector: LicensePlateDetecting
        do { detector = try CoreMLLicensePlateDetector(modelName: modelName, config: config) }
        catch { detector = CoreMLLicensePlateDetectorFailing(error: error) }
        self.init(detector: detector,
                  recognizer: PlateTextRecognizer(languages: recognitionLanguages),
                  recognitionLanguages: recognitionLanguages,
                  ocrEngine: .appleVision,
                  postProcessor: PlatePostProcessor(),
                  enhancements: enhancements)
    }

    /// Convenience initializer that wires the trained German-plate YOLOv8-pose model
    /// (running through ONNX Runtime) with the selected OCR engine.
    /// Requires the ONNX Runtime framework + `De-plate-pose-500.onnx` in the bundle.
    #if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc)
    public convenience init(poseModelName: String = "De-plate-pose-500",
                            config: YOLOConfig = .germanPlatePose,
                            engine: OCREngine = .appleVision,
                            recognitionLanguages: [Locale.Language] = [Locale.Language(identifier: "de-DE")],
                            enhancements: RecognitionEnhancements = .off) {
        let detector: LicensePlateDetecting
        do { detector = try ONNXPoseLicensePlateDetector(modelName: poseModelName, config: config) }
        catch { detector = CoreMLLicensePlateDetectorFailing(error: error) }
        self.init(detector: detector,
                  recognizer: engine.makeRecognizer(languages: recognitionLanguages),
                  recognitionLanguages: recognitionLanguages,
                  ocrEngine: engine,
                  postProcessor: PlatePostProcessor(),
                  enhancements: enhancements)
    }
    #endif

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

        // One frame can hold several plates (car park, traffic scene, forecourt). Each
        // plate is an independent rectify + OCR job, so they run concurrently — this used
        // to be a serial `for` loop, which made latency scale linearly with plate count.
        // Collaborators are captured as locals so each task only touches value types /
        // Sendable existentials.
        let corrector = self.corrector
        let recognizer = self.recognizer
        let postProcessor = self.postProcessor
        // Snapshot the opt-in switches: the App may flip them while a pass is in flight.
        let enhancements = self.enhancements
        // `CGImage` is immutable and safe to read from several tasks at once; the wrapper
        // states that explicitly so the task group does not need to copy the frame.
        let source = SendableImage(image: image)

        let perDetection: [[PlateResult]] = try await withThrowingTaskGroup(
            of: (offset: Int, plates: [PlateResult]).self
        ) { group in
            for (offset, detection) in detections.enumerated() {
                group.addTask {
                    try Task.checkCancellation()
                    // Rectify using this detection's own 4 corner keypoints (preferred),
                    // else fall back to a padded bounding-box crop.
                    let corrected = corrector.correctedPlate(from: source.image, detection: detection)
                    let ocrSource = corrected ?? source.image.cropRegion(detection.rect, padding: 0.05)
                    // Reduce interference (screws, borders, reflections) before OCR.
                    let enhanced = ocrSource.enhancedForOCR()
                    let observations = try await recognizer.recognize(in: enhanced, mode: mode)
                    // One crop can contain several plates (merged detections / padding
                    // pulling in a neighbour), so `process` returns every distinct plate.
                    let plates = postProcessor.process(observations: observations,
                                                       region: detection.rect,
                                                       imageSize: imageSize,
                                                       correctedImage: corrected,
                                                       corners: detection.keypoints,
                                                       enhancements: enhancements)
                    return (offset: offset, plates: plates)
                }
            }
            // Restore detection order: NMS order is by confidence, and the overlay numbers
            // plates by that order, so results must not be reaggregated as they complete.
            var buckets = Array(repeating: [PlateResult](), count: detections.count)
            for try await item in group { buckets[item.offset] = item.plates }
            return buckets
        }

        // Two overlapping detections can cover the same physical plate (IoU below the NMS
        // threshold), which used to yield the same plate twice in the UI. Collapse those.
        let results = PlatePostProcessor.deduplicateOverlapping(perDetection.flatMap { $0 })
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
        let samples = try await recognizeVideoSamples(at: url, mode: mode,
                                                      maxSamples: maxSamples, progress: progress)
        return PlatePostProcessor.deduplicate(samples.flatMap(\.plates))
    }

    /// Same sampling pass as `recognizeVideo`, but keeps every sampled frame's detections,
    /// plates and size. A player overlay needs this: the aggregate result list spans many
    /// frames, so drawing it onto one frame would place boxes in the wrong coordinate space.
    public func recognizeVideoSamples(at url: URL, mode: RecognitionMode = .fast,
                                      maxSamples: Int = 30,
                                      progress: (@Sendable (Double) -> Void)? = nil) async throws -> [PlateVideoSample] {
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
        var samples: [PlateVideoSample] = []
        var seen = 0
        var nextTime = 0.0
        while seen < sampleLimit, let sample = autoreleasepool(invoking: { output.copyNextSampleBuffer() }) {
            try Task.checkCancellation()
            let time = CMSampleBufferGetPresentationTimeStamp(sample).seconds
            guard time >= nextTime, let pixel = CMSampleBufferGetImageBuffer(sample) else { continue }
            let buffer = try autoreleasepool {
                try makePixelBuffer(from: CIImage(cvPixelBuffer: pixel).transformed(by: transform))
            }
            let frame = try await recognizeFrame(buffer, mode: mode)
            samples.append(PlateVideoSample(id: seen, time: time, imageSize: frame.imageSize,
                                            detections: frame.detections, plates: frame.plates))
            seen += 1
            nextTime += step
            progress?(min(1, time / duration))
        }
        if reader.status == .failed { throw reader.error ?? LicensePlateError.invalidInput("Video decoding failed.") }
        progress?(1)
        return samples
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

/// Carries one immutable frame into the concurrent per-plate tasks. `CGImage` is
/// immutable, so reading it from several tasks at once is safe; the compiler just
/// cannot prove it.
private struct SendableImage: @unchecked Sendable {
    let image: CGImage
}
