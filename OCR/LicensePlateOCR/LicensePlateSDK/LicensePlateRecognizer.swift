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
import os

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
                        mode: RecognitionMode, tiled: Bool,
                        onDetections: (@Sendable ([PlateDetection], CGSize) -> Void)?) async throws -> PlateFrameResult
    /// Recognizes plates in a still image.
    func recognize(in image: CGImage, mode: RecognitionMode) async throws -> [PlateResult]
    /// Recognizes plates across a video, sampling frames over time.
    func recognizeVideo(at url: URL, mode: RecognitionMode, maxSamples: Int,
                        progress: (@Sendable (Double) -> Void)?) async throws -> [PlateResult]
}

/// Default implementation wiring together the detector, recognizer and post-processor.
///
/// `LicensePlateRecognizing` refines `Sendable`, and a `Sendable` class may only have
/// immutable (`let`) stored properties. The runtime switches (OCR engine, channel order,
/// enhancements) are therefore grouped behind an `OSAllocatedUnfairLock` instead of
/// plain `var`s — the lock also makes swapping the recognizer mid-flight safe.
public final class LicensePlateRecognizer: LicensePlateRecognizing {

    /// The UI-driven mutable state, guarded by `runtimeState`.
    private struct RuntimeState: Sendable {
        var recognizer: PlateTextRecognizing
        var engine: OCREngine
        var useRGB: Bool
        /// Legacy letterbox-orientation flag (deprecated, unused by the Core ML pipeline).
        var flipY: Bool
        var enhancements: RecognitionEnhancements
    }

    private let detector: LicensePlateDetecting
    private let postProcessor: PlatePostProcessor
    private let corrector: PerspectivePlateCorrector
    private let recognitionLanguages: [Locale.Language]
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let runtimeState: OSAllocatedUnfairLock<RuntimeState>

    /// The OCR engine currently in use. Setting it rebuilds the underlying recognizer,
    /// so the App can let the user switch between Apple Vision and PaddleOCR at runtime.
    public var ocrEngine: OCREngine {
        get { runtimeState.withLock { $0.engine } }
        set {
            runtimeState.withLock { state in
                guard state.engine != newValue else { return }
                state.engine = newValue
                state.recognizer = newValue.makeRecognizer(languages: recognitionLanguages)
            }
        }
    }

    /// Deprecated no-op. Channel order is fixed by the official Core ML pipeline
    /// (Ultralytics exports train on RGB); kept only so the demo's settings sheet
    /// keeps compiling until those controls are removed.
    @available(*, deprecated, message: "The Core ML pipeline fixes the channel order.")
    public var useRGB: Bool {
        get { runtimeState.withLock { $0.useRGB } }
        set { runtimeState.withLock { $0.useRGB = newValue } }
    }

    /// Deprecated no-op. Letterboxing is performed by the official package
    /// (Vision `scaleFit`), there is no buffer orientation to choose.
    @available(*, deprecated, message: "The Core ML pipeline owns letterboxing.")
    public var flipLetterbox: Bool {
        get { runtimeState.withLock { $0.flipY } }
        set { runtimeState.withLock { $0.flipY = newValue } }
    }

    /// Human-readable description of the active detector (on-device diagnostics).
    public var detectorInfo: String { String(describing: type(of: detector)) }

    /// Confidence gate on the raw model output (see `LicensePlateDetecting`).
    /// Default 0.25, matching the PC reference's Ultralytics default.
    public var confidenceThreshold: Float {
        get { detector.confidenceThreshold }
        set { detector.confidenceThreshold = newValue }
    }

    /// NMS IoU (see `LicensePlateDetecting`). Default 0.7, matching the PC reference.
    public var nmsIoUThreshold: Float {
        get { detector.nmsIoUThreshold }
        set { detector.nmsIoUThreshold = newValue }
    }

    /// The exact 640x640 image fed to the model, when snapshotting is enabled. Compare it
    /// with the PC pipeline's input to tell "wrong tensor" from "wrong decode" apart.
    public var debugSnapshot: CGImage? { detector.debugSnapshot }

    /// Enables capturing `debugSnapshot`. Off by default: it costs one extra image copy.
    public var captureDebugSnapshot: Bool {
        get { detector.captureDebugSnapshot }
        set { detector.captureDebugSnapshot = newValue }
    }

    /// Opt-in strengthening of the validation / overlay stage. See
    /// `RecognitionEnhancements`. Toggling it is cheap: the value is read on every
    /// recognition pass, so nothing has to be rebuilt or reloaded.
    public var enhancements: RecognitionEnhancements {
        get { runtimeState.withLock { $0.enhancements } }
        set { runtimeState.withLock { $0.enhancements = newValue } }
    }

    /// The default pipeline: official Ultralytics YOLO Core ML pose detector + Apple OCR.
    public convenience init() {
        let detector: LicensePlateDetecting
        do { detector = try UltralyticsPosePlateDetector() }
        catch { detector = DetectorFailing(error: error) }
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
        self.recognitionLanguages = recognitionLanguages
        self.postProcessor = postProcessor
        self.corrector = PerspectivePlateCorrector()
        self.runtimeState = OSAllocatedUnfairLock(initialState: RuntimeState(
            recognizer: recognizer, engine: ocrEngine, useRGB: true, flipY: true,
            enhancements: enhancements))
    }

    /// Convenience initializer for an explicitly named Core ML pose model.
    /// - Parameters:
    ///   - modelName: Resource name of the exported `.mlpackage` (default `De-plate-pose-500`).
    ///   - recognitionLanguages: OCR languages (default `[de-DE]`).
    ///   - enhancements: Opt-in strengthening (default `.off` = legacy behaviour).
    public convenience init(modelName: String = "De-plate-pose-500",
                            recognitionLanguages: [Locale.Language] = [Locale.Language(identifier: "de-DE")],
                            enhancements: RecognitionEnhancements = .off) {
        let detector: LicensePlateDetecting
        do { detector = try UltralyticsPosePlateDetector(modelName: modelName) }
        catch { detector = DetectorFailing(error: error) }
        self.init(detector: detector,
                  recognizer: PlateTextRecognizer(languages: recognitionLanguages),
                  recognitionLanguages: recognitionLanguages,
                  ocrEngine: .appleVision,
                  postProcessor: PlatePostProcessor(),
                  enhancements: enhancements)
    }

    /// Wires the German-plate pose model (`De-plate-pose-500.mlpackage`) with the
    /// selected OCR engine. Detection runs through the official Ultralytics YOLO
    /// package (Core ML), so preprocessing, decoding and NMS match the official app.
    public convenience init(poseModelName: String = "De-plate-pose-500",
                            engine: OCREngine = .appleVision,
                            recognitionLanguages: [Locale.Language] = [Locale.Language(identifier: "de-DE")],
                            enhancements: RecognitionEnhancements = .off) {
        let detector: LicensePlateDetecting
        do { detector = try UltralyticsPosePlateDetector(modelName: poseModelName) }
        catch { detector = DetectorFailing(error: error) }
        self.init(detector: detector,
                  recognizer: engine.makeRecognizer(languages: recognitionLanguages),
                  recognitionLanguages: recognitionLanguages,
                  ocrEngine: engine,
                  postProcessor: PlatePostProcessor(),
                  enhancements: enhancements)
    }

    public func recognize(in image: CGImage, mode: RecognitionMode = .fast) async throws -> [PlateResult] {
        // Straight to the CGImage core: wrapping the still in a CVPixelBuffer first would
        // add a render round-trip plus a colour-space conversion (see `recognizeImage`).
        try await recognizeImage(image, mode: mode, tiled: true).plates
    }

    /// Detection + OCR on an already-upright `CGImage`.
    ///
    /// Media paths (still photos, video frames) call this directly instead of going through
    /// `recognizeFrame`'s `CVPixelBuffer`. The buffer route is only needed for camera
    /// capture, and for media it costs an extra render **and** an extra conversion
    /// (`CIImage` renders through a wide-gamut working space, so a BGRA buffer round-trip
    /// can shift colours relative to the PC pipeline, which feeds the model the decoded
    /// sRGB bytes unchanged). Keeping the pixel out of the loop removes one of the last
    /// real differences from the PC reference.
    public func recognizeImage(_ image: CGImage,
                               mode: RecognitionMode = .fast,
                               tiled: Bool = false,
                               onDetections: (@Sendable ([PlateDetection], CGSize) -> Void)? = nil) async throws -> PlateFrameResult {
        try Task.checkCancellation()
        // Cap the working resolution the same way `recognizeFrame` caps buffers: a
        // 48 MP photo gains nothing for a 640-input model and only burns memory/time.
        let image = clampedToWorkingSize(image)
        let imageSize = CGSize(width: image.width, height: image.height)
        // Still photos: run the whole-image pass FIRST — that is exactly what the PC
        // reference does (one letterbox of the entire picture), and the plate scale it
        // produces is the scale the model was trained on. The tiled pass is then used only
        // to add the plates that are too small to survive a full-frame letterbox, and the
        // two sets are merged with NMS. Previously only the tiled pass ran, which crops
        // 1024px tiles and therefore shows the model plates at a very different scale —
        // the "right on the PC, wrong on the phone" case.
        let primary = try await detector.detect(in: image)
        let merged: [PlateDetection]
        if tiled {
            let tiledDetections = try await detector.detectTiled(in: image)
            merged = (primary + tiledDetections)
        } else {
            merged = primary
        }
        // Re-anchor each detection's box on its pose keypoints (see
        // `PlateDetection.alignedToKeypoints`): the model's box head and keypoint head can
        // disagree, and the keypoints visibly track the plate. One geometry for the
        // corrector, the OCR fallback crop, the tracker and every overlay.
        let aligned = merged.map(\.alignedToKeypoints)
        // Cross-pass NMS: the same plate seen by both passes must be reported once, while
        // genuinely distinct plates (low IoU) are all kept. The threshold comes from the
        // detector so it matches the in-model NMS — a fixed 0.5 here used to re-merge
        // plates the model had just decided to keep apart.
        let detections = PlateNMS.suppress(aligned, threshold: detector.nmsIoUThreshold)
        onDetections?(detections, imageSize)

        // One frame can hold several plates (car park, traffic scene, forecourt). Each
        // plate is an independent rectify + OCR job, so they run concurrently — this used
        // to be a serial `for` loop, which made latency scale linearly with plate count.
        // Collaborators are captured as locals so each task only touches value types /
        // Sendable existentials.
        let corrector = self.corrector
        // Snapshot the locked runtime state: each task only touches Sendable values, and
        // the App may flip the OCR engine / enhancements while a pass is in flight.
        let (recognizer, enhancements) = runtimeState.withLock { ($0.recognizer, $0.enhancements) }
        let postProcessor = self.postProcessor
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

    /// Converts an upright still image to an external BGRA frame, scaling oversized images.
    public func prepareFrame(from image: CGImage) throws -> CVPixelBuffer {
        try makePixelBuffer(from: CIImage(cgImage: image))
    }

    /// - Parameter tiled: run tiled detection (see `LicensePlateDetecting.detectTiled`).
    ///   Turn this on for **still images** whose plates may be small relative to the
    ///   frame (collages, parking lots, forecourt photos): the pose model's input is a
    ///   fixed 640x640, so a single pass letterboxes the whole picture and small plates
    ///   fall below the confidence threshold. Leave off for video/live frames, where
    ///   latency matters more than the last few tiny plates.
    public func recognizeFrame(_ pixelBuffer: CVPixelBuffer,
                               orientation: CGImagePropertyOrientation = .up,
                               mode: RecognitionMode = .fast,
                               tiled: Bool = false,
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
        // Delegate to the CGImage core: one geometry, one set of thresholds, and no
        // second render for media that never needed a pixel buffer in the first place.
        return try await recognizeImage(image, mode: mode, tiled: tiled, onDetections: onDetections)
    }

    /// Scales an oversized still/frame down to the working resolution the detector is
    /// dimensioned for (3840 x 2160 max). Only kicks in for very large photos — everything
    /// else is passed through untouched, so the bytes the model sees are the decoded image.
    private func clampedToWorkingSize(_ image: CGImage) -> CGImage {
        let w = image.width, h = image.height
        guard w > 0, h > 0, max(w, h) > 3840 || min(w, h) > 2160 else { return image }
        let scale = min(CGFloat(3840) / CGFloat(max(w, h)), CGFloat(2160) / CGFloat(min(w, h)))
        let ci = CIImage(cgImage: image).transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        return context.createCGImage(ci, from: ci.extent) ?? image
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
                               maxSamples: Int = 60,
                               progress: (@Sendable (Double) -> Void)? = nil) async throws -> [PlateResult] {
        let samples = try await recognizeVideoSamples(at: url, mode: mode,
                                                      maxSamples: maxSamples, progress: progress)
        return PlatePostProcessor.deduplicate(samples.flatMap(\.plates))
    }

    /// Same sampling pass as `recognizeVideo`, but keeps every sampled frame's detections,
    /// plates and size. A player overlay needs this: the aggregate result list spans many
    /// frames, so drawing it onto one frame would place boxes in the wrong coordinate space.
    public func recognizeVideoSamples(at url: URL, mode: RecognitionMode = .fast,
                                      maxSamples: Int = 60,
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
            // Apply the track's preferred transform so a sideways phone clip is upright,
            // then hand the CGImage straight to the core — the old `makePixelBuffer`
            // round-trip added a render plus a colour-space conversion per frame.
            let cgImage: CGImage? = autoreleasepool {
                let ci = CIImage(cvPixelBuffer: pixel).transformed(by: transform)
                return context.createCGImage(ci, from: ci.extent)
            }
            guard let cgImage else {
                throw LicensePlateError.invalidInput("Could not render the input frame.")
            }
            let frame = try await recognizeImage(cgImage, mode: mode)
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
private final class DetectorFailing: LicensePlateDetecting {
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
