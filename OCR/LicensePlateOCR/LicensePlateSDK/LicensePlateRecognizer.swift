/*
 License Plate OCR SDK
 
 Abstract:
 The public entry point of the SDK. Locates German license plates with a YOLO pose model,
 corrects each plate to a rectified crop, reads it with PaddleOCR PP-OCRv5 (via the linked
 ONNX Runtime), and cleans the result with German-plate post-processing. Works on a single
 image, a video, or a live camera feed.
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
    /// Format-validated plate text. Recognition results returned by the SDK carry the
    /// flat form (letters and digits only, e.g. `BAB1234`): the canonical `CITY-NUMBER`
    /// form is used internally for validation and flattened at the SDK exit in
    /// `recognizeImage`. Callers that construct a `PlateResult` directly keep whatever
    /// string they pass in.
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
    /// [TEMP-OCR-INPUT-DIAG] Exact image fed to the OCR engine AFTER cleaning +
    /// enhancement. Shown next to `correctedPlateImage` so the eye can compare the
    /// rectified plate with what the model actually reads. Remove with the diagnostic UI.
    public let ocrInputImage: CGImage?
    /// The plate outline (normalized, top-left origin) taken from the detection's pose
    /// keypoints. `nil` for sub-boxes carved out of a multi-plate crop, where the corners
    /// of the shared region would be wrong.
    public let corners: [CGPoint]?
    /// How many analysis frames agreed on this reading (1 = a single sighting).
    ///
    /// Set by `PlateVoteAggregator`, which is what turns a per-frame OCR result into a
    /// de-duplicated, majority-voted plate list. A still image or a single frame keeps
    /// the default of 1; video and live camera raise it to the number of agreeing frames.
    public let votes: Int

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
                ocrInputImage: CGImage? = nil, // TEMP-OCR-INPUT-DIAG
                corners: [CGPoint]? = nil,
                votes: Int = 1) {
        self.text = text
        self.confidence = confidence
        self.boundingBox = boundingBox
        self.imageSize = imageSize
        self.candidates = candidates
        self.correctedPlateImage = correctedPlateImage
        self.ocrInputImage = ocrInputImage // TEMP-OCR-INPUT-DIAG
        self.corners = corners
        self.votes = votes
    }
}

extension PlateResult {
    /// Same plate reading with a different vote count — used when the vote aggregator
    /// promotes a frame result into its majority-voted representative.
    func withVotes(_ votes: Int) -> PlateResult {
        PlateResult(text: text,
                    confidence: confidence,
                    boundingBox: boundingBox,
                    imageSize: imageSize,
                    candidates: candidates,
                    correctedPlateImage: correctedPlateImage,
                    ocrInputImage: ocrInputImage, // TEMP-OCR-INPUT-DIAG
                    corners: corners,
                    votes: votes)
    }
}

/// A single alternative reading of a plate.
public struct PlateCandidate: Sendable, Identifiable {
    public let id = UUID()
    public let text: String
    public let confidence: Float
}

/// The full recognition pipeline: detect -> crop -> recognize -> post-process.
/// The App → SDK boundary is CVPixelBuffer-only. Every public recognition entry
/// takes a buffer; CGImage / Data / URL never cross this line as an image payload.
public protocol LicensePlateRecognizing: Sendable {
    /// Universal buffer entry. Accepts caller-owned NV12 (full or video range) or
    /// BGRA frames without creating a capture session. This is the single funnel
    /// for camera, still and video frames alike.
    func recognizeFrame(_ pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation,
                        mode: RecognitionMode, tiled: Bool,
                        onDetections: (@Sendable ([PlateDetection], CGSize) -> Void)?) async throws -> PlateFrameResult
    /// Recognizes plates in a still image supplied as a CVPixelBuffer. Build the
    /// buffer with `makeInputBuffer(from:)` (NV12 by default) or your own producer.
    func recognize(in buffer: CVPixelBuffer, orientation: CGImagePropertyOrientation,
                   mode: RecognitionMode, tiled: Bool) async throws -> PlateFrameResult
    /// Video is a resource locator, not an image payload. Sampled frames are
    /// converted to CVPixelBuffers internally before entering the buffer funnel.
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
        /// The single text recognizer (PaddleOCR). Its graph is loaded lazily off
        /// the main thread (see `PaddleOCRTextRecognizer.EngineStore`); the wrapper
        /// itself is fixed for the recognizer's lifetime — no engine switching.
        var recognizer: PlateTextRecognizing
        var useRGB: Bool
        /// See `ONNXPoseLicensePlateDetector.flipY`: top-row-first letterbox buffer
        /// (`true` = the PC/OpenCV layout). Default `true`.
        var flipY: Bool
        var enhancements: RecognitionEnhancements
        /// Layout produced by `makeInputBuffer(from:)` for stills / video frames.
        /// NV12 by default (camera-native); BGRA is the opt-in alternative.
        var inputPixelFormat: PlateInputPixelFormat = .nv12
        /// OCR-level gate applied at the SDK exit. Detector confidence gates raw anchors;
        /// this one suppresses low-confidence alternative readings that would otherwise
        /// show up as separate plate rows in the UI.
        var minimumResultConfidence: Float = 0.20
    }

    private let detector: LicensePlateDetecting
    private let postProcessor: PlatePostProcessor
    private let corrector: PerspectivePlateCorrector
    private let plateImageCleaner = PlateImageCleaner()
    /// Shared process-wide context (Apple: do not create many CIContext instances).
    private let context = SDKGraphics.ciContext
    private let runtimeState: OSAllocatedUnfairLock<RuntimeState>

    /// Channel order fed to the ONNX detector (RGB vs BGR). Defaults to RGB because that
    /// is what Ultralytics trains on and what the PC reference implementation feeds
    /// (`Scripts/pose_pipeline_reference.py`: `Image.open(...).convert("RGB")` → `/255`).
    /// Exposed so the Demo app can flip it at runtime if a differently trained export
    /// needs BGR.
    public var useRGB: Bool {
        get { runtimeState.withLock { $0.useRGB } }
        set {
            runtimeState.withLock { $0.useRGB = newValue }
            #if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc) || canImport(onnxruntime_objc)
            if let det = detector as? ONNXPoseLicensePlateDetector { det.useRGB = newValue }
            #endif
        }
    }

    /// Whether the active detector actually honours `useRGB`. `false` for the LibreYOLO
    /// YOLO-NAS-pose export (its training pipeline fixes BGR) and for the legacy Core ML
    /// detector (Core ML normalizes the buffer itself, so the switch is inert). The App
    /// disables the RGB/BGR picker in that case rather than offering a no-op control.
    public var honorsInputChannelOrder: Bool { detector.honorsChannelOrder }

    #if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc) || canImport(onnxruntime_objc)
    /// Highest raw anchor score of the most recent detection pass, before the confidence
    /// threshold. Diagnostic only: on a device where the same image detects fine on the PC
    /// script, this one value splits "the model saw nothing plate-like in the input"
    /// (≈0.01-0.1 — wrong preprocessing/orientation on device) from "the model sees the
    /// plate but the decode/validation drops it" (≥0.25 — a Swift-side bug).
    public var detectorTopScore: Float? {
        (detector as? ONNXPoseLicensePlateDetector)?.lastTopScore
    }
    #endif

    /// Whether the detector's letterbox buffer is written top-row-first (see
    /// `ONNXPoseLicensePlateDetector.flipY`). Default `true`, i.e. the same layout the PC
    /// reference feeds the model. Flip it only for an A/B test on device: with the wrong
    /// value the boxes land vertically mirrored and OCR reads nothing.
    public var flipLetterbox: Bool {
        get { runtimeState.withLock { $0.flipY } }
        set {
            runtimeState.withLock { $0.flipY = newValue }
            #if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc) || canImport(onnxruntime_objc)
            if let det = detector as? ONNXPoseLicensePlateDetector { det.flipY = newValue }
            #endif
        }
    }

    /// Human-readable description of the active detector (on-device diagnostics).
    ///
    /// When detector init failed (e.g. the ONNX model never made it into the app bundle,
    /// or the ORT session could not be created) the recognizer is built around the
    /// `LicensePlateDetectorUnavailable` stub — the *reason* must not be swallowed:
    /// the stub carries the original error, so append it verbatim.
    public var detectorInfo: String {
        let name = String(describing: type(of: detector))
        if let failing = detector as? LicensePlateDetectorUnavailable {
            return "\(name)(\(failing.failureDescription))"
        }
        return name
    }

    /// Pixel layout produced when the SDK wraps a decoded still / video frame in a
    /// CVPixelBuffer. Defaults to `.nv12` (the camera's native layout); set to
    /// `.bgra` for producers that cannot emit NV12. This does not affect the camera
    /// path, which already delivers NV12 buffers directly.
    public var inputPixelFormat: PlateInputPixelFormat {
        get { runtimeState.withLock { $0.inputPixelFormat } }
        set { runtimeState.withLock { $0.inputPixelFormat = newValue } }
    }

    /// Build the CVPixelBuffer the SDK expects from an already-upright CGImage.
    /// Call this in the App for a still photo, then pass the result to
    /// `recognize(in:)`. Uses `inputPixelFormat` (NV12 by default).
    public func makeInputBuffer(from image: CGImage) throws -> CVPixelBuffer {
        try PlatePixelBufferFactory.makeBuffer(from: image, format: inputPixelFormat)
    }

    /// Confidence gate on the raw model output (see `LicensePlateDetecting`).
    /// Default 0.25, matching the PC reference's Ultralytics default.
    public var confidenceThreshold: Float {
        get { detector.confidenceThreshold }
        set { detector.confidenceThreshold = newValue }
    }

    /// Minimum OCR confidence for a reading to be returned as a plate. The detector's
    /// `confidenceThreshold` gates raw plate anchors; this gate removes the low-confidence
    /// tail that OCR emits as alternative readings (e.g. 10% / 4% / 1% rows for one plate).
    /// Default 0.20; set to 0 to keep every internally validated reading.
    public var minimumResultConfidence: Float {
        get { runtimeState.withLock { $0.minimumResultConfidence } }
        set { runtimeState.withLock { $0.minimumResultConfidence = min(1, max(0, newValue)) } }
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

    /// - Parameter recognizer: the text recognizer to use. Production callers pass
    ///   `PaddleOCR.makeRecognizer()`; tests inject a mock. There is no engine switch —
    ///   PaddleOCR is the SDK's sole OCR engine.
    public init(detector: LicensePlateDetecting,
                recognizer: PlateTextRecognizing,
                postProcessor: PlatePostProcessor = PlatePostProcessor(),
                enhancements: RecognitionEnhancements = .off) {
        self.detector = detector
        self.postProcessor = postProcessor
        self.corrector = PerspectivePlateCorrector()
        self.runtimeState = OSAllocatedUnfairLock(initialState: RuntimeState(
            recognizer: recognizer, useRGB: true, flipY: true,
            enhancements: enhancements))
    }

    /// Convenience initializer that wires the trained German-plate corner model
    /// (running through ONNX Runtime) with the PaddleOCR text recognizer.
    ///
    /// The SDK has exactly ONE detector: the LibreYOLO (YOLO-NAS-pose) export
    /// `De-plate-pose` (512 input, four outputs; older Ultralytics exports keep working
    /// via the same detector — the layout is auto-detected from the model's output
    /// names). If it cannot be initialized (model not in the bundle, ORT session
    /// failure, …) the recognizer is built around `LicensePlateDetectorUnavailable`,
    /// whose `detect` throws — the reason is readable from `detectorInfo` on the
    /// device UI instead of being swallowed.
    /// - Parameters:
    ///   - poseModelName: Resource name of the `.onnx` model (without extension).
    ///   - config: YOLO output layout (default `.germanPlatePoseLibre`).
    ///   - enhancements: Opt-in strengthening (default `.off` = legacy behaviour).
    ///   - poseThreads: Intra-op threads for the ONNX session (default 2). ORT's own default
    ///     is one thread per core, which keeps the performance cores saturated during a long
    ///     live session and is the main heat contributor; 0 restores the ORT default.
    #if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc) || canImport(onnxruntime_objc)
    public convenience init(poseModelName: String = "De-plate-pose",
                            config: YOLOConfig = .germanPlatePoseLibre,
                            enhancements: RecognitionEnhancements = .off,
                            poseThreads: Int32 = 2) {
        let detector: LicensePlateDetecting
        do {
            detector = try ONNXPoseLicensePlateDetector(modelName: poseModelName, config: config,
                                                       intraOpThreads: poseThreads)
        } catch {
            detector = LicensePlateDetectorUnavailable(error: error)
        }
        self.init(detector: detector,
                  recognizer: PaddleOCR.makeRecognizer(),
                  postProcessor: PlatePostProcessor(),
                  enhancements: enhancements)
    }
    #endif

    /// Public still-image entry: the caller supplies a CVPixelBuffer (NV12 by
    /// default via `makeInputBuffer(from:)`, or a caller-owned BGRA/NV12 buffer).
    /// The buffer is funnelled through `recognizeFrame`, the same core the camera
    /// and video paths use, so there is exactly one set of thresholds and geometry.
    public func recognize(in buffer: CVPixelBuffer,
                          orientation: CGImagePropertyOrientation = .up,
                          mode: RecognitionMode = .fast,
                          tiled: Bool = true) async throws -> PlateFrameResult {
        try await recognizeFrame(buffer, orientation: orientation, mode: mode,
                                 tiled: tiled, onDetections: nil)
    }

    /// Internal CGImage core. Detection + OCR on an already-upright `CGImage`.
    /// Not part of the public surface: callers hand in CVPixelBuffers, and every
    /// public entry converts its buffer into the CGImage this core needs exactly
    /// once (in `recognizeFrame`). Keeping media on the CGImage path *inside* the
    /// SDK avoids the colour shift of an extra wide-gamut buffer round-trip while
    /// still presenting a buffer-only boundary to the App.
    func recognizeImage(_ image: CGImage,
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
        // Tiling only adds information when the picture actually exceeds one tile
        // (`detectTiled`'s own 1024-px default): for a smaller image it degenerates to a
        // second identical full-frame `detect` pass, so skip it and save an inference.
        if tiled, max(image.width, image.height) > 1024 {
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
        let detections = YOLOOutputDecoder.suppress(aligned, threshold: detector.nmsIoUThreshold)
        onDetections?(detections, imageSize)

        // One frame can hold several plates (car park, traffic scene, forecourt). Each
        // plate is an independent rectify + OCR job, so they run concurrently — this used
        // to be a serial `for` loop, which made latency scale linearly with plate count.
        // Collaborators are captured as locals so each task only touches value types /
        // Sendable existentials.
        let corrector = self.corrector
        // Captured as locals (not `self.x`) because `group.addTask` takes an *escaping*
        // @Sendable closure: an implicit `self` capture of an immutable property is not
        // allowed there without a qualifier, and the qualifier would drag the whole
        // recognizer into the task. `PlateImageCleaner` is a `Sendable` value type.
        let cleaner = self.plateImageCleaner
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
                    var ocrSource = corrected ?? source.image.cropRegion(detection.rect, padding: 0.05)
                    // Adaptive pre-OCR cleaning (EU strip + seals) — masked only when the
                    // feature is actually detected; plates without them pass untouched.
                    if enhancements.effectivePlateImageCleaning {
                        ocrSource = cleaner.clean(ocrSource)
                    }
                    // Reduce interference (screws, borders, reflections) before OCR.
                    let enhanced = ocrSource.enhancedForOCR()
                    let observations = try await recognizer.recognize(in: enhanced, mode: mode)
                    // One crop can contain several plates (merged detections / padding
                    // pulling in a neighbour), so `process` returns every distinct plate.
                    let plates = postProcessor.process(observations: observations,
                                                       region: detection.rect,
                                                       imageSize: imageSize,
                                                       correctedImage: corrected,
                                                       ocrInputImage: enhanced, // TEMP-OCR-INPUT-DIAG
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

        // Drop the low-confidence OCR tail before any geometry grouping: a 1–10% reading
        // is an alternative hypothesis for the SAME physical plate, not a second plate.
        // Keep it configurable so a difficult night scene can still opt back in.
        let minimumConfidence = runtimeState.withLock { $0.minimumResultConfidence }
        let confident = perDetection.flatMap { $0 }.filter {
            $0.confidence.isFinite && $0.confidence >= minimumConfidence
        }
        // Overlapping detections (and OCR alternatives from a merged crop) can produce
        // several readings of one plate with different text. Collapse those to the best
        // reading; genuinely separate plates keep their own rows.
        let results = PlatePostProcessor.deduplicateOverlapping(confident)
        // SDK exit: strip separators from every returned reading. The canonical
        // `CITY-NUMBER` form is still used upstream (validator, vote aggregation,
        // lexicon ranking all rely on the hyphen), so only what crosses this boundary
        // becomes flat. Every public entry point funnels through here, so this single
        // map covers still images, camera frames and sampled video alike.
        let flat = results.map { Self.flattenForReturn($0) }
        return PlateFrameResult(detections: detections, plates: flat, imageSize: imageSize)
    }

    /// Removes every non-alphanumeric character from a result at the SDK boundary.
    /// Umlauts count as letters, so city codes such as `TÜ` survive intact.
    private static func flattenForReturn(_ result: PlateResult) -> PlateResult {
        func flat(_ text: String) -> String {
            String(text.filter { $0.isLetter || $0.isNumber })
        }
        return PlateResult(text: flat(result.text),
                           confidence: result.confidence,
                           boundingBox: result.boundingBox,
                           imageSize: result.imageSize,
                           candidates: result.candidates.map {
                               PlateCandidate(text: flat($0.text), confidence: $0.confidence)
                           },
                           correctedPlateImage: result.correctedPlateImage,
                           ocrInputImage: result.ocrInputImage, // TEMP-OCR-INPUT-DIAG
                           corners: result.corners,
                           votes: result.votes)
    }

    /// Converts an upright still image into a CVPixelBuffer in the SDK's configured
    /// input layout (NV12 by default). Provided as a convenience for callers that
    /// hold a CGImage and want to enter the buffer-only API.
    public func prepareFrame(from image: CGImage) throws -> CVPixelBuffer {
        try makeInputBuffer(from: image)
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
        guard width > 0, height > 0,
              [kCVPixelFormatType_32BGRA, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
               kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange].contains(format) else {
            throw LicensePlateError.invalidInput("Expected BGRA or NV12, and a non-empty frame.")
        }
        try Task.checkCancellation()
        var ci = CIImage(cvPixelBuffer: pixelBuffer).oriented(orientation)
        // Same working-size cap as the CGImage core (3840 x 2160): an oversized frame is
        // scaled down instead of rejected, so both entry points behave identically.
        let w = ci.extent.width, h = ci.extent.height
        if max(w, h) > 3840 || min(w, h) > 2160 {
            let scale = min(CGFloat(3840) / max(w, h), CGFloat(2160) / min(w, h))
            ci = ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }
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

    public func recognizeVideo(at url: URL, mode: RecognitionMode = .fast,
                               maxSamples: Int = 60,
                               progress: (@Sendable (Double) -> Void)? = nil) async throws -> [PlateResult] {
        let samples = try await recognizeVideoSamples(at: url, mode: mode,
                                                      maxSamples: maxSamples, progress: progress)
        return PlateVoteAggregator.consolidate(samples.map(\.plates))
    }

    /// Same sampling pass as `recognizeVideo`, but keeps every sampled frame's detections,
    /// plates and size. A player overlay needs this: the aggregate result list spans many
    /// frames, so drawing it onto one frame would place boxes in the wrong coordinate space.
    public func recognizeVideoSamples(at url: URL, mode: RecognitionMode = .fast,
                                      maxSamples: Int = 60,
                                      progress: (@Sendable (Double) -> Void)? = nil) async throws -> [PlateVideoSample] {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        guard duration.isFinite, duration > 0 else { throw LicensePlateError.invalidInput("invalid video duration") }

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
            // wrap it as a CVPixelBuffer (NV12 by default), then enter the SAME buffer
            // funnel as camera and stills. Every public recognition path now crosses
            // the boundary as a CVPixelBuffer — CGImage is an internal detail only.
            let cgImage: CGImage? = autoreleasepool {
                let ci = CIImage(cvPixelBuffer: pixel).transformed(by: transform)
                return context.createCGImage(ci, from: ci.extent)
            }
            guard let cgImage else {
                throw LicensePlateError.invalidInput("Could not render the input frame.")
            }
            let frameBuffer = try makeInputBuffer(from: cgImage)
            let frame = try await recognizeFrame(frameBuffer, orientation: .up, mode: mode)
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

/// The SDK ships exactly one detector: the LibreYOLO YOLO-NAS-pose ONNX export. When it
/// cannot be initialized (model missing from the bundle, ORT session failure, no
/// onnxruntime linked) this stub keeps the recognizer constructible and surfaces the
/// cause through `detectorInfo` instead of crashing or failing silently.
public final class LicensePlateDetectorUnavailable: LicensePlateDetecting, @unchecked Sendable {
    private let error: Error
    public init(error: Error) { self.error = error }
    /// Human-readable cause for the UI triage line (see `detectorInfo`).
    /// `localizedDescription` (not `String(describing:)`) so `LicensePlateError`'s
    /// localized `errorDescription` — or an ORT NSError's message — is what shows up.
    public var failureDescription: String { error.localizedDescription }
    public func detect(in image: CGImage) async throws -> [PlateDetection] {
        throw error
    }
}

/// Carries one immutable frame into the concurrent per-plate tasks. `CGImage` is
/// immutable, so reading it from several tasks at once is safe; the compiler just
/// cannot prove it.
private struct SendableImage: @unchecked Sendable {
    let image: CGImage
}
