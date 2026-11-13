@preconcurrency import AVFoundation
import SwiftUI
import CoreImage
import ImageIO
import os

/// Retains the buffer until asynchronous recognition finishes.
private struct CameraFrame: @unchecked Sendable { let buffer: CVPixelBuffer }

/// Drops frames while recognition is busy instead of creating a backlog, and gates the
/// whole capture callback.
///
/// `enabled` exists so a camera that is idle (tab just opened), paused, or suspended can
/// reject a frame **on the capture queue**, before any `Task { @MainActor }` hop. The
/// previous code hopped to the main actor first and failed a `guard` inside — ~5 pointless
/// main-thread wake-ups per second for as long as the camera tab was on screen, even with
/// detection switched off.
///
/// `OSAllocatedUnfairLock` is Apple's recommended replacement for a bare `NSLock` in Swift
/// concurrency code: it is `Sendable`, and `withLock` cannot leave the lock held across an
/// `await` (which is exactly the bug a manual lock/unlock pair invites here). Also note
/// `os_unfair_lock` must never be used from Swift directly — `OSAllocatedUnfairLock` is the
/// supported wrapper.
private final class FrameGate: @unchecked Sendable {
    private struct State {
        var busy = false
        var lastStart = Date.distantPast
        var enabled = false
    }
    private let state = OSAllocatedUnfairLock(initialState: State())

    /// Cheap read used on the capture queue before touching the buffer.
    var isEnabled: Bool { state.withLock { $0.enabled } }

    /// Mirrors `active && liveEnabled && !paused` onto the capture queue.
    func setEnabled(_ enabled: Bool) {
        state.withLock { $0.enabled = enabled }
    }

    /// Returns true when this frame should be analysed (at most one in flight, ≥0.2 s apart).
    func begin() -> Bool {
        state.withLock { s in
            guard s.enabled, !s.busy,
                  Date().timeIntervalSince(s.lastStart) >= 0.2 else { return false }
            s.busy = true
            s.lastStart = Date()
            return true
        }
    }

    func finish() {
        state.withLock { $0.busy = false }
    }
}

struct CameraRecognitionLogEntry: Identifiable, Equatable {
    /// Stable across vote updates (the aggregator's entry id), so a row is refreshed in
    /// place instead of being torn down and re-created on every analysis cycle.
    let id: UUID
    let date: Date
    let result: PlateResult
    /// Frames that agreed on `result.text`, and all frames that saw this plate.
    let votes: Int
    let totalVotes: Int

    /// Equality is deliberately limited to what the row actually renders — identity,
    /// timestamp, text, counts and the corrected-image instance. `PlateResult` carries a
    /// fresh `UUID` per instance and an array of candidates, so a structural compare would
    /// report a change on *every* analysis cycle; this lets the controller skip the
    /// published assignment (and the whole list's re-render) when nothing visible moved.
    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.id == rhs.id
            && lhs.date == rhs.date
            && lhs.votes == rhs.votes
            && lhs.totalVotes == rhs.totalVotes
            && lhs.result.text == rhs.result.text
            && lhs.result.confidence == rhs.result.confidence
            && lhs.result.correctedPlateImage === rhs.result.correctedPlateImage
    }
}

@MainActor @Observable
final class CameraController: NSObject {
    nonisolated let session = AVCaptureSession()
    private nonisolated let sessionQueue = DispatchQueue(label: "licenseplate.camera.session")
    private nonisolated let frameGate = FrameGate()
    private let recognizer: LicensePlateRecognizing
    private var didSetup = false
    @ObservationIgnored private var configurationTask: Task<Bool, Never>?
    private var active = false
    private var generation = 0
    private var analysisIdentifier = 0
    private var lastPrinted: [String: Date] = [:]
    var isAuthorized = false
    var setupFailed = false
    var liveResults: [PlateResult] = []
    var liveDetections: [PlateDetection] = []
    var recognitionHistory: [CameraRecognitionLogEntry] = []
    var isAnalyzing = false
    var elapsedMilliseconds = 0
    var errorMessage: String?
    /// Live triage values (see the status pill): which detector implementation actually
    /// runs, and the model's best raw anchor score before the confidence threshold. On a
    /// device where the PC script detects the same scene, a Core ML name means the ONNX
    /// path never compiled in / failed to init; a near-zero top score means the model
    /// never saw a plate-like input (preprocessing/orientation), not a threshold problem.
    var detectorName: String = ""
    var lastTopScore: Float?
    /// Analysis cycles started since the app launched. Diagnostic only: together with the
    /// latency pill it separates the two ways the feed can look "dead" — a counter that
    /// stays at 0 means the capture pipeline never delivered a frame the gate accepted
    /// (session/permission/source problem), while a rising counter with 0 ms latency means
    /// every analysis threw before finishing (detector init/run failure).
    var frameCount = 0
    var mode: RecognitionMode = .fast
    var liveEnabled = true {
        didSet {
            generation += 1
            if !liveEnabled {
                // Close the voting session first: this lists a confident one-frame sighting
                // and drops every live track, so the next session cannot revive a plate the
                // user saw half a minute ago.
                publishHistory(voteAggregator.finish())
                liveResults = []
                liveDetections = []
                isAnalyzing = false
                trackedPlates.removeAll()
                foldedFrameIdentifier = -1
                previewView?.clearOverlay()
            }
            refreshAnalysisGate()
        }
    }
    /// When true, analysis is halted (frozen) but the camera keeps showing the last frame,
    /// and results + overlay stay on screen. Toggled by the pause button — distinct from
    /// `liveEnabled`, which fully stops the live feature and clears everything.
    ///
    /// `isAnalyzing` is cleared here rather than in the view: it drives the status pill and
    /// the scan beam, and a paused pipeline must not keep claiming "实时检测中" just because
    /// the button that flipped the flag happened to also reset it (any other caller — a
    /// scene-phase handler, a future auto-pause — would have left it stuck true).
    var paused = false {
        didSet {
            guard paused != oldValue else { return }
            if paused { isAnalyzing = false }
            refreshAnalysisGate()
        }
    }
    weak var previewView: PreviewView?
    /// Diagnostic: echoes the exact frame the SDK analyzed (as-is, rotation included)
    /// into a corner thumbnail, with the same normalized boxes drawn on it. If the box
    /// fits the plate on the echo but not on the preview → preview mapping bug; if the
    /// echo itself is rotated → the output connection's rotation did not apply.
    var debugEchoEnabled = false {
        didSet {
            if !debugEchoEnabled { previewView?.showDebugFrame(image: nil, detections: []) }
        }
    }

    /// Unified-logging handles. `print` writes to stdout, which is slow, has no levels and
    /// is invisible to the Console app; `Logger` + `OSSignposter` is what Apple's logging
    /// guidance recommends, and the signpost interval shows up directly in the Instruments
    /// Points of Interest track — the only reliable way to measure a per-frame change.
    private nonisolated static let logger = Logger(subsystem: "com.itlogica.PlateOCR",
                                                   category: "recognition")
    private nonisolated static let signposter = OSSignposter(subsystem: "com.itlogica.PlateOCR",
                                                             category: "recognition")

    /// Tells the capture queue whether frames are worth analysing at all.
    /// Called from every state change that can start or stop the live pipeline.
    private func refreshAnalysisGate() {
        frameGate.setEnabled(active && liveEnabled && !paused)
    }

    // MARK: - Temporal detection stability
    //
    // The pose model occasionally emits a one-frame phantom box (e.g. a quad regressed
    // onto the bumper *below* the real plate, or onto a plate-like reflection). Showing
    // every raw detection makes those phantoms flash on the live feed. Instead, boxes are
    // tracked across analysis cycles: a box is only drawn once it has been seen in
    // `confirmationHits` consecutive cycles at the same place (IoU association), its
    // geometry is exponentially smoothed, and a confirmed box survives a single missed
    // cycle for `trackPersistence` seconds so stable boxes don't flicker.

    /// One plate box tracked across consecutive analysis cycles.
    private struct TrackedPlate {
        var rect: CGRect
        var quad: [CGPoint]
        var confidence: Float
        var hits: Int
        var lastSeen: Date
    }

    @ObservationIgnored private var trackedPlates: [TrackedPlate] = []
    /// Identifier of the analysis frame whose detections were already folded into the
    /// tracker by the intermediate detection callback (so the final publish of the same
    /// frame reuses the tracks instead of double-counting `hits`).
    @ObservationIgnored private var foldedFrameIdentifier = -1

    /// Cross-frame voting for the history list (see `PlateVoteAggregator`). Kept separate
    /// from `trackedPlates`, which only stabilises the *overlay geometry*: this one decides
    /// what the list shows, and runs on the results rather than on raw detections.
    /// `@ObservationIgnored` because it is written every analysis cycle and must not make
    /// the whole view tree re-evaluate — `recognitionHistory` is the published output.
    @ObservationIgnored private var voteAggregator = PlateVoteAggregator()

    /// Cycles a box must be seen consecutively before it is drawn.
    private static let confirmationHits = 2
    /// A confirmed track stays drawn (frozen) this long without a matching detection.
    private static let trackPersistence: TimeInterval = 0.45
    /// Minimum IoU to associate an incoming detection with an existing track.
    private static let associationIoU: CGFloat = 0.3
    /// Weight of the *new* observation when smoothing track geometry (0...1).
    private static let smoothing: CGFloat = 0.5

    private nonisolated static func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        guard !inter.isEmpty else { return 0 }
        let interArea = inter.width * inter.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        return unionArea > 0 ? interArea / unionArea : 0
    }

    private nonisolated static func lerp(_ a: CGPoint, _ b: CGPoint, _ t: CGFloat) -> CGPoint {
        CGPoint(x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t)
    }

    private nonisolated static func lerp(_ a: CGRect, _ b: CGRect, _ t: CGFloat) -> CGRect {
        CGRect(x: a.minX + (b.minX - a.minX) * t,
               y: a.minY + (b.minY - a.minY) * t,
               width: a.width + (b.width - a.width) * t,
               height: a.height + (b.height - a.height) * t)
    }

    /// Folds the current frame's raw detections into the tracker and returns the
    /// stabilized set to draw: confirmed tracks (smoothed, `hits >= confirmationHits`),
    /// including confirmed tracks that missed this cycle but are still fresh.
    private func stabilizeDetections(_ incoming: [PlateDetection]) -> [PlateDetection] {
        let now = Date()
        trackedPlates.removeAll { now.timeIntervalSince($0.lastSeen) > Self.trackPersistence }

        var matchedIndices = Set<Int>()
        var nextTracks: [TrackedPlate] = []
        for detection in incoming {
            var bestIndex = -1
            var bestIoU = Self.associationIoU
            for (index, track) in trackedPlates.enumerated() where !matchedIndices.contains(index) {
                let score = Self.iou(detection.rect, track.rect)
                if score > bestIoU { bestIoU = score; bestIndex = index }
            }
            let incomingQuad = detection.plateQuad
            if bestIndex >= 0 {
                matchedIndices.insert(bestIndex)
                var track = trackedPlates[bestIndex]
                let t = Self.smoothing
                track.rect = Self.lerp(track.rect, detection.rect, t)
                // Pose corners have a fixed order, so point-wise smoothing is safe.
                if track.quad.count == incomingQuad.count {
                    track.quad = zip(track.quad, incomingQuad).map { Self.lerp($0, $1, t) }
                } else {
                    track.quad = incomingQuad
                }
                track.confidence = max(track.confidence, detection.confidence)
                track.hits += 1
                track.lastSeen = now
                nextTracks.append(track)
            } else {
                nextTracks.append(TrackedPlate(rect: detection.rect, quad: incomingQuad,
                                               confidence: detection.confidence, hits: 1, lastSeen: now))
            }
        }
        // Tracks that missed this cycle keep their last geometry (frozen) and simply age
        // out via the `lastSeen` prune above — that's the anti-flicker persistence.
        for (index, track) in trackedPlates.enumerated() where !matchedIndices.contains(index) {
            nextTracks.append(track)
        }
        trackedPlates = nextTracks
        return confirmedTracks()
    }

    /// The currently confirmed tracks, reboxed as `PlateDetection`s for the overlay.
    private func confirmedTracks() -> [PlateDetection] {
        trackedPlates
            .filter { $0.hits >= Self.confirmationHits }
            .map { PlateDetection(rect: $0.rect, confidence: $0.confidence, label: "license_plate",
                                  keypoints: $0.quad, keypointConfidences: nil) }
    }

    /// Single publish path for the live overlay. `updateTracks` is true only for the
    /// intermediate detection callback (the one place per analysis cycle that folds raw
    /// detections into the tracker); the final publish of the same frame reuses those
    /// tracks. Results pass through untouched — a newly recognized plate keeps its label
    /// even on the confirming frame, and OCR'd plates are already validated downstream.
    private func publishLiveOverlay(detections: [PlateDetection], frameIdentifier: Int,
                                    results: [PlateResult]) {
        let updateTracks = foldedFrameIdentifier != frameIdentifier
        let stabilized: [PlateDetection]
        if updateTracks {
            stabilized = stabilizeDetections(detections)
            foldedFrameIdentifier = frameIdentifier
        } else {
            stabilized = confirmedTracks()
        }
        liveDetections = stabilized
        previewView?.draw(detections: stabilized, results: results)
    }

    /// Shared context for the debug echo (thread-safe; created once).
    private nonisolated static let debugImageContext = CIContext()

    /// Downscaled CGImage of the analysed buffer, in the same upright orientation the
    /// detector saw it (so the echo can be compared against the live preview directly).
    private nonisolated func debugThumbnail(from buffer: CVPixelBuffer,
                                            orientation: CGImagePropertyOrientation) -> CGImage? {
        let width = CGFloat(CVPixelBufferGetWidth(buffer))
        let height = CGFloat(CVPixelBufferGetHeight(buffer))
        guard width > 0, height > 0 else { return nil }
        // `oriented` changes the extent (and its origin); normalize to (0,0) before scaling.
        var image = CIImage(cvPixelBuffer: buffer).oriented(orientation)
        image = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX,
                                                       y: -image.extent.minY))
        let maxDimension: CGFloat = 360
        let scale = min(maxDimension / image.extent.width, maxDimension / image.extent.height, 1)
        let scaled = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        return Self.debugImageContext.createCGImage(scaled, from: scaled.extent)
    }

    /// Mirrors the aggregator's listed set into the published history array.
    ///
    /// Equality-gated on purpose: the array is rebuilt on every analysis cycle (~5 fps) and
    /// `@Observable` notifies on *every* assignment, which would re-evaluate the whole
    /// history list — up to 300 rows — five times a second for no visible change.
    private func publishHistory(_ aggregates: [PlateVoteAggregate]) {
        let entries = aggregates.map {
            CameraRecognitionLogEntry(id: $0.id, date: $0.firstSeen,
                                      result: $0.result, votes: $0.votes,
                                      totalVotes: $0.totalVotes)
        }
        guard entries != recognitionHistory else { return }
        recognitionHistory = entries
    }

    func clearRecognitionHistory() {
        recognitionHistory.removeAll()
        lastPrinted.removeAll()
        voteAggregator.reset()
        // Invalidate the frame that may be in flight: its final publish only checks the
        // generation token, so without this bump a pass that started before the tap would
        // re-list its plates (and re-draw the labels) a few hundred ms later.
        generation += 1
        // Reset everything the list was derived from, so "清空" is a real reset and not a
        // one-second-later reappearance:
        //  • the overlay tracker held boxes whose `hits` were accumulated for the old
        //    session — dropping them means the next frame re-confirms from scratch;
        //  • the on-screen result cards/labels are part of the same accumulated state.
        // The capture session itself keeps running (this is not 停止).
        trackedPlates.removeAll()
        foldedFrameIdentifier = -1
        liveResults = []
        liveDetections = []
        isAnalyzing = false
        previewView?.clearOverlay()
    }

    init(recognizer: LicensePlateRecognizing, mode: RecognitionMode = .fast) {
        self.recognizer = recognizer
        self.mode = mode
        super.init()
    }

    @discardableResult
    func setup() async -> Bool {
        active = true
        refreshAnalysisGate()
        let activationGeneration = generation
        let granted: Bool
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: granted = true
        case .notDetermined: granted = await AVCaptureDevice.requestAccess(for: .video)
        default: granted = false
        }
        isAuthorized = granted
        guard granted, active, generation == activationGeneration, !Task.isCancelled else { return false }
        if !didSetup {
            let ownsConfiguration = configurationTask == nil
            if ownsConfiguration {
                configurationTask = Task {
                    await withCheckedContinuation { continuation in
                        sessionQueue.async { continuation.resume(returning: self.configureSession()) }
                    }
                }
            }
            guard let task = configurationTask else { return false }
            let ok = await task.value
            if ownsConfiguration {
                setupFailed = !ok
                didSetup = ok
                configurationTask = nil
            }
            if !ok { errorMessage = "相机初始化失败"; return false }
        }
        guard active, generation == activationGeneration, !Task.isCancelled else { return false }
        previewView?.ensurePortraitOrientation()
        sessionQueue.async { if !self.session.isRunning { self.session.startRunning() } }
        return true
    }

    private nonisolated func configureSession() -> Bool {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .hd1920x1080
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: camera), session.canAddInput(input) else { return false }
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "licenseplate.camera.frames"))
        guard session.canAddOutput(output) else { return false }
        session.addInput(input)
        session.addOutput(output)
        // NOTE: we deliberately do NOT set the data-output connection's `videoRotationAngle`.
        // That hardware rotation is documented as unreliable — the front camera ignores it,
        // some rear configurations do too, and a connection can still deliver landscape
        // buffers even after 90° was requested. Instead the *analysis* path keeps the raw
        // landscape buffer and hands it to the SDK with `orientation: .right`, so the one
        // render that produces the detector's CGImage also rotates it (no second full-frame
        // pass, no extra 8 MB buffer). The *preview* layer keeps its own portrait rotation
        // (set in `CameraPreview.ensurePortraitOrientation`), which is a separate, reliable
        // display path.
        return true
    }

    /// Halts the capture session and cancels in-flight analysis, but keeps `liveResults`
    /// so the result cards can be shown again when the camera tab returns. Used when
    /// switching to another media source or when the app is backgrounded — this is the
    /// "pause the hardware" path, distinct from a full teardown that would also wipe the
    /// recognition history. Keeping the controller (and its `AVCaptureVideoPreviewLayer`)
    /// mounted across tab switches avoids the black flash caused by tearing the preview
    /// layer down and rebuilding it.
    func suspendSession() {
        active = false
        generation += 1
        refreshAnalysisGate()
        // Freeze the history here too: the preview keeps its last frame, so no further
        // analysis (and therefore no track expiry) will run until the camera returns.
        publishHistory(voteAggregator.finish())
        trackedPlates.removeAll()
        foldedFrameIdentifier = -1
        previewView?.clearOverlay()
        sessionQueue.async { if self.session.isRunning { self.session.stopRunning() } }
    }

    // MARK: - Frame analysis (off the main actor)

    /// Heavy recognition runs on a detached background task so the UI thread is never
    /// blocked by ONNX inference + OCR. All state/overlay writes hop back to the main
    /// actor explicitly (see `captureOutput`). The detection overlay mapping
    /// (`PreviewView.toLayer`, aspect-fill) is untouched — only the scheduling changed.
    private nonisolated func performAnalysis(frame: CameraFrame, token: Int, frameIdentifier: Int,
                                             mode: RecognitionMode,
                                             recognizer: LicensePlateRecognizing) async {
        let started = Date()
        let signpost = Self.signposter.beginInterval("frame")
        // Both cleanups are deferred: a cancellation between the render and the publish
        // would otherwise leave the gate busy forever and silently stop live detection.
        defer { Self.signposter.endInterval("frame", signpost) }
        defer { frameGate.finish() }
        do {
            // The raw buffer is the sensor's native landscape frame. Hand it to the SDK
            // as-is with `orientation: .right` and let `CIImage.oriented` rotate it inside
            // the render that already has to happen — the old path first rotated into a
            // brand-new 8 MB BGRA buffer, then had the SDK render the same frame a second
            // time. Detections still come back in upright portrait space, so the overlay
            // mapping, the corrector and the vote tracker are unaffected.
            let result = try await recognizer.recognizeFrame(frame.buffer, orientation: .right,
                                                             mode: mode, tiled: false) {
                [weak self] detections, _ in
                Task { @MainActor in
                    guard let self, self.active, self.liveEnabled, !self.paused,
                          self.generation == token, self.analysisIdentifier == frameIdentifier else { return }
                    // Keep the previous cycle's readings instead of publishing an empty
                    // result set: the labels/tints are derived from `results`, so passing
                    // `[]` here (the old behaviour) tore every chip off the preview at the
                    // start of each analysis cycle and only put it back when OCR finished —
                    // the text visibly blinked off for most of every ~0.4 s cycle.
                    self.publishLiveOverlay(detections: detections, frameIdentifier: frameIdentifier,
                                            results: self.liveResults)
                }
            }
            let latency = Int(Date().timeIntervalSince(started) * 1000)
            await MainActor.run { [weak self] in
                guard let self, self.active, self.liveEnabled, !self.paused, self.generation == token else { return }
                self.isAnalyzing = false
                self.elapsedMilliseconds = latency
                // Triage values for the status pill (see `lastTopScore`).
                if let plateRecognizer = recognizer as? LicensePlateRecognizer {
                    self.detectorName = plateRecognizer.detectorInfo
                }
                #if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc) || canImport(onnxruntime_objc)
                self.lastTopScore = (recognizer as? LicensePlateRecognizer)?.detectorTopScore
                #endif
                self.liveResults = result.plates
                self.publishLiveOverlay(detections: result.detections, frameIdentifier: frameIdentifier,
                                        results: result.plates)
                self.errorMessage = nil
                // Diagnostic echo: the analysed (upright) buffer + the same boxes, so the
                // preview-vs-buffer alignment question can be answered on device.
                if self.debugEchoEnabled {
                    self.previewView?.showDebugFrame(
                        image: self.debugThumbnail(from: frame.buffer, orientation: .right),
                        detections: result.detections)
                }
                // Cross-frame voting decides what the history lists: a reading must repeat
                // and hold a majority before it appears, so a one-frame misread never gets
                // its own row, and a plate seen at 1s/3s/5s stays a single row whose text
                // and image follow the winning vote (not whichever frame finished last).
                self.publishHistory(self.voteAggregator.ingest(result.plates, at: Date()))
                // Throttled per text — diagnostics only, not the list.
                for plate in result.plates {
                    let now = Date()
                    if now.timeIntervalSince(self.lastPrinted[plate.text] ?? .distantPast) >= 2 {
                        let box = String(describing: plate.boundingBox)
                        Self.logger.debug(
                            "plate=\(plate.text, privacy: .public) confidence=\(Double(plate.confidence)) box=\(box, privacy: .public) latency=\(latency)ms")
                        self.lastPrinted[plate.text] = now
                    }
                }
                self.lastPrinted = self.lastPrinted.filter { Date().timeIntervalSince($0.value) < 10 }
            }
        } catch {
            await MainActor.run { [weak self] in
                guard let self, self.active, self.generation == token else { return }
                self.isAnalyzing = false
                self.errorMessage = error.localizedDescription
                self.liveResults = []
                self.liveDetections = []
                self.trackedPlates.removeAll()
                self.foldedFrameIdentifier = -1
                self.previewView?.clearOverlay()
            }
        }
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                                   from connection: AVCaptureConnection) {
        // First line on the capture queue: an idle, paused or suspended camera drops the
        // frame here. `begin()` also keeps at most one analysis in flight and enforces the
        // 0.2 s spacing, so a dropped frame costs no rotation, no inference and no OCR.
        guard frameGate.isEnabled,
              let buffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              frameGate.begin() else { return }
        let frame = CameraFrame(buffer: buffer)
        // Mint the generation token and check gating on the main actor (cheap), then run the
        // heavy recognition off the main actor so the scan beam, controls and latency never
        // stall. Final state/overlay writes happen back on the main actor.
        Task { @MainActor in
            guard self.active, self.liveEnabled, !self.paused else { self.frameGate.finish(); return }
            self.analysisIdentifier += 1
            self.frameCount += 1
            let token = self.generation
            let frameIdentifier = self.analysisIdentifier
            self.isAnalyzing = true
            let mode = self.mode
            let recognizer = self.recognizer
            Task.detached(priority: .userInitiated) { [weak self] in
                await self?.performAnalysis(frame: frame, token: token, frameIdentifier: frameIdentifier,
                                           mode: mode, recognizer: recognizer)
            }
        }
    }
}
