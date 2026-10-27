@preconcurrency import AVFoundation
import SwiftUI
import CoreImage
import ImageIO

/// Retains the buffer until asynchronous recognition finishes.
private struct CameraFrame: @unchecked Sendable { let buffer: CVPixelBuffer }

/// Drops frames while recognition is busy instead of creating a backlog.
private final class FrameGate: @unchecked Sendable {
    private let lock = NSLock()
    private var busy = false
    private var lastStart = Date.distantPast
    func begin() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !busy, Date().timeIntervalSince(lastStart) >= 0.2 else { return false }
        busy = true
        lastStart = Date()
        return true
    }
    func finish() {
        lock.lock()
        busy = false
        lock.unlock()
    }
}

struct CameraRecognitionLogEntry: Identifiable {
    let id = UUID()
    let date: Date
    let result: PlateResult
    let detectionNumber: Int
}

@MainActor @Observable
final class CameraController: NSObject {
    nonisolated let session = AVCaptureSession()
    private nonisolated let sessionQueue = DispatchQueue(label: "licenseplate.camera.session")
    private nonisolated let frameGate = FrameGate()
    private let recognizer: LicensePlateRecognizing
    private var didSetup = false
    @ObservationIgnored private var configurationTask: Task<Bool, Never>?
    private var waitingForOCR = false
    private var active = false
    private var generation = 0
    private var analysisIdentifier = 0
    private var lastPrinted: [String: Date] = [:]
    var isAuthorized = false
    var setupFailed = false
    var liveResults: [PlateResult] = []
    var liveDetections: [PlateDetection] = []
    var recognitionHistory: [CameraRecognitionLogEntry] = []
    var frameSize = CGSize.zero
    var isAnalyzing = false
    var elapsedMilliseconds = 0
    var errorMessage: String?
    var mode: RecognitionMode = .fast
    var liveEnabled = true {
        didSet {
            generation += 1
            if !liveEnabled {
                liveResults = []
                liveDetections = []
                isAnalyzing = false
                waitingForOCR = false
                trackedPlates.removeAll()
                foldedFrameIdentifier = -1
                previewView?.clearOverlay()
            }
        }
    }
    /// When true, analysis is halted (frozen) but the camera keeps showing the last frame,
    /// and results + overlay stay on screen. Toggled by the pause button — distinct from
    /// `liveEnabled`, which fully stops the live feature and clears everything.
    var paused = false
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

    /// Renders the software-rotated buffers. CIContext is thread-safe for rendering, so
    /// the per-frame rotation and the debug thumbnail both draw through it. Declared
    /// `nonisolated static` so the off-main analysis task can use it without hopping.
    private nonisolated static let ciContext = CIContext(options: [.cacheIntermediates: false])

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
                                    imageSize: CGSize, results: [PlateResult]) {
        let updateTracks = foldedFrameIdentifier != frameIdentifier
        let stabilized: [PlateDetection]
        if updateTracks {
            stabilized = stabilizeDetections(detections)
            foldedFrameIdentifier = frameIdentifier
        } else {
            stabilized = confirmedTracks()
        }
        liveDetections = stabilized
        frameSize = imageSize
        previewView?.draw(detections: stabilized, results: results, imageSize: imageSize)
    }

    /// Shared context for the debug echo (thread-safe; created once).
    private nonisolated static let debugImageContext = CIContext()

    /// Downscaled CGImage of the analyzed buffer, exactly as the SDK received it.
    private nonisolated func debugThumbnail(from buffer: CVPixelBuffer) -> CGImage? {
        let width = CGFloat(CVPixelBufferGetWidth(buffer))
        let height = CGFloat(CVPixelBufferGetHeight(buffer))
        guard width > 0, height > 0 else { return nil }
        let maxDimension: CGFloat = 360
        let scale = min(maxDimension / width, maxDimension / height, 1)
        let scaled = CIImage(cvPixelBuffer: buffer)
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        return Self.debugImageContext.createCGImage(scaled, from: scaled.extent)
    }

    /// Rotates the raw sample buffer (sensor orientation, landscape for a back camera held
    /// upright) to upright portrait, so the detector and the on-screen overlay share one
    /// coordinate space. We do this in software rather than trust the data-output
    /// connection's `videoRotationAngle`, which — per Apple's own behaviour and multiple
    /// field reports — is not reliably applied: the front camera ignores it, some rear
    /// configurations do too, and a connection can still deliver landscape buffers after
    /// 90° was requested. Software rotation makes the mapping deterministic: a portrait
    /// detection box then maps to screen pixels through the preview layer's native
    /// conversion with no rotate/mirror math that can drift.
    ///
    /// `.right` (90° clockwise) is the rotation that turns the landscape sensor image into
    /// the upright portrait the user sees, matching the preview connection's 90° rotation.
    private nonisolated func rotateToPortrait(_ buffer: CVPixelBuffer) -> CVPixelBuffer? {
        let ci = CIImage(cvPixelBuffer: buffer).oriented(.right)
        let width = Int(ci.extent.width), height = Int(ci.extent.height)
        guard width > 0, height > 0 else { return nil }
        var out: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &out)
        guard status == kCVReturnSuccess, let out else { return nil }
        Self.ciContext.render(ci, to: out)
        return out
    }

    func clearRecognitionHistory() {
        recognitionHistory.removeAll()
        lastPrinted.removeAll()
    }

    init(recognizer: LicensePlateRecognizing, mode: RecognitionMode = .fast) {
        self.recognizer = recognizer
        self.mode = mode
        super.init()
    }

    @discardableResult
    func setup() async -> Bool {
        active = true
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
        // buffers even after 90° was requested. Instead we rotate the buffer to upright
        // portrait in software (see `rotateToPortrait`) so the detector and the on-screen
        // overlay always share exactly one coordinate space. The *preview* layer keeps its
        // own portrait rotation (set in `CameraPreview.ensurePortraitOrientation`), which is
        // a separate, reliable display path.
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
        // Rotate the raw buffer to upright portrait so the detector and the overlay agree
        // on orientation (see `rotateToPortrait`). Falls back to the raw buffer if the
        // GPU render fails — recognition still runs, only the orientation may be off.
        let workingBuffer = rotateToPortrait(frame.buffer) ?? frame.buffer
        do {
            let result = try await recognizer.recognizeFrame(workingBuffer, orientation: .up, mode: mode, tiled: false) {
                [weak self] detections, size in
                Task { @MainActor in
                    guard let self, self.active, self.liveEnabled, !self.paused,
                          self.generation == token, self.analysisIdentifier == frameIdentifier else { return }
                    self.publishLiveOverlay(detections: detections, frameIdentifier: frameIdentifier,
                                            imageSize: size, results: [])
                }
            }
            let latency = Int(Date().timeIntervalSince(started) * 1000)
            await MainActor.run { [weak self] in
                guard let self, self.active, self.liveEnabled, !self.paused, self.generation == token else { return }
                self.waitingForOCR = false
                self.isAnalyzing = false
                self.elapsedMilliseconds = latency
                self.liveResults = result.plates
                self.publishLiveOverlay(detections: result.detections, frameIdentifier: frameIdentifier,
                                        imageSize: result.imageSize, results: result.plates)
                self.errorMessage = nil
                // Diagnostic echo: the analyzed (already portrait) buffer + the same boxes, so the
                // preview-vs-buffer alignment question can be answered on device.
                if self.debugEchoEnabled {
                    self.previewView?.showDebugFrame(image: self.debugThumbnail(from: workingBuffer),
                                                    detections: result.detections)
                }
                for plate in result.plates {
                    let now = Date()
                    let locationKey = "\(Int(plate.boundingBox.midX * 10)):\(Int(plate.boundingBox.midY * 10))"
                    let historyKey = "\(plate.text):\(locationKey)"
                    if now.timeIntervalSince(self.lastPrinted[historyKey] ?? .distantPast) >= 2 {
                        print("[PlateOCR] \(plate.text) confidence=\(plate.confidence) box=\(plate.boundingBox) latency=\(latency)ms")
                        self.lastPrinted[historyKey] = now
                        let detectionNumber = (result.detections.firstIndex { $0.rect.contains(plate.boundingBox) }
                            ?? result.detections.firstIndex { $0.rect.intersects(plate.boundingBox) }
                            ?? -1) + 1
                        self.recognitionHistory.append(CameraRecognitionLogEntry(date: now, result: plate,
                                                                                 detectionNumber: detectionNumber))
                    }
                }
                if self.recognitionHistory.count > 300 {
                    self.recognitionHistory.removeFirst(self.recognitionHistory.count - 300)
                }
                self.lastPrinted = self.lastPrinted.filter { Date().timeIntervalSince($0.value) < 10 }
            }
        } catch {
            await MainActor.run { [weak self] in
                guard let self, self.active, self.generation == token else { return }
                self.waitingForOCR = false
                self.isAnalyzing = false
                self.errorMessage = error.localizedDescription
                self.liveResults = []
                self.liveDetections = []
                self.trackedPlates.removeAll()
                self.foldedFrameIdentifier = -1
                self.previewView?.clearOverlay()
            }
        }
        frameGate.finish()
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                                   from connection: AVCaptureConnection) {
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer), frameGate.begin() else { return }
        let frame = CameraFrame(buffer: buffer)
        // Mint the generation token and check gating on the main actor (cheap), then run the
        // heavy recognition off the main actor so the scan beam, controls and latency never
        // stall. Final state/overlay writes happen back on the main actor.
        Task { @MainActor in
            guard self.active, self.liveEnabled, !self.paused else { self.frameGate.finish(); return }
            self.analysisIdentifier += 1
            let token = self.generation
            let frameIdentifier = self.analysisIdentifier
            self.isAnalyzing = true
            self.waitingForOCR = true
            let mode = self.mode
            let recognizer = self.recognizer
            Task.detached(priority: .userInitiated) { [weak self] in
                await self?.performAnalysis(frame: frame, token: token, frameIdentifier: frameIdentifier,
                                           mode: mode, recognizer: recognizer)
            }
        }
    }
}
