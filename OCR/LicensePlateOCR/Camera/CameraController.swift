@preconcurrency import AVFoundation
import SwiftUI
import CoreImage

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
                previewView?.clearOverlay()
            }
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
        // Physically rotate once; the SDK receives upright portrait frames.
        if let connection = output.connection(with: .video), connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }
        return true
    }

    func stop() {
        active = false
        generation += 1
        liveResults = []
        liveDetections = []
        previewView?.clearOverlay()
        sessionQueue.async { if self.session.isRunning { self.session.stopRunning() } }
    }

    private func analyze(_ frame: CameraFrame) async {
        defer { waitingForOCR = false; isAnalyzing = false; frameGate.finish() }
        guard active, liveEnabled else { return }
        isAnalyzing = true
        waitingForOCR = true
        analysisIdentifier += 1
        let frameIdentifier = analysisIdentifier
        let token = generation
        let started = Date()
        do {
            let result = try await recognizer.recognizeFrame(frame.buffer, orientation: .up, mode: mode) {
                [weak self] detections, size in
                Task { @MainActor in
                    guard let self, self.active, self.liveEnabled, self.waitingForOCR,
                          self.generation == token, self.analysisIdentifier == frameIdentifier else { return }
                    self.liveDetections = detections
                    self.frameSize = size
                    self.previewView?.draw(detections: detections, results: [], imageSize: size)
                }
            }
            // Diagnostic echo: the analyzed buffer as-is + the same boxes, so the
            // preview-vs-buffer alignment question can be answered on device.
            if debugEchoEnabled {
                previewView?.showDebugFrame(image: debugThumbnail(from: frame.buffer),
                                            detections: result.detections)
            }
            guard active, liveEnabled, generation == token else { return }
            waitingForOCR = false
            elapsedMilliseconds = Int(Date().timeIntervalSince(started) * 1000)
            liveResults = result.plates
            liveDetections = result.detections
            frameSize = result.imageSize
            errorMessage = nil
            previewView?.draw(detections: result.detections, results: result.plates, imageSize: result.imageSize)
            for plate in result.plates {
                let now = Date()
                let locationKey = "\(Int(plate.boundingBox.midX * 10)):\(Int(plate.boundingBox.midY * 10))"
                let historyKey = "\(plate.text):\(locationKey)"
                if now.timeIntervalSince(lastPrinted[historyKey] ?? .distantPast) >= 2 {
                    print("[PlateOCR] \(plate.text) confidence=\(plate.confidence) box=\(plate.boundingBox) latency=\(elapsedMilliseconds)ms")
                    lastPrinted[historyKey] = now
                    let detectionNumber = (result.detections.firstIndex { $0.rect.contains(plate.boundingBox) }
                        ?? result.detections.firstIndex { $0.rect.intersects(plate.boundingBox) }
                        ?? -1) + 1
                    recognitionHistory.append(CameraRecognitionLogEntry(date: now, result: plate,
                                                                          detectionNumber: detectionNumber))
                }
            }
            if recognitionHistory.count > 300 {
                recognitionHistory.removeFirst(recognitionHistory.count - 300)
            }
            lastPrinted = lastPrinted.filter { Date().timeIntervalSince($0.value) < 10 }
        } catch {
            guard active, generation == token else { return }
            errorMessage = error.localizedDescription
            liveResults = []
            liveDetections = []
            previewView?.clearOverlay()
        }
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                                   from connection: AVCaptureConnection) {
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer), frameGate.begin() else { return }
        let frame = CameraFrame(buffer: buffer)
        Task { @MainActor in await analyze(frame) }
    }
}
