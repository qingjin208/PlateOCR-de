/*
 License Plate OCR SDK — Demo App: Camera Controller

 Abstract:
 Drives the live camera. Mirrors Apple's "Locating and displaying recognized text"
 sample `Camera` class (AVCaptureSession + AVCapturePhotoOutput), but adds a
 `AVCaptureVideoDataOutput` so we can run the plate pipeline on throttled live
 frames (real-time detection) and on the captured still.

 The pipeline used here is the same `LicensePlateRecognizer` the rest of the app
 uses: with the bundled `yolov8n` (COCO) placeholder it detects *cars* and OCRs
 each car region, keeping only German-plate text. Swap in your own model later.
 */

import AVFoundation
import SwiftUI
import CoreImage
import CoreGraphics
import VideoToolbox

@Observable
final class CameraController: NSObject {

    // MARK: Capture session

    let session = AVCaptureSession()
    private let photoOutput = AVCapturePhotoOutput()
    private let videoOutput = AVCaptureVideoDataOutput()

    // MARK: State (published for SwiftUI)

    var isAuthorized = false
    var setupFailed = false
    /// Live detection boxes, refreshed on a throttle while the camera runs.
    var liveResults: [PlateResult] = []
    /// Results from the last shutter capture.
    var capturedResults: [PlateResult] = []
    var isAnalyzing = false
    /// Toggle real-time analysis of live frames on/off.
    var liveEnabled = true
    /// `true` once a still photo has been captured and the preview is frozen.
    var isFrozen = false
    var errorMessage: String?
    /// Recognition mode; bound to the app's segmented control.
    var mode: RecognitionMode = .fast

    // MARK: Private

    private let recognizer: LicensePlateRecognizing
    private let ciContext = CIContext()
    private var lastAnalysis = Date.distantPast
    private let analysisInterval: TimeInterval = 0.3
    private var didSetup = false
    /// Set by `CameraPreview` so analysis results can be drawn onto the preview.
    weak var previewView: PreviewView?

    init(recognizer: LicensePlateRecognizing, mode: RecognitionMode = .fast) {
        self.recognizer = recognizer
        self.mode = mode
        super.init()
    }

    // MARK: Authorization & setup

    func checkAuthorization() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .video)
        default:
            return false
        }
    }

    /// Configures the session (on a background thread, since session ops block).
    /// Safe to call more than once; only configures the session the first time.
    /// The returned value reflects *setup success*, while `isAuthorized` reflects
    /// the camera *permission* state (they are intentionally decoupled: a user can
    /// grant permission but the session may still fail to configure).
    @discardableResult
    func setup() async -> Bool {
        let granted = await checkAuthorization()
        await MainActor.run { self.isAuthorized = granted }
        guard granted else { return false }
        guard !didSetup else { return !setupFailed }
        let ok = await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
            Task.detached(priority: .userInitiated) {
                cont.resume(returning: self.configureSession())
            }
        }
        didSetup = true
        if !ok { await MainActor.run { self.setupFailed = true } }
        return ok
    }

    private func configureSession() -> Bool {
        session.beginConfiguration()
        session.sessionPreset = .high

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video, position: .back) else {
            session.commitConfiguration()
            return false
        }
        guard let input = try? AVCaptureDeviceInput(device: device) else {
            session.commitConfiguration()
            return false
        }
        guard session.canAddInput(input) else {
            session.commitConfiguration()
            return false
        }
        session.addInput(input)

        guard session.canAddOutput(photoOutput) else {
            session.commitConfiguration()
            return false
        }
        session.addOutput(photoOutput)

        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "licenseplate.camera.video"))
        guard session.canAddOutput(videoOutput) else {
            session.commitConfiguration()
            return false
        }
        session.addOutput(videoOutput)

        // Display & analyze in portrait.
        if let conn = videoOutput.connection(with: .video), conn.isVideoOrientationSupported {
            conn.videoOrientation = .portrait
        }
        if let conn = photoOutput.connection(with: .video), conn.isVideoOrientationSupported {
            conn.videoOrientation = .portrait
        }

        session.commitConfiguration()

        Task.detached(priority: .background) { [weak self] in
            self?.session.startRunning()
        }
        return true
    }

    func stop() {
        Task.detached(priority: .background) { [weak self] in
            self?.session.stopRunning()
        }
    }

    // MARK: Shutter

    /// Captures a still and freezes the live preview so the OCR result is stable.
    /// Pausing `liveEnabled` immediately stops the live frame delegate from
    /// overwriting the captured result boxes.
    func capturePhoto() {
        liveEnabled = false
        isFrozen = true
        photoOutput.capturePhoto(with: AVCapturePhotoSettings(), delegate: self)
    }

    /// Resumes the live preview after inspecting a captured still (retake).
    func resumeLive() {
        capturedResults = []
        errorMessage = nil
        previewView?.clearOverlay()
        isFrozen = false
        liveEnabled = true
        Task.detached(priority: .background) { [weak self] in
            self?.session.startRunning()
        }
    }

    // MARK: Frame -> CGImage

    /// Converts a video frame to a *portrait* CGImage so it matches what the user
    /// sees in the preview (the raw pixel buffer is landscape). `.right` is correct
    /// for the rear camera in portrait; flip to `.left` if your preview is rotated.
    private func portraitCGImage(from buffer: CVPixelBuffer) -> CGImage? {
        let ci = CIImage(cvPixelBuffer: buffer).oriented(.right)
        guard let cg = ciContext.createCGImage(ci, from: ci.extent) else { return nil }
        return cg
    }
}

// MARK: - Photo capture delegate

extension CameraController: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishProcessingPhoto photo: AVCapturePhoto,
                     error: Error?) {
        guard error == nil, let cg = photo.cgImageRepresentation() else {
            errorMessage = "拍摄失败"
            return
        }
        // Freeze the live preview so the captured result stays put (the preview
        // layer keeps showing the last frame after the session stops).
        stop()
        Task {
            do {
                let results = try await recognizer.recognize(in: cg, mode: mode)
                await MainActor.run {
                    capturedResults = results
                    previewView?.draw(results: results)
                }
            } catch {
                await MainActor.run { errorMessage = error.localizedDescription }
            }
        }
    }
}

// MARK: - Live video frame delegate

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureVideoDataOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard liveEnabled, !isAnalyzing else { return }
        guard Date().timeIntervalSince(lastAnalysis) >= analysisInterval else { return }
        guard let pixel = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        guard let cg = portraitCGImage(from: pixel) else { return }

        lastAnalysis = Date()
        isAnalyzing = true

        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            do {
                let results = try await self.recognizer.recognize(in: cg, mode: self.mode)
                DispatchQueue.main.async { self.previewView?.draw(results: results) }
                await MainActor.run {
                    self.liveResults = results
                    self.isAnalyzing = false
                }
            } catch {
                await MainActor.run { self.isAnalyzing = false }
            }
        }
    }
}
