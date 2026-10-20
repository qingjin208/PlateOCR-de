/*
 License Plate OCR SDK — Demo App: Camera Preview

 Abstract:
 Hosts a portrait `AVCaptureVideoPreviewLayer` and draws boxes and text using
 the same oriented pixel dimensions and aspect-fill scaling as the preview.
 */

import AVFoundation
import SwiftUI

struct CameraPreview: UIViewRepresentable {
    let controller: CameraController

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.session = controller.session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        if let connection = view.videoPreviewLayer.connection, connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }
        controller.previewView = view
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        // Keep the session attached in case the controller was recreated.
        uiView.videoPreviewLayer.session = controller.session
        uiView.ensurePortraitOrientation()
    }
}

final class PreviewView: UIView {
    private var detections: [PlateDetection] = []
    private var results: [PlateResult] = []
    private var imageSize = CGSize.zero
    private var labels: [CATextLayer] = []

    /// Overlay layer used to draw detection boxes; one path, many rects.
    private let overlayLayer = CAShapeLayer()
    private let recognizedLayer = CAShapeLayer()

    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        (layer as? AVCaptureVideoPreviewLayer) ?? AVCaptureVideoPreviewLayer()
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        overlayLayer.fillColor = nil
        overlayLayer.strokeColor = UIColor.systemYellow.cgColor
        overlayLayer.lineWidth = 3
        recognizedLayer.fillColor = nil
        recognizedLayer.strokeColor = UIColor.systemGreen.cgColor
        recognizedLayer.lineWidth = 3
        videoPreviewLayer.addSublayer(overlayLayer)
        videoPreviewLayer.addSublayer(recognizedLayer)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        ensurePortraitOrientation()
        overlayLayer.frame = bounds
        recognizedLayer.frame = bounds
        renderOverlay()
    }

    func ensurePortraitOrientation() {
        if let connection = videoPreviewLayer.connection, connection.isVideoRotationAngleSupported(90),
           connection.videoRotationAngle != 90 {
            connection.videoRotationAngle = 90
        }
    }

    /// Draws top-left normalized boxes for an already oriented portrait frame.
    func draw(detections: [PlateDetection], results: [PlateResult], imageSize: CGSize) {
        self.detections = detections
        self.results = results
        self.imageSize = imageSize
        renderOverlay()
    }

    private func renderOverlay() {
        labels.forEach { $0.removeFromSuperlayer() }
        labels = []
        let path = CGMutablePath()
        let recognizedPath = CGMutablePath()
        guard imageSize.width > 0, imageSize.height > 0 else {
            overlayLayer.path = nil
            recognizedLayer.path = nil
            return
        }
        // The SDK boxes are normalized (top-left, 0...1) in the *oriented* frame the
        // preview layer shows (portrait after the connection's videoRotationAngle is
        // applied). The preview uses videoGravity = .resizeAspectFill, so map the
        // normalized box to layer points with the same aspect-fill scaling. This avoids
        // relying on `layerRectConverted(fromMetadataOutputRect:)` orientation semantics
        // (which previously double-rotated the box and drew it in the wrong place).
        let scale = max(bounds.width / imageSize.width, bounds.height / imageSize.height)
        let dispW = imageSize.width * scale
        let dispH = imageSize.height * scale
        let offX = (bounds.width - dispW) / 2
        let offY = (bounds.height - dispH) / 2
        func toLayer(_ rect: CGRect) -> CGRect {
            CGRect(x: offX + rect.minX * dispW,
                   y: offY + rect.minY * dispH,
                   width: rect.width * dispW,
                   height: rect.height * dispH)
        }
        for detection in detections {
            path.addRect(toLayer(detection.rect))
        }
        for result in results {
            let box = toLayer(result.boundingBox)
            recognizedPath.addRect(box)
            let label = CATextLayer()
            let detectionNumber = (detections.firstIndex { $0.rect == result.boundingBox } ?? -1) + 1
            label.string = "#\(detectionNumber) \(result.text)  \(Int(result.confidence * 100))%"
            label.fontSize = 14
            label.contentsScale = traitCollection.displayScale
            label.foregroundColor = UIColor.white.cgColor
            label.backgroundColor = UIColor.systemGreen.withAlphaComponent(0.95).cgColor
            label.frame = CGRect(x: max(0, box.minX), y: max(0, box.minY - 23),
                                 width: min(bounds.width, max(140, box.width)), height: 23)
            overlayLayer.addSublayer(label)
            labels.append(label)
        }
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        overlayLayer.path = path
        recognizedLayer.path = recognizedPath
        CATransaction.commit()
    }

    func clearOverlay() {
        detections = []
        results = []
        labels.forEach { $0.removeFromSuperlayer() }
        labels = []
        overlayLayer.path = nil
        recognizedLayer.path = nil
    }
}
