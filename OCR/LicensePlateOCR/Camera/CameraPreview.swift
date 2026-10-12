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

    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        (layer as? AVCaptureVideoPreviewLayer) ?? AVCaptureVideoPreviewLayer()
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        overlayLayer.fillColor = nil
        overlayLayer.strokeColor = UIColor.systemYellow.cgColor
        overlayLayer.lineWidth = 3
        videoPreviewLayer.addSublayer(overlayLayer)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        ensurePortraitOrientation()
        overlayLayer.frame = bounds
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
        guard imageSize.width > 0, imageSize.height > 0 else { overlayLayer.path = nil; return }
        // Boxes refer to rotated pixels. Map aspect-fill without rotating a second time.
        let scale = max(bounds.width / imageSize.width, bounds.height / imageSize.height)
        let offsetX = (bounds.width - imageSize.width * scale) / 2
        let offsetY = (bounds.height - imageSize.height * scale) / 2
        func convert(_ rect: CGRect) -> CGRect {
            CGRect(x: offsetX + rect.minX * imageSize.width * scale,
                   y: offsetY + rect.minY * imageSize.height * scale,
                   width: rect.width * imageSize.width * scale,
                   height: rect.height * imageSize.height * scale)
        }
        for detection in detections {
            let box = convert(detection.rect)
            path.addRect(box)
        }
        for result in results {
            let box = convert(result.boundingBox)
            let label = CATextLayer()
            label.string = "\(result.text)  \(Int(result.confidence * 100))%"
            label.fontSize = 14
            label.contentsScale = traitCollection.displayScale
            label.foregroundColor = UIColor.black.cgColor
            label.backgroundColor = UIColor.systemYellow.cgColor
            label.frame = CGRect(x: max(0, box.minX), y: max(0, box.minY - 23),
                                 width: min(bounds.width, max(140, box.width)), height: 23)
            overlayLayer.addSublayer(label)
            labels.append(label)
        }
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        overlayLayer.path = path
        CATransaction.commit()
    }

    func clearOverlay() {
        detections = []
        results = []
        labels.forEach { $0.removeFromSuperlayer() }
        labels = []
        overlayLayer.path = nil
    }
}
