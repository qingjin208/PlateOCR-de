/*
 License Plate OCR SDK — Demo App: Camera Preview

 Abstract:
 Hosts `AVCaptureVideoPreviewLayer` and draws detection boxes on top of it.
 Boxes are mapped onto the preview with
 `layerRectConverted(fromMetadataOutputRect:)`, so they stay aligned with the
 video regardless of `videoGravity` and device orientation.
 */

import AVFoundation
import SwiftUI

struct CameraPreview: UIViewRepresentable {
    let controller: CameraController

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.session = controller.session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        controller.previewView = view
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        // Keep the session attached in case the controller was recreated.
        uiView.videoPreviewLayer.session = controller.session
    }
}

final class PreviewView: UIView {

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
        overlayLayer.frame = bounds
    }

    /// Draws normalized (top-left origin, 0...1) boxes onto the preview layer.
    /// `layerRectConverted(fromMetadataOutputRect:)` maps the image-space rect to
    /// the layer's coordinate space, accounting for `videoGravity` and orientation.
    func draw(results: [PlateResult]) {
        let path = CGMutablePath()
        for r in results {
            let box = videoPreviewLayer.layerRectConverted(fromMetadataOutputRect: r.boundingBox)
            path.addRect(box)
        }
        overlayLayer.path = path
    }

    func clearOverlay() {
        overlayLayer.path = nil
    }
}
