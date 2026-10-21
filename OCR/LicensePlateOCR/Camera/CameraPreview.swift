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
    private var labels: [CATextLayer] = []
    /// One layer per plate: semi-transparent solid fill + same-colour solid outline.
    private var plateLayers: [CAShapeLayer] = []

    /// Overlay layer used to draw the detection boxes; one path, many rects.
    private let overlayLayer = CAShapeLayer()
    private let plateContainerLayer = CALayer()

    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        (layer as? AVCaptureVideoPreviewLayer) ?? AVCaptureVideoPreviewLayer()
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        // The detection box stays as a thin yellow rectangle; the plate tint (below) is
        // the "recognised plate area" filled with a solid colour.
        overlayLayer.fillColor = nil
        overlayLayer.strokeColor = UIColor.systemYellow.cgColor
        overlayLayer.lineWidth = PlateOverlayPalette.detectionStrokeWidth
        videoPreviewLayer.addSublayer(overlayLayer)
        videoPreviewLayer.addSublayer(plateContainerLayer)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        ensurePortraitOrientation()
        overlayLayer.frame = bounds
        plateContainerLayer.frame = bounds
        renderOverlay()
    }

    func ensurePortraitOrientation() {
        if let connection = videoPreviewLayer.connection, connection.isVideoRotationAngleSupported(90),
           connection.videoRotationAngle != 90 {
            connection.videoRotationAngle = 90
        }
    }

    /// Draws top-left normalized boxes for an already oriented portrait frame.
    /// `imageSize` is accepted for API compatibility with the controller; the actual
    /// on-screen mapping is derived from the preview layer's `videoRect`.
    func draw(detections: [PlateDetection], results: [PlateResult], imageSize: CGSize) {
        self.detections = detections
        self.results = results
        renderOverlay()
    }

    private func renderOverlay() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        labels.forEach { $0.removeFromSuperlayer() }
        labels = []
        plateLayers.forEach { $0.removeFromSuperlayer() }
        plateLayers = []
        CATransaction.commit()
        let detectionPath = CGMutablePath()
        // Tie each box to the rectangle where the preview layer *actually* draws the
        // video. `videoRect` already folds in videoGravity (aspect-fill), the layer's
        // bounds, and the connection's rotation, so a normalized (top-left, 0...1) box
        // from the detector maps straight onto the displayed frame. This removes the
        // dependency on `imageSize`/hand-rolled aspect math, which is what made the box
        // sit above the plate and creep as the frame size or layout shifted.
        let videoRect = videoPreviewLayer.videoRect
        guard !videoRect.isEmpty, videoRect.width > 0, videoRect.height > 0 else {
            overlayLayer.path = nil
            return
        }
        func toLayer(_ point: CGPoint) -> CGPoint {
            CGPoint(x: videoRect.minX + point.x * videoRect.width,
                    y: videoRect.minY + point.y * videoRect.height)
        }
        func toLayer(_ rect: CGRect) -> CGRect {
            CGRect(x: videoRect.minX + rect.minX * videoRect.width,
                   y: videoRect.minY + rect.minY * videoRect.height,
                   width: rect.width * videoRect.width,
                   height: rect.height * videoRect.height)
        }
        /// Closed quad path for four normalized plate corners.
        func quadPath(_ quad: [CGPoint]) -> CGPath? {
            guard quad.count == 4 else { return nil }
            let path = CGMutablePath()
            path.addLines(between: quad.map(toLayer))
            path.closeSubpath()
            return path
        }
        func plateLayer(path: CGPath, key: String, recognized: Bool) -> PlateOverlayColor {
            let color = PlateOverlayPalette.color(for: key)
            let layer = CAShapeLayer()
            layer.path = path
            layer.fillColor = color.cgColor(alpha: recognized ? PlateOverlayPalette.fillAlpha
                                                              : PlateOverlayPalette.fillAlpha * 0.4)
            layer.strokeColor = color.cgColor
            layer.lineWidth = recognized ? PlateOverlayPalette.strokeWidth
                                         : PlateOverlayPalette.strokeWidth * 0.6
            layer.lineJoin = .round
            plateContainerLayer.addSublayer(layer)
            plateLayers.append(layer)
            return color
        }

        var usedResultIndices = Set<Int>()
        for (index, detection) in detections.enumerated() {
            detectionPath.addRect(toLayer(detection.rect))
            // Prefer the OCR result that belongs to this detection so the colour is keyed
            // on the plate text (stable across frames); otherwise key on the position.
            let matched = results.indices.first { i in
                !usedResultIndices.contains(i) && (detection.rect.contains(results[i].boundingBox)
                    || detection.rect.intersects(results[i].boundingBox))
            }
            if let matched { usedResultIndices.insert(matched) }
            let result = matched.map { results[$0] }
            let key = PlateOverlayPalette.key(text: result?.text, box: detection.rect)
            // Resolve the quad without nested-optional chaining (`result?.corners` is an
            // optional-of-optional property access, which makes `??` ambiguous).
            var quad = detection.plateQuad
            if let result, let corners = result.corners, corners.count == 4 {
                quad = corners
            }
            guard let path = quadPath(quad) else { continue }
            let color = plateLayer(path: path, key: key, recognized: result != nil)
            if let result {
                addLabel(for: result, number: index + 1, color: color, quad: quad)
            }
        }
        // Results whose box was carved out of a shared crop (no dedicated detection).
        for (index, result) in results.enumerated() where !usedResultIndices.contains(index) {
            guard let path = quadPath(result.plateQuad) else { continue }
            let color = plateLayer(path: path,
                                   key: PlateOverlayPalette.key(text: result.text, box: result.boundingBox),
                                   recognized: true)
            addLabel(for: result, number: index + 1, color: color, quad: result.plateQuad)
        }
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        overlayLayer.path = detectionPath
        CATransaction.commit()
    }

    /// Text chip anchored to the top edge of the plate quad, tinted with the plate colour.
    private func addLabel(for result: PlateResult, number: Int, color: PlateOverlayColor, quad: [CGPoint]) {
        let videoRect = videoPreviewLayer.videoRect
        let points = quad.map { CGPoint(x: videoRect.minX + $0.x * videoRect.width,
                                        y: videoRect.minY + $0.y * videoRect.height) }
        let minX = points.map(\.x).min() ?? 0
        let minY = points.map(\.y).min() ?? 0
        let maxX = points.map(\.x).max() ?? 0
        let width = min(bounds.width, max(140, maxX - minX))
        let label = CATextLayer()
        label.string = "#\(number) \(result.text)  \(Int(result.confidence * 100))%"
        label.fontSize = 14
        label.contentsScale = traitCollection.displayScale
        label.foregroundColor = UIColor.white.cgColor
        label.backgroundColor = color.cgColor(alpha: 0.95)
        label.alignmentMode = .left
        label.frame = CGRect(x: max(0, min(bounds.width - width, minX)),
                             y: max(0, minY - 23), width: width, height: 23)
        plateContainerLayer.addSublayer(label)
        labels.append(label)
    }

    func clearOverlay() {
        detections = []
        results = []
        labels.forEach { $0.removeFromSuperlayer() }
        labels = []
        plateLayers.forEach { $0.removeFromSuperlayer() }
        plateLayers = []
        overlayLayer.path = nil
    }
}
