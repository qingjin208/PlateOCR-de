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
    // Detection points are normalized (top-left, 0..1) in the upright portrait frame the
    // SDK analyzes. The preview connection is also portrait, so the capture-device
    // coordinate space already matches — no rotate/mirror math is needed; the native
    // `layerPointConverted` folds in aspect-fill and the layer bounds.

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

    /// Draws top-left normalized boxes for the upright portrait frame the SDK analyzes.
    /// `imageSize` is the pixel size of that portrait frame (e.g. 1080x1920 for an
    /// `.hd1920x1080` session after the software rotation in `CameraController`).
    func draw(detections: [PlateDetection], results: [PlateResult], imageSize: CGSize) {
        self.detections = detections
        self.results = results
        renderOverlay()
    }

    /// Maps a detection-space point (normalized, top-left origin, in the upright portrait
    /// frame the SDK analyzes) onto the preview layer. Because the frame is already
    /// portrait — the camera path software-rotates every buffer to match what the user
    /// sees, and the preview connection is also portrait — the capture-device coordinate
    /// space is identical to the frame's space. So we pass the point straight to the
    /// preview layer's native conversion, which folds in `videoGravity` (aspect-fill) and
    /// the current layer bounds with no hand-rolled rotate/mirror math that can drift.
    private func toLayer(_ point: CGPoint) -> CGPoint {
        videoPreviewLayer.layerPointConverted(fromCaptureDevicePoint: point)
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
        // Detection points are normalized (top-left, 0..1) in the delivered buffer.
        // The class method `toLayer(_:)` maps them through the preview layer's native
        // coordinate conversion, so boxes track the video exactly under aspect-fill,
        // rotation and any layout change — no hand-rolled aspect math to drift.
        func toLayer(_ rect: CGRect) -> CGRect {
            let a = toLayer(CGPoint(x: rect.minX, y: rect.minY))
            let b = toLayer(CGPoint(x: rect.maxX, y: rect.maxY))
            return CGRect(x: min(a.x, b.x), y: min(a.y, b.y),
                          width: abs(b.x - a.x), height: abs(b.y - a.y))
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
        let points = quad.map { toLayer($0) }
        let minX = points.map { $0.x }.min() ?? 0
        let minY = points.map { $0.y }.min() ?? 0
        let maxX = points.map { $0.x }.max() ?? 0
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

    /// Diagnostic echo layer (analyzed frame thumbnail + boxes), when enabled.
    private var debugLayer: CALayer?

    /// Shows the exact frame the SDK analyzed — as-is, with whatever rotation the
    /// buffer actually has — plus the same normalized detection boxes drawn on it.
    /// The thumbnail displays the whole buffer 1:1 (no crop), so if a box fits the
    /// plate here but not on the live preview, the preview mapping is wrong; if the
    /// thumbnail itself appears sideways, the output rotation was never applied.
    func showDebugFrame(image: CGImage?, detections: [PlateDetection]) {
        debugLayer?.removeFromSuperlayer()
        debugLayer = nil
        guard let image, bounds.width > 0, bounds.height > 0 else { return }
        let maxSide: CGFloat = 150
        let aspect = CGFloat(image.width) / max(CGFloat(image.height), 1)
        let size = aspect >= 1 ? CGSize(width: maxSide, height: maxSide / aspect)
                               : CGSize(width: maxSide * aspect, height: maxSide)
        let container = CALayer()
        container.frame = CGRect(x: bounds.maxX - size.width - 8,
                                 y: bounds.maxY - size.height - 8,
                                 width: size.width, height: size.height)
        container.backgroundColor = UIColor.black.cgColor
        container.borderColor = UIColor.white.withAlphaComponent(0.85).cgColor
        container.borderWidth = 1
        container.cornerRadius = 4
        container.masksToBounds = true
        let imageLayer = CALayer()
        imageLayer.frame = container.bounds
        imageLayer.contents = image
        imageLayer.contentsGravity = .resize
        container.addSublayer(imageLayer)
        let boxes = CAShapeLayer()
        let path = CGMutablePath()
        for detection in detections {
            path.addRect(CGRect(x: detection.rect.minX * size.width,
                                y: detection.rect.minY * size.height,
                                width: detection.rect.width * size.width,
                                height: detection.rect.height * size.height))
        }
        boxes.path = path
        boxes.strokeColor = UIColor.systemYellow.cgColor
        boxes.fillColor = nil
        boxes.lineWidth = 1.5
        container.addSublayer(boxes)
        videoPreviewLayer.addSublayer(container)
        debugLayer = container
    }

    func clearOverlay() {
        detections = []
        results = []
        labels.forEach { $0.removeFromSuperlayer() }
        labels = []
        plateLayers.forEach { $0.removeFromSuperlayer() }
        plateLayers = []
        overlayLayer.path = nil
        debugLayer?.removeFromSuperlayer()
        debugLayer = nil
    }
}
