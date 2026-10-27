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
    /// Size (in pixels) of the upright portrait frame the SDK analyzes. The preview layer
    /// shows that same portrait frame (the raw landscape buffer rotated to portrait by the
    /// preview connection), so the overlay maps detection points through an aspect-fill
    /// transform keyed on this size — not through `layerPointConverted`, whose capture-device
    /// space is the *unrotated* (landscape) buffer and would double-rotate the boxes.
    private var imageSize: CGSize = .zero
    private var labels: [CATextLayer] = []
    /// One layer per plate: semi-transparent solid fill + same-colour solid outline.
    private var plateLayers: [CAShapeLayer] = []
    // Detection points are normalized (top-left, 0..1) in the upright portrait frame the
    // SDK analyzes. The preview shows that same portrait frame, so `toLayer` maps them with
    // a plain aspect-fill transform (see the function) — no rotate/mirror math, no
    // `layerPointConverted`, nothing that can drift between frames or devices.

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
        self.imageSize = imageSize
        renderOverlay()
    }

    /// Maps a detection-space point (normalized, top-left origin, in the upright portrait
    /// frame the SDK analyzes) onto the preview layer's bounds.
    ///
    /// We do **not** use `layerPointConverted(fromCaptureDevicePoint:)`: per Apple's own
    /// definition the capture-device coordinate space is the *unrotated* (landscape) sensor
    /// picture, and that method applies the preview connection's rotation as part of the
    /// conversion. Our detection points are already portrait, so feeding them in would
    /// rotate them a second time and push every box ~90° off — the exact drift that was
    /// reported. Instead we map portrait→portrait directly: the preview displays the same
    /// portrait frame (raw landscape buffer rotated to portrait by the preview connection),
    /// scaled with `videoGravity = .resizeAspectFill`, so an aspect-fill transform keyed on
    /// `imageSize` is exact and leaves no rotate/mirror ambiguity.
    private func toLayer(_ point: CGPoint) -> CGPoint {
        let bounds = videoPreviewLayer.bounds
        guard bounds.width > 0, bounds.height > 0,
              imageSize.width > 0, imageSize.height > 0 else {
            // Defensive fallback before the first frame's size is known.
            return CGPoint(x: point.x * bounds.width, y: point.y * bounds.height)
        }
        let scale = max(bounds.width / imageSize.width, bounds.height / imageSize.height)
        let contentW = imageSize.width * scale
        let contentH = imageSize.height * scale
        let offsetX = (bounds.width - contentW) * 0.5
        let offsetY = (bounds.height - contentH) * 0.5
        return CGPoint(x: offsetX + point.x * contentW,
                       y: offsetY + point.y * contentH)
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
        // Detection points are normalized (top-left, 0..1) in the upright portrait frame.
        // `toLayer` maps them with a plain aspect-fill transform that matches what the
        // preview layer actually draws, so boxes track the video exactly under aspect-fill
        // and any layout change — there is no rotate/mirror step that could drift.
        func toLayerRect(_ rect: CGRect) -> CGRect {
            // NB: the outer `toLayer(_ point: CGPoint)` is the point mapper. Name this
            // helper differently so the calls below resolve to that CGPoint version,
            // not to this CGRect overload (shadowing caused CGPoint→CGRect errors).
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
            detectionPath.addRect(toLayerRect(detection.rect))
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
