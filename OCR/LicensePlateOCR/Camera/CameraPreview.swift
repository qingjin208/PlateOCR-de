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
    /// Size (in pixels) of the upright portrait frame the SDK analyzes. Retained for
    /// reference/telemetry; the overlay no longer maps through this directly — it goes via
    /// `layerPointConverted`, which applies the preview's own videoGravity + rotation, so
    /// `imageSize` is no longer part of the coordinate math.
    private var imageSize: CGSize = .zero
    private var labels: [CATextLayer] = []
    /// One layer per plate: semi-transparent solid fill + same-colour solid outline.
    private var plateLayers: [CAShapeLayer] = []
    // Detection points are normalized (top-left, 0..1) in the upright portrait frame the
    // SDK analyzes. `toLayer` converts them to native-landscape coords and routes them
    // through `layerPointConverted`, so the boxes always track the (rotated) video.

    /// Overlay layer used to draw the detection boxes; one path, many rects.
    private let overlayLayer = CAShapeLayer()
    private let plateContainerLayer = CALayer()

    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        (layer as? AVCaptureVideoPreviewLayer) ?? AVCaptureVideoPreviewLayer()
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        // The detection rectangle draws in the same unified green as the plate tint
        // (see `PlateOverlayPalette.unifiedColor`).
        overlayLayer.fillColor = nil
        overlayLayer.strokeColor = PlateOverlayPalette.unifiedColor.cgColor
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

    /// Maps a detection-space point (normalized, top-left origin, in the upright PORTRAIT
    /// frame the SDK analyzes) onto the preview layer's local coordinate space.
    ///
    /// The detection points live in the portrait frame produced by `rotateToPortrait`
    /// (the raw landscape buffer rotated 90° CW via `CIImage.oriented(.right)`). The
    /// preview layer, however, shows the *native landscape* frame and rotates it 90° CW
    /// through `videoRotationAngle`; the overlay is a plain sublayer of that layer, so it
    /// sits in the layer's UNROTATED local space while the video content is rotated
    /// within it. Mapping portrait→portrait with a naive aspect-fill therefore puts the
    /// boxes in the wrong place (the real plate's box can even fall off-screen) — the
    /// exact "定位错误 / 检测不到" symptom. The correct path is:
    ///   1. undo the 90° CW rotation to return to native-landscape normalized coords, then
    ///   2. let `layerPointConverted` apply the preview's own `videoGravity` +
    ///      `videoRotationAngle` to reach the on-screen point.
    /// This removes the entire class of rotate/mirror/aspect-fill drift the manual math
    /// was prone to, and it is the Apple-recommended way to draw detections on a preview.
    private func toLayer(_ point: CGPoint) -> CGPoint {
        // `rotateToPortrait` turns the landscape sensor image into the upright portrait
        // frame the detector analyzes via a 90° CW rotation (`CIImage.oriented(.right)`).
        // To draw a portrait-space point on the preview we must undo that rotation: the
        // inverse of a 90° CW turn is a 90° CCW turn, which maps a portrait point (px, py)
        // back to native-landscape normalized coords as (1 - py, px). `layerPointConverted`
        // then re-applies the preview's own `videoGravity` + 90° CW `videoRotationAngle`,
        // closing the loop so the box lands exactly on the plate. (Verified by a round-trip:
        // portrait (0.5, 0.4) → native (0.6, 0.5) → on-screen (0.5, 0.4).)
        let native = CGPoint(x: 1 - point.y, y: point.x)
        return videoPreviewLayer.layerPointConverted(fromCaptureDevicePoint: native)
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
        // Detection points are normalized (top-left, 0..1) in the upright portrait frame
        // the SDK analyzes. `toLayer` sends them through `layerPointConverted`, which
        // applies the preview's own videoGravity + videoRotationAngle, so the boxes track
        // the rotated video exactly under any layout or aspect-fill change.
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
        boxes.strokeColor = PlateOverlayPalette.unifiedColor.cgColor
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
