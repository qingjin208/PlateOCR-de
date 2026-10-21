/*
 License Plate OCR SDK — Perspective plate corrector

 Abstract:
 Given a detection's 4 corner keypoints (left_top, right_top, right_bottom,
 left_bottom), rectifies the plate into a fronto-parallel image using Core Image's
 `CIPerspectiveCorrection`. The rectified image is what gets handed to the OCR
 engine, which is far more robust than running OCR on a skewed/trapezoidal crop.

 Correcting a quad that is geometrically impossible is worse than not correcting at
 all: a self-intersecting or degenerate quad warps the plate into unreadable garbage
 and OCR then fails on an image we could have cropped by box instead. So the corners
 are validated first (convexity, area, side ratios, aspect, minimum size) and the
 caller falls back to a bounding-box crop when they do not describe a plate.

 If the corners are missing or below the visibility threshold, `correctedPlate`
 returns `nil` and the recognizer falls back to a plain bounding-box crop.
 */

import Foundation
import CoreGraphics
import CoreImage

public struct PerspectivePlateCorrector: Sendable {
    private let context: CIContext

    /// Mean corner visibility required to trust the keypoints. Ultralytics YOLOv8-pose
    /// ONNX exports run the keypoint branch's visibility through a Sigmoid, so it is a
    /// [0, 1] confidence — NOT the COCO label convention (0 absent / 1 occluded / 2
    /// visible). A threshold above 1.0 therefore rejects every detection and disables
    /// rectification entirely (the old default was 1.5, which silently did exactly that).
    ///
    /// This is a gate on the *mean* because a single occluded corner (bolt, frame, glare)
    /// still leaves a usable quad; the hard floor below rejects corners that are really
    /// absent, and the geometry checks reject quads that do not describe a plate.
    private let minAverageVisibility: Float
    /// Any corner below this is treated as "not actually detected".
    private let hardMinVisibility: Float
    /// Upper bound for the long edge of the rectified crop handed to OCR.
    private let maxOutputEdge: CGFloat

    public init(context: CIContext = CIContext(options: [.cacheIntermediates: false]),
                minAverageVisibility: Float = 0.3,
                hardMinVisibility: Float = 0.1,
                maxOutputEdge: CGFloat = 960) {
        self.context = context
        self.minAverageVisibility = minAverageVisibility
        self.hardMinVisibility = hardMinVisibility
        self.maxOutputEdge = maxOutputEdge
    }

    /// Rectifies a detected plate using its 4 corner keypoints.
    /// - Parameters:
    ///   - image: the oriented source frame (same coordinate space the detection was produced in).
    ///   - detection: detection carrying 4 normalized top-left corner keypoints.
    ///   - minVisibility: mean corner visibility required to trust the keypoints
    ///     (see `minAverageVisibility`; kept as a parameter for callers that want to be
    ///     stricter for a specific frame).
    /// - Returns: a rectified `CGImage`, or `nil` when the corners are unusable.
    public func correctedPlate(from image: CGImage, detection: PlateDetection,
                               minVisibility: Float? = nil) -> CGImage? {
        guard let keypoints = detection.keypoints, keypoints.count == 4 else { return nil }
        if let vis = detection.keypointConfidences, vis.count == 4 {
            guard !vis.contains(where: { $0 < hardMinVisibility }) else { return nil }
            let mean = vis.reduce(0, +) / 4
            guard mean >= (minVisibility ?? minAverageVisibility) else { return nil }
        }

        let w = CGFloat(image.width), h = CGFloat(image.height)
        let imageSize = CGSize(width: w, height: h)
        // Reject quads that cannot be a plate *before* asking Core Image to warp them.
        guard let outputSize = Self.validatedQuad(keypoints,
                                                  imageSize: imageSize,
                                                  box: detection.rect,
                                                  maxEdge: maxOutputEdge) else { return nil }

        // Detection keypoints are normalized, top-left origin (y-down). Core Image uses a
        // bottom-left origin in pixels, so flip Y: ciY = height * (1 - ny).
        let toCI: (CGPoint) -> CIVector = { p in
            CIVector(x: p.x * w, y: h * (1 - p.y))
        }

        let ci = CIImage(cgImage: image)
        guard let filter = CIFilter(name: "CIPerspectiveCorrection") else { return nil }
        filter.setValue(ci, forKey: kCIInputImageKey)
        filter.setValue(toCI(keypoints[0]), forKey: "inputTopLeft")      // left_top
        filter.setValue(toCI(keypoints[1]), forKey: "inputTopRight")     // right_top
        filter.setValue(toCI(keypoints[2]), forKey: "inputBottomRight")  // right_bottom
        filter.setValue(toCI(keypoints[3]), forKey: "inputBottomLeft")   // left_bottom

        guard let output = filter.outputImage else { return nil }
        let extent = output.extent
        // `CGRect` has no `isFinite`; check the components (an infinite / null extent means
        // the warp degenerated, which happens when the quad is nearly collinear).
        guard !extent.isInfinite, !extent.isNull,
              extent.width.isFinite, extent.height.isFinite,
              extent.origin.x.isFinite, extent.origin.y.isFinite,
              extent.width > 1, extent.height > 1 else { return nil }

        // Normalise to the quad's own dimensions. `CIPerspectiveCorrection` derives its
        // extent from the input points, which can be very large for a plate filling the
        // frame; resampling to a predictable size keeps OCR cost stable and caps memory.
        let normalized = output.transformed(
            by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
        let scaled = normalized.transformed(
            by: CGAffineTransform(scaleX: outputSize.width / extent.width,
                                  y: outputSize.height / extent.height))
        return context.createCGImage(scaled, from: CGRect(origin: .zero, size: outputSize))
    }

    // MARK: - Geometry validation

    /// Tolerance for the plate centroid relative to the detection box, as a fraction of the
    /// box size. The keypoints describe the plate, which is inside its own box, so this
    /// only absorbs small coordinate round-off — not a quad in a different part of the frame.
    private static let boxInPixelsSlack: CGFloat = 0.25

    /// Validates the 4 normalized corners as a plate quadrilateral and returns the size the
    /// rectified image should have, or `nil` when the quad is not a plausible plate.
    private static func validatedQuad(_ keypoints: [CGPoint],
                                      imageSize: CGSize,
                                      box: CGRect,
                                      maxEdge: CGFloat) -> CGSize? {
        guard keypoints.count == 4, imageSize.width > 0, imageSize.height > 0 else { return nil }
        let p = keypoints.map { CGPoint(x: $0.x * imageSize.width, y: $0.y * imageSize.height) }
        guard p.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else { return nil }

        // A little slack: a plate at the frame border can legitimately have a corner just
        // outside it. Anything further out means the keypoints are not tracking the plate.
        let marginX = imageSize.width * 0.05, marginY = imageSize.height * 0.05
        guard p.allSatisfy({ $0.x >= -marginX && $0.x <= imageSize.width + marginX
                          && $0.y >= -marginY && $0.y <= imageSize.height + marginY }) else { return nil }

        // Signed area (shoelace). Near-zero area = the corners collapsed onto a line.
        var doubleArea: CGFloat = 0
        for i in 0..<4 {
            let a = p[i], b = p[(i + 1) % 4]
            doubleArea += a.x * b.y - b.x * a.y
        }
        let signedArea = doubleArea / 2
        guard abs(signedArea) > 24 else { return nil }

        // Convexity: every consecutive cross product must share the area's sign. A reflex
        // vertex means the corner order is wrong (or the quad self-intersects), which makes
        // the warp fold the plate over itself.
        let orientation: CGFloat = signedArea > 0 ? 1 : -1
        for i in 0..<4 {
            let a = p[i], b = p[(i + 1) % 4], c = p[(i + 2) % 4]
            let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            guard cross * orientation > 0 else { return nil }
        }

        let top = distance(p[0], p[1]), bottom = distance(p[3], p[2])
        let left = distance(p[0], p[3]), right = distance(p[1], p[2])
        // Below these sizes OCR has nothing to read; a box crop is the better fallback.
        guard top > 12, bottom > 12, left > 8, right > 8 else { return nil }

        // Opposite sides of a photographed plate stay within a factor of ~2.5 under
        // perspective. A larger ratio means the "quad" is a triangle-ish artefact.
        let horizontalRatio = max(top, bottom) / max(1, min(top, bottom))
        let verticalRatio = max(left, right) / max(1, min(left, right))
        guard horizontalRatio <= 2.5, verticalRatio <= 2.5 else { return nil }

        let width = (top + bottom) / 2, height = (left + right) / 2
        let aspect = width / height
        // German plates are ~4.7:1; strong foreshortening can halve that, but anything
        // near 1:1 or 12:1 is not a plate.
        guard aspect >= 1.2, aspect <= 12 else { return nil }

        // The quad must roughly coincide with the detector's box, otherwise the corners
        // belong to something else in the crop.
        let boxArea = box.width * imageSize.width * box.height * imageSize.height
        if boxArea > 1 {
            let coverage = abs(signedArea) / boxArea
            guard coverage >= 0.2, coverage <= 1.6 else { return nil }
        }
        // Area alone is not enough: a correctly sized quad that sits in a different corner
        // of the frame has the same area. Anchor it to the detection box.
        let centroid = CGPoint(x: p.map(\.x).reduce(0, +) / 4, y: p.map(\.y).reduce(0, +) / 4)
        let boxInPixels = CGRect(x: box.minX * imageSize.width, y: box.minY * imageSize.height,
                                 width: box.width * imageSize.width,
                                 height: box.height * imageSize.height)
            .insetBy(dx: -boxInPixelsSlack * box.width * imageSize.width,
                     dy: -boxInPixelsSlack * box.height * imageSize.height)
        guard boxInPixels.contains(centroid) else { return nil }

        let scale = min(1, maxEdge / max(width, height))
        return CGSize(width: max(16, (width * scale).rounded()),
                      height: max(8, (height * scale).rounded()))
    }

    private static func distance(_ a: CGPoint, _ b: CGPoint) -> CGFloat {
        hypot(b.x - a.x, b.y - a.y)
    }
}
