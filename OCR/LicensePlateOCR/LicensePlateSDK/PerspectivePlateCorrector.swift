/*
 License Plate OCR SDK — Perspective plate corrector

 Abstract:
 Given a detection's 4 corner keypoints (left_top, right_top, right_bottom,
 left_bottom), rectifies the plate into a fronto-parallel image using Core Image's
 `CIPerspectiveCorrection`. The rectified image is what gets handed to the OCR
 engine, which is far more robust than running OCR on a skewed/trapezoidal crop.

 If the corners are missing or below the visibility threshold, `correctedPlate`
 returns `nil` and the recognizer falls back to a plain bounding-box crop.
 */

import Foundation
import CoreGraphics
import CoreImage

public struct PerspectivePlateCorrector: Sendable {
    private let context: CIContext

    public init(context: CIContext = CIContext(options: [.cacheIntermediates: false])) {
        self.context = context
    }

    /// Rectifies a detected plate using its 4 corner keypoints.
    /// - Parameters:
    ///   - image: the oriented source frame (same coordinate space the detection was produced in).
    ///   - detection: detection carrying 4 normalized top-left corner keypoints.
    ///   - minVisibility: minimum per-corner visibility (YOLO pose: 0 absent, 1 occluded, 2 visible).
    /// - Returns: a rectified `CGImage`, or `nil` when the corners are unusable.
    public func correctedPlate(from image: CGImage, detection: PlateDetection,
                               minVisibility: Float = 1.5) -> CGImage? {
        guard let keypoints = detection.keypoints, keypoints.count == 4 else { return nil }
        if let vis = detection.keypointConfidences, vis.count == 4,
           vis.contains(where: { $0 < minVisibility }) {
            return nil
        }

        let w = CGFloat(image.width), h = CGFloat(image.height)
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
        guard extent.width > 1, extent.height > 1, !extent.isInfinite else { return nil }
        return context.createCGImage(output, from: extent)
    }
}
