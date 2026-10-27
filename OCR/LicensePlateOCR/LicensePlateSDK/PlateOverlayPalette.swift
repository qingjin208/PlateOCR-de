/*
 License Plate OCR SDK — Overlay styling

 Abstract:
 Draws every plate with ONE fixed green (`unifiedColor`): a semi-transparent green fill
 plus a solid green outline, for recognized and not-yet-recognized plates alike. The
 original per-plate FNV-1a palette helpers are kept below so the key/hashing logic and
 the ability to restore distinct per-plate colours remain available.
 */

import Foundation
import CoreGraphics

/// A solid RGB colour with no alpha; callers use `cgColor(alpha:)` for fills.
public struct PlateOverlayColor: Sendable, Hashable {
    public let red: Double
    public let green: Double
    public let blue: Double

    public init(red: Double, green: Double, blue: Double) {
        self.red = red
        self.green = green
        self.blue = blue
    }

    /// Opaque colour (used for the solid outline).
    public var cgColor: CGColor { cgColor(alpha: 1) }

    /// Same colour at a custom alpha (used for the semi-transparent fill).
    public func cgColor(alpha: Double) -> CGColor {
        CGColor(red: CGFloat(red), green: CGFloat(green), blue: CGFloat(blue), alpha: CGFloat(alpha))
    }
}

public enum PlateOverlayPalette {
    /// Alpha of the solid-colour plate fill.
    public static let fillAlpha: Double = 0.28
    /// Stroke width of the matching solid outline.
    public static let strokeWidth: CGFloat = 2.5
    /// Stroke width used for the underlying detection box, drawn under the plate tint.
    public static let detectionStrokeWidth: CGFloat = 1.5

    /// The single overlay colour requested by the App (2026-10-27): every plate —
    /// recognized or not — draws in the same green instead of the per-plate palette, so
    /// the boxes no longer change colour from frame to frame.
    public static let unifiedColor = PlateOverlayColor(red: 0.10, green: 0.78, blue: 0.35)

    /// Stable colour for an arbitrary key. Returns the fixed green for every key; the
    /// hashing helpers below are kept so existing call sites (and the ability to restore
    /// per-plate colours) keep working unchanged.
    public static func color(for key: String) -> PlateOverlayColor {
        unifiedColor
    }

    /// Stable colour for a plate: keyed on its canonical text when available, else on a
    /// quantised centre so a not-yet-recognised plate still keeps a steady colour.
    public static func color(forText text: String?, box: CGRect) -> PlateOverlayColor {
        color(for: key(text: text, box: box))
    }

    public static func key(text: String?, box: CGRect) -> String {
        if let text, !text.isEmpty { return "text:" + text }
        // 10 x 10 grid: stable while the plate stays roughly in place, so the colour does
        // not flicker between frames before OCR has produced any text.
        return String(format: "pos:%.1f,%.1f", box.midX * 10, box.midY * 10)
    }

    /// FNV-1a (64-bit). Deterministic across launches, unlike `String.hashValue`,
    /// which is seeded randomly per process.
    public static func fnv1a(_ string: String) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in string.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return hash
    }
}

// MARK: - Plate outline geometry

public extension PlateDetection {
    /// The plate outline as 4 points in the same normalized (top-left origin) space as
    /// `rect`. Uses the pose corner keypoints when present, otherwise the box corners.
    var plateQuad: [CGPoint] {
        if let keypoints, keypoints.count == 4 { return keypoints }
        return [CGPoint(x: rect.minX, y: rect.minY),
                CGPoint(x: rect.maxX, y: rect.minY),
                CGPoint(x: rect.maxX, y: rect.maxY),
                CGPoint(x: rect.minX, y: rect.maxY)]
    }

    /// Re-anchors the detection box on the pose keypoints: the box head and the keypoint
    /// head of the pose model can disagree (field screenshots show the regression box
    /// drifting a fraction of its own width off the plate while the corner quad sits
    /// exactly on it). Every consumer — the perspective corrector's box-vs-quad checks,
    /// the OCR fallback crop, the NMS/tracker matching and the on-screen rectangles —
    /// then works from ONE geometry: the quad the keypoints actually describe.
    ///
    /// The original regression box is kept when the keypoints are absent or not a
    /// plausible plate quad (non-convex ordering, degenerate area, non-plate aspect), so
    /// a broken keypoint read can never make the box worse than it already was.
    var alignedToKeypoints: PlateDetection {
        guard let quad = keypoints, quad.count == 4,
              quad.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else { return self }
        let xs = quad.map(\.x), ys = quad.map(\.y)
        let minX = max(0, (xs.min() ?? 0)), maxX = min(1, (xs.max() ?? 0))
        let minY = max(0, (ys.min() ?? 0)), maxY = min(1, (ys.max() ?? 0))
        guard minX < maxX, minY < maxY else { return self }
        let width = maxX - minX, height = maxY - minY
        // Plausibility gate (same spirit as `PerspectivePlateCorrector.validatedQuad`):
        // a plate is a wide, convex quadrilateral. Anything else — keep the box.
        guard width / max(height, 1e-6) >= 1.0, width / max(height, 1e-6) <= 14 else { return self }
        let a = quad[0], b = quad[1], c = quad[2], d = quad[3]
        let doubleArea = (a.x*b.y - b.x*a.y) + (b.x*c.y - c.x*b.y)
            + (c.x*d.y - d.x*c.y) + (d.x*a.y - a.x*d.y)
        guard abs(doubleArea) / 2 >= 1e-5 else { return self }
        let orientation = doubleArea > 0 ? CGFloat(1) : CGFloat(-1)
        let points = [a, b, c, d]
        for i in 0..<4 {
            let p0 = points[i], p1 = points[(i + 1) % 4], p2 = points[(i + 2) % 4]
            let cross = (p1.x - p0.x) * (p2.y - p1.y) - (p1.y - p0.y) * (p2.x - p1.x)
            guard cross * orientation > 0 else { return self }
        }
        return PlateDetection(rect: CGRect(x: minX, y: minY, width: width, height: height),
                              confidence: confidence, label: label,
                              keypoints: quad, keypointConfidences: keypointConfidences)
    }
}

public extension PlateResult {
    /// The plate outline as 4 normalized (top-left origin) points: the rectified
    /// detection's pose corners when available, otherwise the bounding-box corners.
    var plateQuad: [CGPoint] {
        if let corners, corners.count == 4 { return corners }
        return [CGPoint(x: boundingBox.minX, y: boundingBox.minY),
                CGPoint(x: boundingBox.maxX, y: boundingBox.minY),
                CGPoint(x: boundingBox.maxX, y: boundingBox.maxY),
                CGPoint(x: boundingBox.minX, y: boundingBox.maxY)]
    }
}
