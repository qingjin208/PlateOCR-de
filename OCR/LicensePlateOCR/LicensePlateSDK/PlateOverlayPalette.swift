/*
 License Plate OCR SDK — Overlay styling

 Abstract:
 Gives every plate a stable pseudo-random solid colour so a caller can fill the plate
 area with a semi-transparent tint and outline it with a solid stroke of the same hue.

 The colour is derived from a deterministic FNV-1a hash of the plate's key (its
 canonical text, else a quantised position), which means:
   - the same plate keeps the same colour across video samples / live frames;
   - different plates get visually distinct colours;
   - the palette is stable across app launches (`hashValue` is not, so it is not used).
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
    /// Solid, high-contrast hues. Deliberately saturated so a 25–30 % fill stays
    /// readable on both the white German plate and a dark car body.
    public static let colors: [PlateOverlayColor] = [
        PlateOverlayColor(red: 0.95, green: 0.20, blue: 0.26),  // red
        PlateOverlayColor(red: 1.00, green: 0.52, blue: 0.06),  // orange
        PlateOverlayColor(red: 0.98, green: 0.78, blue: 0.10),  // amber
        PlateOverlayColor(red: 0.55, green: 0.84, blue: 0.12),  // lime
        PlateOverlayColor(red: 0.13, green: 0.78, blue: 0.40),  // green
        PlateOverlayColor(red: 0.04, green: 0.76, blue: 0.70),  // teal
        PlateOverlayColor(red: 0.09, green: 0.70, blue: 0.95),  // cyan
        PlateOverlayColor(red: 0.20, green: 0.45, blue: 0.98),  // blue
        PlateOverlayColor(red: 0.44, green: 0.36, blue: 0.96),  // indigo
        PlateOverlayColor(red: 0.66, green: 0.34, blue: 0.95),  // violet
        PlateOverlayColor(red: 0.90, green: 0.27, blue: 0.84),  // magenta
        PlateOverlayColor(red: 0.98, green: 0.36, blue: 0.60)   // pink
    ]

    /// Alpha of the solid-colour plate fill.
    public static let fillAlpha: Double = 0.28
    /// Stroke width of the matching solid outline.
    public static let strokeWidth: CGFloat = 2.5
    /// Stroke width used for the underlying detection box, drawn under the plate tint.
    public static let detectionStrokeWidth: CGFloat = 1.5

    /// Stable colour for an arbitrary key.
    public static func color(for key: String) -> PlateOverlayColor {
        guard !colors.isEmpty else { return PlateOverlayColor(red: 0.13, green: 0.78, blue: 0.40) }
        return colors[Int(fnv1a(key) % UInt64(colors.count))]
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
