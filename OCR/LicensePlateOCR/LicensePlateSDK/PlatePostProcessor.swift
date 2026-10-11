/*
 License Plate OCR SDK
 
 Abstract:
 Post-processing pipeline. Given the raw Vision observations for one cropped plate
 region, it cleans the text, validates it against the German plate format, and returns
 the most plausible plate (or `nil` when nothing valid was read). Also de-duplicates
 results aggregated across video frames.
 */

import Foundation
import CoreGraphics
import CoreImage

public struct PlatePostProcessor: Sendable {
    public init() {}

    /// Builds a `PlateResult` from the observations of a single detected region.
    /// - Parameters:
    ///   - observations: Vision observations for the cropped plate region.
    ///   - region: The detection's normalized bounding box (used for overlay).
    ///   - imageSize: Original image size, kept for reference/overlay math.
    /// - Returns: A validated `PlateResult`, or `nil` if no plausible plate was found.
    public func process(observations: [PlateObservation],
                        region: CGRect,
                        imageSize: CGSize) -> PlateResult? {
        // Collect every candidate string Vision produced (primary + alternatives).
        var raw: [(text: String, confidence: Float)] = []
        for obs in observations {
            if !obs.string.isEmpty { raw.append((obs.string, obs.confidence)) }
            raw.append(contentsOf: obs.candidates)
        }

        // Canonicalize each, tracking validity.
        var scored: [(text: String, confidence: Float, valid: Bool)] = []
        for (text, conf) in raw where !text.isEmpty {
            if let canon = GermanPlateValidator.canonicalize(text) {
                scored.append((canon, conf, GermanPlateValidator.isValid(canon)))
            } else {
                scored.append((text, conf, false))
            }
        }
        guard !scored.isEmpty else { return nil }

        // Keep the highest-confidence occurrence of each canonical string.
        var best: [String: (confidence: Float, valid: Bool)] = [:]
        for item in scored {
            if let existing = best[item.text] {
                if item.confidence > existing.confidence {
                    best[item.text] = (item.confidence, item.valid)
                }
            } else {
                best[item.text] = (item.confidence, item.valid)
            }
        }

        // Prefer format-valid plates; among those, highest confidence.
        let validOnes = best.filter { $0.value.valid }
            .sorted { $0.value.confidence > $1.value.confidence }
        guard let chosen = validOnes.first else { return nil }

        let candidates = best.map { PlateCandidate(text: $0.key, confidence: $0.value.confidence) }
            .sorted { $0.confidence > $1.confidence }

        return PlateResult(text: chosen.key,
                           confidence: chosen.value.confidence,
                           boundingBox: region,
                           imageSize: imageSize,
                           candidates: candidates)
    }

    /// De-duplicates plates detected across multiple video frames, keeping the
    /// highest-confidence sighting of each distinct plate.
    public static func deduplicate(_ results: [PlateResult]) -> [PlateResult] {
        var best: [String: PlateResult] = [:]
        for r in results {
            if let existing = best[r.text] {
                if r.confidence > existing.confidence { best[r.text] = r }
            } else {
                best[r.text] = r
            }
        }
        return best.values.sorted { $0.confidence > $1.confidence }
    }
}

// MARK: - Image enhancement before OCR (reduces interference)

extension CGImage {
    /// Lightly enhances a cropped plate region for OCR: contrast boost + mild
    /// sharpening. This helps when the crop includes screws, borders or reflections.
    func enhancedForOCR() -> CGImage {
        let ciImage = CIImage(cgImage: self)
        guard let color = CIFilter(name: "CIColorControls") else { return self }
        color.setValue(ciImage, forKey: kCIInputImageKey)
        color.setValue(1.35, forKey: kCIInputContrastKey)
        color.setValue(0.05, forKey: kCIInputBrightnessKey)

        guard let sharpen = CIFilter(name: "CISharpenLuminance") else { return self }
        sharpen.setValue(color.outputImage, forKey: kCIInputImageKey)
        sharpen.setValue(0.6, forKey: kCIInputSharpnessKey)

        guard let output = sharpen.outputImage else { return self }
        let context = CIContext(options: [.useSoftwareRenderer: false])
        if let cg = context.createCGImage(output, from: output.extent) { return cg }
        return self
    }

    /// Crops a (small, padded) region from this image. `normalizedRect` uses a
    /// top-left origin with 0...1 coordinates.
    func cropRegion(_ normalizedRect: CGRect, padding: CGFloat = 0.05) -> CGImage {
        let w = CGFloat(width), h = CGFloat(height)
        var x = normalizedRect.origin.x * w - normalizedRect.width * w * padding
        var y = normalizedRect.origin.y * h - normalizedRect.height * h * padding
        var cw = normalizedRect.width * w * (1 + 2 * padding)
        var ch = normalizedRect.height * h * (1 + 2 * padding)
        x = max(0, x); y = max(0, y)
        cw = min(w - x, cw); ch = min(h - y, ch)
        let rect = CGRect(x: x, y: y, width: cw, height: ch)
        return cropping(to: rect) ?? self
    }
}
