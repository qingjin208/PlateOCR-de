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

private let plateOCRContext = CIContext(options: [.cacheIntermediates: false])

public struct PlatePostProcessor: Sendable {
    public init() {}

    /// Builds a `PlateResult` from the observations of a single detected region.
    /// - Parameters:
    ///   - observations: Vision observations for the cropped plate region.
    ///   - region: The detection's normalized bounding box (used for overlay).
    ///   - imageSize: Original image size, kept for reference/overlay math.
    ///   - correctedImage: The rectified plate image fed to OCR, if available.
    /// - Returns: A validated `PlateResult`, or `nil` if no plausible plate was found.
    public func process(observations: [PlateObservation],
                        region: CGRect,
                        imageSize: CGSize,
                        correctedImage: CGImage? = nil) -> PlateResult? {
        // Collect every candidate string Vision produced (primary + alternatives).
        var raw: [(text: String, confidence: Float)] = []
        for obs in observations {
            if !obs.string.isEmpty { raw.append((obs.string, obs.confidence)) }
            raw.append(contentsOf: obs.candidates)
        }
        // Vision often splits a plate into city, series and number observations.
        // Join spatial neighbors instead of dropping a valid multi-part reading.
        let byRow = observations.filter { !$0.string.isEmpty }.sorted {
            if $0.boundingBox.midY != $1.boundingBox.midY { return $0.boundingBox.midY > $1.boundingBox.midY }
            return $0.boundingBox.minX < $1.boundingBox.minX
        }
        var rows: [[PlateObservation]] = []
        for observation in byRow {
            if let anchor = rows.last?.first, abs(anchor.boundingBox.midY - observation.boundingBox.midY) <= 0.25 {
                rows[rows.count - 1].append(observation)
            } else { rows.append([observation]) }
        }
        let ordered = rows.flatMap { $0.sorted { $0.boundingBox.minX < $1.boundingBox.minX } }
        var assembledText: String?
        if ordered.count > 1 {
            let joined = ordered.map(\.string).joined(separator: " ")
            assembledText = GermanPlateValidator.canonicalize(joined)
            raw.append((joined,
                        ordered.map(\.confidence).min() ?? 0))
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

        // A complete joined plate takes precedence over a fragment that happens to match the format.
        let validOnes = best.filter { $0.value.valid }
            .sorted {
                if ($0.key == assembledText) != ($1.key == assembledText) { return $0.key == assembledText }
                if $0.value.confidence != $1.value.confidence { return $0.value.confidence > $1.value.confidence }
                return $0.key < $1.key
            }
        guard let chosen = validOnes.first else { return nil }

        let candidates = best.map { PlateCandidate(text: $0.key, confidence: $0.value.confidence) }
            .sorted { $0.confidence > $1.confidence }

        return PlateResult(text: chosen.key,
                           confidence: chosen.value.confidence,
                           boundingBox: region,
                           imageSize: imageSize,
                           candidates: candidates,
                           correctedPlateImage: correctedImage)
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
        if let cg = plateOCRContext.createCGImage(output, from: output.extent) { return cg }
        return self
    }

    /// Crops a (small, padded) region from this image. `normalizedRect` uses a
    /// top-left origin with 0...1 coordinates.
    func cropRegion(_ normalizedRect: CGRect, padding: CGFloat = 0.05) -> CGImage {
        let w = CGFloat(width), h = CGFloat(height)
        let rect = CGRect(x: normalizedRect.minX * w, y: normalizedRect.minY * h,
                          width: normalizedRect.width * w, height: normalizedRect.height * h)
            .insetBy(dx: -normalizedRect.width * w * padding, dy: -normalizedRect.height * h * padding)
            .intersection(CGRect(x: 0, y: 0, width: w, height: h)).integral
        return cropping(to: rect) ?? self
    }
}
