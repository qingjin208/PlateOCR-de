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

    /// Builds one or more `PlateResult`s from the observations of a single detected region.
    ///
    /// A detection crop normally holds exactly one plate, so this typically returns a
    /// single result. But when two plates land in one crop (the detector occasionally
    /// merges adjacent plates, or the crop padding pulls in a neighbour), Vision may
    /// read both — this returns every *distinct* valid plate instead of collapsing them
    /// into one. Single-plate OCR fragments are still joined into one reading.
    /// - Parameters:
    ///   - observations: Vision observations for the cropped plate region.
    ///   - region: The detection's normalized bounding box (used for overlay / sub-boxing).
    ///   - imageSize: Original image size, kept for reference/overlay math.
    ///   - correctedImage: The rectified plate image fed to OCR, if available.
    ///   - corners: The detection's pose corners (normalized), attached to the plate that
    ///     owns the whole region so overlays can outline the actual plate quad.
    ///   - enhancements: Opt-in strengthening (default `.off` = legacy output). See
    ///     `RecognitionEnhancements`.
    /// - Returns: All validated plates found in the region (empty when none are plausible).
    public func process(observations: [PlateObservation],
                        region: CGRect,
                        imageSize: CGSize,
                        correctedImage: CGImage? = nil,
                        corners: [CGPoint]? = nil,
                        enhancements: RecognitionEnhancements = .off) -> [PlateResult] {
        guard !observations.isEmpty else { return [] }

        // (text, confidence, sourceBoxInCrop). `sourceBoxInCrop` is the observation's
        // normalized box (bottom-left origin) inside the OCR crop; nil means the whole crop.
        struct Cand { let text: String; let confidence: Float; let box: CGRect? }
        var raw: [Cand] = []
        for obs in observations {
            let box = obs.boundingBox
            if !obs.string.isEmpty { raw.append(Cand(text: obs.string, confidence: obs.confidence, box: box)) }
            for c in obs.candidates { raw.append(Cand(text: c.text, confidence: c.confidence, box: box)) }
        }

        // Vision often splits a plate into city, series and number observations.
        // Join spatial neighbours into one plausible plate (single plate, split across observations).
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
        var assembledConfidence: Float = 0
        if ordered.count > 1 {
            let joined = ordered.map(\.string).joined(separator: " ")
            assembledText = GermanPlateValidator.canonicalize(joined, enhancements: enhancements)
            assembledConfidence = ordered.map(\.confidence).min() ?? 0
        }

        // Canonicalize every candidate, keeping only format-valid plates.
        var scored: [(text: String, confidence: Float, box: CGRect?)] = []
        if let a = assembledText, GermanPlateValidator.isValid(a) {
            scored.append((a, assembledConfidence, nil))
        }
        for cand in raw where !cand.text.isEmpty {
            if let canon = GermanPlateValidator.canonicalize(cand.text, enhancements: enhancements),
               GermanPlateValidator.isValid(canon) {
                scored.append((canon, cand.confidence, cand.box))
            }
        }
        guard !scored.isEmpty else { return [] }

        // Keep the highest-confidence occurrence of each canonical string.
        var best: [String: (confidence: Float, box: CGRect?)] = [:]
        for item in scored {
            if let existing = best[item.text] {
                if item.confidence > existing.confidence { best[item.text] = (item.confidence, item.box) }
            } else {
                best[item.text] = (item.confidence, item.box)
            }
        }

        // Prefer the assembled complete plate, then emit other distinct valid plates
        // (two plates read from one crop) — but skip fragments that are substrings of a
        // plate we already emitted (e.g. "AB1234" inside "HH-AB 1234").
        let flatNorm: (String) -> String = {
            $0.uppercased().replacingOccurrences(of: " ", with: "").replacingOccurrences(of: "-", with: "")
        }
        var emitted: [(text: String, confidence: Float, box: CGRect?)] = []
        if let a = assembledText, let info = best[a] {
            emitted.append((a, info.confidence, info.box))
        }
        let remaining = best
            .filter { $0.key != assembledText }
            .sorted { $0.value.confidence > $1.value.confidence }
        for (text, info) in remaining {
            let n = flatNorm(text)
            let redundant = emitted.contains {
                let e = flatNorm($0.text)
                return e.contains(n) || n.contains(e)
            }
            if !redundant { emitted.append((text, info.confidence, info.box)) }
        }

        let chosen = Array(emitted.prefix(8))
        guard !chosen.isEmpty else { return [] }

        let candidates = best.map { PlateCandidate(text: $0.key, confidence: $0.value.confidence) }
            .sorted { $0.confidence > $1.confidence }

        return chosen.map { item in
            // Only the plate that fills the crop can inherit the region's box / corners;
            // a sub-plate carved out of a shared crop has its own sub-box.
            let ownsRegion = Self.ownsRegion(item.box)
            let box: CGRect
            if enhancements.effectiveKeypointQuads, ownsRegion, correctedImage != nil {
                // OCR ran on a perspective-corrected image, whose observation coordinates
                // have no linear relation to the frame — so keep the detection's own box
                // instead of mapping the observation box back (that mapping is only valid
                // when OCR ran on a plain box crop).
                box = region
            } else {
                box = item.box.map { Self.mapObservationBox($0, into: region) } ?? region
            }
            // Legacy: only an assembled reading (box == nil) carries the pose corners.
            // Enhanced: any plate that owns the crop does, which is what makes the tinted
            // outline follow the plate on a still image too.
            let quad: [CGPoint]?
            if let corners, enhancements.effectiveKeypointQuads, ownsRegion {
                quad = corners
            } else {
                quad = item.box == nil ? corners : nil
            }
            return PlateResult(text: item.text,
                               confidence: item.confidence,
                               boundingBox: box,
                               imageSize: imageSize,
                               candidates: candidates,
                               correctedPlateImage: correctedImage,
                               corners: quad)
        }
    }

    /// Whether an observation box fills (almost) the whole OCR crop. The area threshold
    /// leaves room for Vision's habit of reporting a slightly inset box around the text.
    private static func ownsRegion(_ box: CGRect?) -> Bool {
        guard let box else { return true }
        return box.width * box.height >= 0.6
    }

    /// Maps a Vision observation box (normalized, bottom-left origin, inside the OCR crop)
    /// into the detection's normalized region (top-left origin, original frame). This is a
    /// linear approximation used to give each plate its own sub-box when several share a crop.
    private static func mapObservationBox(_ obsBox: CGRect, into region: CGRect) -> CGRect {
        let x = region.minX + obsBox.minX * region.width
        let y = region.minY + (1 - obsBox.maxY) * region.height
        return CGRect(x: x, y: y,
                      width: obsBox.width * region.width,
                      height: obsBox.height * region.height)
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

    /// Collapses the same plate reported more than once *within one frame*, which happens
    /// when the detector emits two boxes over one plate (their IoU can stay below the NMS
    /// threshold) — previously that produced two identical rows and two tinted quads.
    ///
    /// Unlike `deduplicate` (text only, across video frames) this also requires the boxes
    /// to overlap, so two physically distinct plates that happen to read the same text are
    /// both kept.
    public static func deduplicateOverlapping(_ results: [PlateResult],
                                              minimumOverlap: CGFloat = 0.45) -> [PlateResult] {
        var kept: [PlateResult] = []
        kept.reserveCapacity(results.count)
        for candidate in results.sorted(by: { $0.confidence > $1.confidence }) {
            let isDuplicate = kept.contains { existing in
                guard existing.text == candidate.text else { return false }
                return overlapRatio(existing.boundingBox, candidate.boundingBox) >= minimumOverlap
            }
            if !isDuplicate { kept.append(candidate) }
        }
        // Restore the caller's ordering (detection order drives the overlay numbering).
        var order: [UUID: Int] = [:]
        for (index, result) in results.enumerated() { order[result.id] = index }
        return kept.sorted { (order[$0.id] ?? 0) < (order[$1.id] ?? 0) }
    }

    /// Intersection over the *smaller* box rather than IoU: a merged detection and the
    /// single plate inside it have IoU ≈ 0.3 while one box fully contains the other, so
    /// IoU would fail to recognise them as the same plate.
    private static func overlapRatio(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let intersection = a.intersection(b)
        guard !intersection.isNull, intersection.width > 0, intersection.height > 0 else { return 0 }
        let area = intersection.width * intersection.height
        let smaller = min(a.width * a.height, b.width * b.height)
        return smaller > 0 ? area / smaller : 0
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
