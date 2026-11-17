/*
 License Plate OCR SDK — Adaptive plate image cleaner (pre-OCR)

 Abstract:
 Cleans the two geometrically-determined interference sources on a German plate
 image BEFORE it is handed to OCR. Parameters are calibrated on real detection
 crops (Scripts/plate_cleaner_calibration.py, 13 real-world samples):

   1. EU blue strip  — the member-state letter "D" under the stars otherwise lands
      inside the read text (`B D AB 123` …). Detected as a contiguous run of blue
      columns starting at the left edge; masked white when found.
   2. Circular seals — the inspection seal (Prüfplakette, often colored) and the
      registration seal (Stempelplakette, gray) sit in the gap between the city
      code and the identification number, and OCR reads them as O/B/8/D/E. Detected
      as compact round blobs; masked white when found.

 Both detections are ADAPTIVE: masking happens only when the feature is actually
 found. Plates without seals (new cars, front-only plates, blurry shots) and
 crops without the blue strip pass through untouched — a fixed crop would eat
 real characters. Calibrated result: 4/13 seals masked, 0 characters damaged,
 EU strip correct in every crop that has one.

 Calibration thresholds (mirrors the Python script; tune there first, then here):
   - blue hue 95–135 (OpenCV 0–180 scale) = 190–270°, saturation ≥ 50/255
   - column blue ratio ≥ 0.35, strip width ≤ 0.22 × plate width
   - colored seal: saturation ≥ 70/255, non-blue, bbox aspect 0.6–1.7, fill ≥ 0.5,
     diameter 0.12–0.95 × height, interior (0.6r disc) white fraction ≤ 0.45,
     center x ≥ 0.24 × width, center y ≤ 0.9 × height
   - blue seal: same as colored but hue 190–270° (the EU strip is rejected by the
     center-x and aspect gates, since it hugs the left edge as a tall narrow column).
     Added because the original non-blue mask could not see a genuine blue Prüfplakette.
     Both colored floors were relaxed from 0.30 to 0.12 after real samples showed a
     16%-height orange seal (two-row plate) and a 20%-height blue seal (small plate).
   - gray seal: dark connected component (adaptive threshold), aspect 0.55–1.5,
     fill ≥ 0.55, diameter 0.30–0.95 × height (kept strict: characters are also
     dark), PLUS isolation (the vertical strips at ±1.45r must be background — real
     characters always have neighbours) and center y in 0.18–0.86 × height.
     Max 2 seals per plate (stacked pair).
 */

import Foundation
import CoreGraphics
import os

public struct PlateImageCleaner: Sendable {

    // MARK: Calibrated parameters (see header)

    private let targetHeight = 128.0
    private let euMaxFraction = 0.22
    private let euHueRange = (190.0, 270.0)      // degrees
    private let euSaturationMin: UInt8 = 50
    private let euValueMin: UInt8 = 30
    private let euColumnRatio = 0.35
    // Gray seals keep the strict 0.30 lower bound (characters are also dark, so a
    // loose gray gate would eat real letters). Colored / blue seals are safer: text
    // strokes carry no saturation. Real samples showed a 20%-of-height blue seal on
    // a small single-row plate and a 16% orange seal on a two-row plate, both of
    // which the 0.30 floor silently dropped; hence the relaxed colored floor.
    private let badgeMinDiameterFraction = 0.30
    private let coloredBadgeMinDiameterFraction = 0.12
    private let blueBadgeMinDiameterFraction = 0.12
    private let badgeMaxDiameterFraction = 0.95
    private let badgeMinCenterXFraction = 0.24
    private let interiorWhiteMax = 0.45
    private let grayCenterYRange = (0.18, 0.86)
    private let isolationDarkMax = 0.15
    private let maxBadges = 2

    public init() {}

    // MARK: Entry point

    /// Returns the cleaned copy of `image`, or `image` itself when nothing was found.
    public func clean(_ image: CGImage) -> CGImage {
        guard let probe = PixelBuffer(image: image, targetHeight: targetHeight) else { return image }

        // Both detectors run on a 128 px tall copy — the scale they were calibrated at —
        // and the masks are then painted onto the image at its OWN resolution. Masking the
        // probe and returning it (the old behaviour) downscaled every close-up plate to
        // 128 px, throwing away the resolution the corrector had just upscaled to: a
        // 960x204 crop came back as 614x128 before its most detail-rich OCR pass.
        let euEdge = detectEUStripEdge(probe)
        let badges = detectSeals(probe, euStripRightEdge: euEdge)
        guard euEdge > 0 || !badges.isEmpty else { return image }

        // Convert probe pixels to fractions so they can be applied at any resolution.
        let probeW = Double(probe.width), probeH = Double(probe.height)
        let euFraction = Double(euEdge) / probeW
        let badgeFractions = badges.map { (x: $0.cx / probeW, y: $0.cy / probeH, r: $0.r / probeH) }

        // Natural size == probe size for small crops: mask in place, no second buffer.
        var canvas = (probe.width == image.width && probe.height == image.height)
            ? probe
            : (PixelBuffer(image: image) ?? probe)
        let w = Double(canvas.width), h = Double(canvas.height)

        if euFraction > 0 {
            // EU strip fix: paint only the BLUE pixels inside the strip columns,
            // never a full-height rectangle. On a two-row plate the EU band sits
            // above the first bottom-row character (e.g. the 'H' in DA/H 220H);
            // a full-height fillRect wiped that character. Black glyph strokes
            // carry no blue saturation, so a mask-based white-out spares them.
            canvas.whiteOutBlue(x1: Int((euFraction * w).rounded()),
                                hueRange: euHueRange,
                                satMin: euSaturationMin,
                                valMin: euValueMin)
        }
        for badge in badgeFractions {
            canvas.fillCircle(cx: badge.x * w, cy: badge.y * h,
                              radius: badge.r * h * 1.15, value: 255)
        }
        return canvas.makeImage() ?? image
    }

    // MARK: EU strip

    /// Right edge (exclusive) of the EU strip, 0 when none found. Mirrors the Python
    /// logic: a contiguous run of blue columns from the left edge within the width limit.
    private func detectEUStripEdge(_ px: PixelBuffer) -> Int {
        let limit = max(1, Int(Double(px.width) * euMaxFraction))
        var edge = 0
        var runBroken = false
        for x in 0..<limit {
            let ratio = px.blueColumnRatio(x,
                                           hueRange: euHueRange,
                                           satMin: euSaturationMin,
                                           valMin: euValueMin)
            if ratio >= euColumnRatio && !runBroken {
                edge = x + 1
            } else if x < 3 && ratio < euColumnRatio {
                break   // the very left edge is not a strip
            } else if ratio < euColumnRatio {
                runBroken = true
            }
        }
        return edge
    }

    // MARK: Seals

    private func detectSeals(_ px: PixelBuffer, euStripRightEdge: Int)
        -> [(cx: Double, cy: Double, r: Double)] {
        // Observation mirror: built alongside the gates, never read by them. The
        // original guards below are kept verbatim, so accepting/rejecting a candidate
        // is decided by exactly the same conditions as before this diagnostics work.
        struct Obs {
            let kind: Kind
            let cx: Double, cy: Double, r: Double
            let aspect: Double, fill: Double, diameter: Double
            var interiorWhite: Double? = nil
            var isolation: Double? = nil
            var failed: [PlateCleanerDiagnostics.Gate] = []
        }
        var observations: [Obs] = []
        var candidates: [(kind: Kind, cx: Double, cy: Double, r: Double, obs: Int)] = []
        let h = Double(px.height), w = Double(px.width)
        let grayMinDia = badgeMinDiameterFraction * h
        let coloredMinDia = coloredBadgeMinDiameterFraction * h
        let blueMinDia = blueBadgeMinDiameterFraction * h
        let maxDia = badgeMaxDiameterFraction * h

        // --- Colored seals (non-blue saturation; characters are grayscale) ---
        // Diameter floor relaxed to 0.12h: a real two-row plate's orange年检章 was
        // measured at 16% of plate height, which the old 0.30 floor dropped.
        let coloredMask = px.coloredMask(satMin: 70, valMin: 60, hueRange: euHueRange)
        for (bbox, area) in connectedComponents(coloredMask, width: px.width, height: px.height, minArea: 30) {
            let bw = Double(bbox.width), bh = Double(bbox.height)
            let aspect = bw / max(1, bh)
            let fill = Double(area) / max(1, bw * bh)
            let dia = max(bw, bh)
            let cx = Double(bbox.minX) + bw / 2, cy = Double(bbox.minY) + bh / 2
            var failed: [PlateCleanerDiagnostics.Gate] = []
            if aspect < 0.6 || aspect > 1.7 { failed.append(.aspect) }
            if fill < 0.5 { failed.append(.fill) }
            if dia < coloredMinDia || dia > maxDia { failed.append(.diameter) }
            observations.append(Obs(kind: .colored, cx: cx, cy: cy, r: dia / 2,
                                    aspect: aspect, fill: fill, diameter: dia, failed: failed))
            guard aspect >= 0.6, aspect <= 1.7 else { continue }
            guard fill >= 0.5 else { continue }
            guard coloredMinDia <= dia, dia <= maxDia else { continue }
            candidates.append((.colored, cx, cy, dia / 2, observations.count - 1))
        }

        // --- Blue seals (blue saturation; the EU strip is also blue but excluded) ---
        // `coloredMask` deliberately excludes blue so the EU strip is not mistaken
        // for a seal — which also made a genuine blue Prüfplakette invisible. This
        // dedicated channel recovers it. The EU strip is rejected by two gates:
        //   (a) it starts at the crop's left edge, so its center fails centerX;
        //   (b) it is a tall narrow column, so its aspect fails the round-shape gate.
        // Verified on a real small plate: blue seal cx=186/496, dia=26 (20% h),
        // accepted; EU strip cx=32, dia=108 rejected by aspect+centerX.
        let blueMask = px.coloredMaskBlue(satMin: 70, valMin: 60, hueRange: euHueRange)
        for (bbox, area) in connectedComponents(blueMask, width: px.width, height: px.height, minArea: 30) {
            let bw = Double(bbox.width), bh = Double(bbox.height)
            let aspect = bw / max(1, bh)
            let fill = Double(area) / max(1, bw * bh)
            let dia = max(bw, bh)
            let cx = Double(bbox.minX) + bw / 2, cy = Double(bbox.minY) + bh / 2
            var failed: [PlateCleanerDiagnostics.Gate] = []
            if aspect < 0.6 || aspect > 1.7 { failed.append(.aspect) }
            if fill < 0.5 { failed.append(.fill) }
            if dia < blueMinDia || dia > maxDia { failed.append(.diameter) }
            // Exclude anything overlapping the EU strip's column range (with margin).
            if cx < max(Double(euStripRightEdge) + bw / 2, badgeMinCenterXFraction * w) {
                failed.append(.centerX)
            }
            observations.append(Obs(kind: .blue, cx: cx, cy: cy, r: dia / 2,
                                    aspect: aspect, fill: fill, diameter: dia, failed: failed))
            guard aspect >= 0.6, aspect <= 1.7 else { continue }
            guard fill >= 0.5 else { continue }
            guard blueMinDia <= dia, dia <= maxDia else { continue }
            guard cx >= max(Double(euStripRightEdge) + bw / 2, badgeMinCenterXFraction * w) else { continue }
            candidates.append((.blue, cx, cy, dia / 2, observations.count - 1))
        }

        // --- Gray seals (compact dark blobs; strictly gated) ---
        let darkMask = px.darkMask(structuringRadius: 2)
        for (bbox, area) in connectedComponents(darkMask, width: px.width, height: px.height, minArea: 40) {
            let bw = Double(bbox.width), bh = Double(bbox.height)
            let aspect = bw / max(1, bh)
            let dia = max(bw, bh)
            let fill = Double(area) / max(1, bw * bh)
            let cx = Double(bbox.minX) + bw / 2, cy = Double(bbox.minY) + bh / 2
            var failed: [PlateCleanerDiagnostics.Gate] = []
            if aspect < 0.55 || aspect > 1.5 { failed.append(.aspect) }
            if dia < grayMinDia || dia > maxDia { failed.append(.diameter) }
            if fill < 0.55 { failed.append(.fill) }   // seals are solid, characters are strokes
            observations.append(Obs(kind: .gray, cx: cx, cy: cy, r: dia / 2,
                                    aspect: aspect, fill: fill, diameter: dia, failed: failed))
            guard aspect >= 0.55, aspect <= 1.5 else { continue }
            guard grayMinDia <= dia, dia <= maxDia else { continue }
            guard fill >= 0.55 else { continue }
            candidates.append((.gray, cx, cy, dia / 2, observations.count - 1))
        }

        // --- Gating ---
        var scored: [(iso: Double, white: Double, kind: Kind, cx: Double, cy: Double, r: Double)] = []
        for c in candidates {
            // Observation only: compute every signal that the old gates looked at,
            // including ones the old short-circuit `guard`s would have skipped. This
            // is why the summary can name the *first* gate that blocked a real seal.
            var obs = observations[c.obs]
            let white = px.interiorWhiteFraction(cx: c.cx, cy: c.cy, radius: c.r * 0.6)
            obs.interiorWhite = white
            if c.cx < badgeMinCenterXFraction * w { obs.failed.append(.centerX) }
            if white > interiorWhiteMax { obs.failed.append(.interiorWhite) }
            var iso = 0.0
            switch c.kind {
            case .colored, .blue:
                // Inherently safe: characters carry no saturation. Only geometry applies
                // (a seal cut off at the crop's top edge has its center near y = 0).
                if c.cy < 0 || c.cy > 0.9 * h { obs.failed.append(.centerY) }
            case .gray:
                if c.cy < grayCenterYRange.0 * h || c.cy > grayCenterYRange.1 * h {
                    obs.failed.append(.centerY)
                }
                iso = px.gapIsolation(cx: c.cx, cy: c.cy, r: c.r)
                obs.isolation = iso
                if iso > isolationDarkMax { obs.failed.append(.isolation) }
            }
            observations[c.obs] = obs

            // Original gating, unchanged:
            guard c.cx >= badgeMinCenterXFraction * w else { continue }
            guard white <= interiorWhiteMax else { continue }
            switch c.kind {
            case .colored, .blue:
                guard c.cy >= 0, c.cy <= 0.9 * h else { continue }
                scored.append((0, white, c.kind, c.cx, c.cy, c.r))
            case .gray:
                guard c.cy >= grayCenterYRange.0 * h, c.cy <= grayCenterYRange.1 * h else { continue }
                guard iso <= isolationDarkMax else { continue }
                scored.append((iso, white, c.kind, c.cx, c.cy, c.r))
            }
        }

        // Publish the observation mirror last; it cannot affect `scored`.
        PlateCleanerDiagnostics.record(
            probeW: px.width, probeH: px.height,
            observations.map { o in
                PlateCleanerDiagnostics.Candidate(
                    kind: o.kind == .gray ? "gray" : (o.kind == .blue ? "blue" : "colored"),
                    cx: o.cx, cy: o.cy, diameter: o.diameter,
                    aspect: o.aspect, fill: o.fill,
                    interiorWhite: o.interiorWhite, isolation: o.isolation,
                    failedGates: o.failed)
            })

        // Lowest isolation first (most badge-like); at most a stacked pair.
        scored.sort { ($0.iso, $0.white) < ($1.iso, $1.white) }
        return scored.prefix(maxBadges).map { ($0.cx, $0.cy, $0.r) }
    }

    private enum Kind { case colored, blue, gray }
}

// MARK: - Seal detection diagnostics (observation only)

/// Last-frame record of every seal candidate the cleaner considered, including the
/// candidates it rejected and the exact gate that rejected them. This is a READ-ONLY
/// mirror of `detectSeals`: recording a value here never changes whether that gate
/// passes. Its purpose is to answer, on a real device, why a visible seal was left
/// in the OCR image — without guessing which threshold to loosen.
///
/// All geometry is expressed as fractions of the 128 px probe (center / diameter),
/// so a small plate and a close-up can be compared directly.
public enum PlateCleanerDiagnostics {
    public enum Gate: String, Sendable {
        case minArea, aspect, fill, diameter, centerX, interiorWhite, centerY, isolation
    }

    public struct Candidate: Sendable {
        public let kind: String              // "colored" / "gray"
        public let cx: Double                // fraction of probe width
        public let cy: Double                // fraction of probe height
        public let diameter: Double          // fraction of probe height
        public let aspect: Double
        public let fill: Double
        public let interiorWhite: Double?    // nil when rejected before this gate ran
        public let isolation: Double?        // only computed for gray candidates
        public let failedGates: [Gate]       // empty == accepted and masked
        public var accepted: Bool { failedGates.isEmpty }
    }

    private struct State {
        var probeWidth = 0
        var probeHeight = 0
        var candidates: [Candidate] = []
    }

    /// `OSAllocatedUnfairLock` cannot be held across an `await`; every access here is
    /// synchronous, matching `PaddleOCRDiagnostics`.
    private final class Box: @unchecked Sendable {
        private let state = OSAllocatedUnfairLock(initialState: State())

        func record(probeW: Int, probeH: Int, _ candidates: [Candidate]) {
            state.withLock {
                $0.probeWidth = probeW
                $0.probeHeight = probeH
                $0.candidates = candidates
            }
        }

        func summary() -> String? {
            state.withLock { s -> String? in
                guard s.probeHeight > 0 else { return nil }
                let accepted = s.candidates.filter(\.accepted).count
                var line = "徽章 \(accepted)/\(s.candidates.count) 遮罩 @\(s.probeWidth)×\(s.probeHeight)"
                for (i, c) in s.candidates.prefix(4).enumerated() {
                    let verdict = c.accepted ? "✓" : c.failedGates.map(\.rawValue).joined(separator: "+")
                    line += String(format: " · #%d %@ %@ a%.2f f%.2f",
                                   i + 1, c.kind, verdict, c.aspect, c.fill)
                    if let white = c.interiorWhite {
                        line += String(format: " w%.2f", white)
                    }
                    if let iso = c.isolation {
                        line += String(format: " i%.2f", iso)
                    }
                }
                return line
            }
        }
    }

    private static let box = Box()

    public static func record(probeW: Int, probeH: Int, _ candidates: [Candidate]) {
        box.record(probeW: probeW, probeH: probeH, candidates)
    }

    /// `nil` before the first cleaning pass. Callers decide whether to display it.
    public static var summary: String? { box.summary() }
}

// MARK: - Pixel buffer

/// RGBA8 pixel access with the small image ops the cleaner needs.
/// Two instances are used per call: a 128 px tall probe (the scale every threshold was
/// calibrated at, and where the O(pixels) morphology/HSV scans run) and, unless the crop
/// is that small already, a native-resolution canvas the masks are painted onto.
struct PixelBuffer {
    let width: Int
    let height: Int
    private var data: [UInt8]     // RGBA, 4 bytes/pixel

    /// Scaled so the image is exactly `targetHeight` tall — up **and down**.
    ///
    /// The scale must be unconditional: every threshold in this file (minimum blob area,
    /// 5×5 morphology radius, seal diameter fractions, strip width limit) was calibrated on
    /// a 128 px tall crop. The corrector upscales plates to ≥128 px, so a close-up crop can
    /// be 400 px tall — running the detectors there changes the effective blob area by
    /// (400/128)² ≈ 10×, which is how seals start slipping through `minArea` while noise
    /// starts passing it. The old `max(1, …)` guard (never shrink) silently did exactly
    /// that. Working at 128 px is also cheaper: the O(pixels) HSV/morphology scans get
    /// ~10× fewer pixels, and the masks are painted onto the native-resolution canvas
    /// afterwards, so the OCR input keeps its full detail.
    init?(image: CGImage, targetHeight: CGFloat) {
        let sourceHeight = max(1, Double(image.height))
        let scale = Double(targetHeight) / sourceHeight
        self.init(image: image,
                  width: max(1, Int((Double(image.width) * scale).rounded())),
                  height: max(1, Int((sourceHeight * scale).rounded())))
    }

    /// Native resolution — used to paint masks without changing the OCR source's size.
    init?(image: CGImage) {
        self.init(image: image, width: image.width, height: image.height)
    }

    private init?(image: CGImage, width w: Int, height h: Int) {
        guard let ctx = CGContext(data: nil, width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: SDKGraphics.sRGB,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let base = ctx.data else { return nil }
        ctx.interpolationQuality = .medium
        // A CGContext's origin is bottom-left: a plain `draw` stores the image bottom-row
        // first in the buffer. This buffer is read top-down (y = 0 is the image top) by
        // every detector below, and its thresholds were calibrated in Python/OpenCV
        // (top-down numpy arrays) — so flip while drawing to make row 0 the image TOP and
        // mirror the calibration semantics exactly. `makeImage()` flips back on the way out.
        ctx.translateBy(x: 0, y: CGFloat(h))
        ctx.scaleBy(x: 1, y: -1)
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        self.width = w
        self.height = h
        self.data = Array(UnsafeBufferPointer(start: base.assumingMemoryBound(to: UInt8.self), count: w * h * 4))
    }

    func makeImage() -> CGImage? {
        // The buffer is top-row-first (see init); a CGImage's bitmap convention is
        // bottom-row-first, so render the buffer through a flipped context to get an
        // upright image — the inverse of the flip applied in init.
        data.withUnsafeBufferPointer { ptr -> CGImage? in
            guard let base = ptr.baseAddress,
                  let src = CGContext(data: UnsafeMutableRawPointer(mutating: base),
                                      width: width, height: height, bitsPerComponent: 8,
                                      bytesPerRow: width * 4,
                                      space: SDKGraphics.sRGB,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
                  let raw = src.makeImage(),
                  let dst = CGContext(data: nil, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: SDKGraphics.sRGB,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return nil }
            dst.translateBy(x: 0, y: CGFloat(height))
            dst.scaleBy(x: 1, y: -1)
            dst.draw(raw, in: CGRect(x: 0, y: 0, width: width, height: height))
            return dst.makeImage()
        }
    }

    private func hsv(at x: Int, _ y: Int) -> (h: Double, s: Double, v: Double, gray: Double) {
        let i = (y * width + x) * 4
        let r = Double(data[i]), g = Double(data[i + 1]), b = Double(data[i + 2])
        let mx = max(r, g, b), mn = min(r, g, b)
        let delta = mx - mn
        var hue = 0.0
        if delta > 0 {
            if mx == r { hue = 60 * ((g - b) / delta).truncatingRemainder(dividingBy: 6) }
            else if mx == g { hue = 60 * ((b - r) / delta + 2) }
            else { hue = 60 * ((r - g) / delta + 4) }
            if hue < 0 { hue += 360 }
        }
        return (hue, mx == 0 ? 0 : delta / mx * 255, mx, mx)
    }

    /// Fraction of blue pixels in column `x` (RGB HSV: hue in degrees, s/v 0-255).
    func blueColumnRatio(_ x: Int, hueRange: (Double, Double), satMin: UInt8, valMin: UInt8) -> Double {
        var hits = 0
        for y in 0..<height {
            let c = hsv(at: x, y)
            if c.h >= hueRange.0, c.h <= hueRange.1, c.s >= Double(satMin), c.v >= Double(valMin) {
                hits += 1
            }
        }
        return Double(hits) / Double(max(1, height))
    }

    /// Binary mask: saturated non-blue pixels (colored seals).
    func coloredMask(satMin: UInt8, valMin: UInt8, hueRange: (Double, Double)) -> [Bool] {
        var mask = [Bool](repeating: false, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let c = hsv(at: x, y)
                if c.s >= Double(satMin), c.v >= Double(valMin),
                   !(c.h >= hueRange.0 && c.h <= hueRange.1) {
                    mask[y * width + x] = true
                }
            }
        }
        return mask
    }

    /// Binary mask: saturated BLUE pixels — the counterpart of `coloredMask`.
    /// Needed because `coloredMask` excludes blue (to avoid the EU strip), which
    /// otherwise makes a genuine blue Prüfplakette invisible. Callers reject the
    /// EU strip separately via the centerX/aspect gates in `detectSeals`.
    func coloredMaskBlue(satMin: UInt8, valMin: UInt8, hueRange: (Double, Double)) -> [Bool] {
        var mask = [Bool](repeating: false, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let c = hsv(at: x, y)
                if c.s >= Double(satMin), c.v >= Double(valMin),
                   c.h >= hueRange.0, c.h <= hueRange.1 {
                    mask[y * width + x] = true
                }
            }
        }
        return mask
    }

    /// Binary mask of dark pixels, morphologically closed (5×5 ellipse ≈ radius 2).
    /// Threshold is adaptive (below 0.62 × median) so dark night shots still separate.
    func darkMask(structuringRadius: Int) -> [Bool] {
        var grays = [UInt8]()
        grays.reserveCapacity(width * height)
        for i in stride(from: 0, to: data.count, by: 4) {
            let r = Double(data[i]), g = Double(data[i + 1]), b = Double(data[i + 2])
            grays.append(UInt8(min(255, (r + g + b) / 3)))
        }
        let sorted = grays.sorted()
        let median = Double(sorted[sorted.count / 2])
        let threshold = max(60, median * 0.62)
        let mask = grays.map { Double($0) < threshold }
        return morphologicalClose(mask, radius: structuringRadius)
    }

    /// Morphological CLOSE (dilate then erode) — joins broken strokes into solid
    /// blobs, mirroring the Python calibration's `cv2.MORPH_CLOSE`.
    private func morphologicalClose(_ mask: [Bool], radius: Int) -> [Bool] {
        erode(dilate(mask, radius: radius), radius: radius)
    }

    private func dilate(_ mask: [Bool], radius: Int) -> [Bool] {
        var out = mask
        for y in 0..<height {
            for x in 0..<width where !mask[y * width + x] {
                var hit = false
                outer: for dy in -radius...radius {
                    let yy = y + dy
                    guard yy >= 0, yy < height else { continue }
                    for dx in -radius...radius where dx*dx + dy*dy <= radius*radius {
                        let xx = x + dx
                        if xx >= 0, xx < width, mask[yy * width + xx] { hit = true; break outer }
                    }
                }
                out[y * width + x] = hit
            }
        }
        return out
    }

    private func erode(_ mask: [Bool], radius: Int) -> [Bool] {
        var out = mask
        for y in 0..<height {
            for x in 0..<width where mask[y * width + x] {
                var hit = true
                outer: for dy in -radius...radius {
                    let yy = y + dy
                    guard yy >= 0, yy < height else { hit = false; break }
                    for dx in -radius...radius where dx*dx + dy*dy <= radius*radius {
                        let xx = x + dx
                        // Note: `!(a, b)` would negate a *tuple*, not chain the
                        // conditions — that is a compile error, not a style issue.
                        if !(xx >= 0 && xx < width) || !mask[yy * width + xx] { hit = false; break outer }
                    }
                }
                out[y * width + x] = hit
            }
        }
        return out
    }

    /// Fraction of near-white pixels inside the disc (0.6r): a character "O" has a white
    /// center, a seal is filled with pattern. Mirrors Python `circle_stats(white@0.6)`.
    func interiorWhiteFraction(cx: Double, cy: Double, radius: Double) -> Double {
        let r2 = radius * radius
        var white = 0, total = 0
        let y0 = max(0, Int(cy - radius)), y1 = min(height - 1, Int(cy + radius))
        let x0 = max(0, Int(cx - radius)), x1 = min(width - 1, Int(cx + radius))
        for y in y0...max(y0, y1) {
            for x in x0...max(x0, x1) {
                let dx = Double(x) - cx, dy = Double(y) - cy
                if dx * dx + dy * dy <= r2 {
                    total += 1
                    let i = (y * width + x) * 4
                    let v = (Double(data[i]) + Double(data[i + 1]) + Double(data[i + 2])) / 3
                    if v > 190 { white += 1 }
                }
            }
        }
        return total == 0 ? 1.0 : Double(white) / Double(total)
    }

    /// Isolation: dark-pixel fraction summed over the two vertical strips at ±1.45r.
    /// A seal sits alone in a blank gap; a character always has neighbours nearby.
    func gapIsolation(cx: Double, cy: Double, r: Double) -> Double {
        let halfW = max(2, Int(r * 0.25))
        var total = 0.0
        for sx in [cx - r * 1.45, cx + r * 1.45] {
            let x = Int(sx.rounded())
            guard x - halfW >= 0, x + halfW < width else { return 1.0 }   // out of frame
            var dark = 0
            for y in 0..<height {
                for xx in (x - halfW)...(x + halfW) {
                    let i = (y * width + xx) * 4
                    let v = (Double(data[i]) + Double(data[i + 1]) + Double(data[i + 2])) / 3
                    if v < 130 { dark += 1 }
                }
            }
            total += Double(dark) / Double(height * (2 * halfW + 1))
        }
        return total
    }

    /// Paints only blue pixels (plus a 1px halo for antialiased edges) white inside
    /// columns `0..<x1`. Unlike a full-height rect this cannot erase dark glyphs that
    /// happen to sit under the EU strip on a two-row plate.
    mutating func whiteOutBlue(x1: Int, hueRange: (Double, Double),
                               satMin: UInt8, valMin: UInt8) {
        let limit = max(0, min(x1, width))
        // First pass: mark blue pixels, dilate by one pixel so the strip's soft edge
        // is covered too. Glyph strokes are neutral/black and never match the blue gate.
        var mark = [Bool](repeating: false, count: width * height)
        for y in 0..<height {
            for x in 0..<limit {
                let c = hsv(at: x, y)
                if c.h >= hueRange.0, c.h <= hueRange.1,
                   c.s >= Double(satMin), c.v >= Double(valMin) {
                    mark[y * width + x] = true
                }
            }
        }
        for y in 0..<height {
            for x in 0..<limit {
                var hit = mark[y * width + x]
                if !hit {
                    search: for dy in -1...1 {
                        for dx in -1...1 {
                            let ny = y + dy, nx = x + dx
                            if ny >= 0, ny < height, nx >= 0, nx < limit,
                               mark[ny * width + nx] {
                                hit = true
                                break search
                            }
                        }
                    }
                }
                if hit {
                    let i = (y * width + x) * 4
                    data[i] = 255; data[i + 1] = 255; data[i + 2] = 255; data[i + 3] = 255
                }
            }
        }
    }

    mutating func fillRect(x0: Int, x1: Int, y0: Int, y1: Int, value: UInt8) {
        for y in y0..<min(y1, height) {
            for x in x0..<min(x1, width) {
                let i = (y * width + x) * 4
                data[i] = value; data[i + 1] = value; data[i + 2] = value; data[i + 3] = 255
            }
        }
    }

    mutating func fillCircle(cx: Double, cy: Double, radius: Double, value: UInt8) {
        let y0 = max(0, Int(cy - radius)), y1 = min(height - 1, Int(cy + radius))
        let x0 = max(0, Int(cx - radius)), x1 = min(width - 1, Int(cx + radius))
        let r2 = radius * radius
        for y in y0...max(y0, y1) {
            for x in x0...max(x0, x1) {
                let dx = Double(x) - cx, dy = Double(y) - cy
                if dx * dx + dy * dy <= r2 {
                    let i = (y * width + x) * 4
                    data[i] = value; data[i + 1] = value; data[i + 2] = value; data[i + 3] = 255
                }
            }
        }
    }
}

// MARK: - Connected components (8-neighbour BFS; images here are ~460×128)

func connectedComponents(_ mask: [Bool], width: Int, height: Int, minArea: Int)
        -> [(bbox: CGRect, area: Int)] {
    var visited = [Bool](repeating: false, count: width * height)
    var results: [(CGRect, Int)] = []
    var queue = [Int]()
    for start in 0..<(width * height) where mask[start] && !visited[start] {
        visited[start] = true
        queue.removeAll(keepingCapacity: true)
        queue.append(start)
        var minX = width, maxX = -1, minY = height, maxY = -1, area = 0
        var head = 0
        while head < queue.count {
            let idx = queue[head]; head += 1
            let x = idx % width, y = idx / width
            area += 1
            minX = min(minX, x); maxX = max(maxX, x)
            minY = min(minY, y); maxY = max(maxY, y)
            for dy in -1...1 {
                let yy = y + dy
                guard yy >= 0, yy < height else { continue }
                for dx in -1...1 {
                    let xx = x + dx
                    guard xx >= 0, xx < width else { continue }
                    let n = yy * width + xx
                    if mask[n] && !visited[n] {
                        visited[n] = true
                        queue.append(n)
                    }
                }
            }
        }
        guard area >= minArea else { continue }
        results.append((CGRect(x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1), area))
    }
    return results
}
