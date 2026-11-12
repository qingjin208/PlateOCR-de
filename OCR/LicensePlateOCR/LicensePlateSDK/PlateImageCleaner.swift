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
     diameter 0.30–0.95 × height, interior (0.6r disc) white fraction ≤ 0.45,
     center x ≥ 0.24 × width, center y ≤ 0.9 × height
   - gray seal: dark connected component (adaptive threshold), aspect 0.55–1.5,
     fill ≥ 0.55, diameter 0.30–0.95 × height, PLUS isolation (the vertical strips
     at ±1.45r must be background — real characters always have neighbours) and
     center y in 0.18–0.86 × height. Max 2 seals per plate (stacked pair).
 */

import Foundation
import CoreGraphics

public struct PlateImageCleaner: Sendable {

    // MARK: Calibrated parameters (see header)

    private let targetHeight = 128.0
    private let euMaxFraction = 0.22
    private let euHueRange = (190.0, 270.0)      // degrees
    private let euSaturationMin: UInt8 = 50
    private let euValueMin: UInt8 = 30
    private let euColumnRatio = 0.35
    private let badgeMinDiameterFraction = 0.30
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
        guard let buf = PixelBuffer(image: image, targetHeight: targetHeight) else { return image }
        var pixels = buf

        // 1) EU strip
        let euEdge = detectEUStripEdge(pixels)
        if euEdge > 0 {
            pixels.fillRect(x0: 0, x1: euEdge, y0: 0, y1: pixels.height, value: 255)
        }

        // 2) Seals
        let badges = detectSeals(pixels)
        for (cx, cy, r) in badges {
            pixels.fillCircle(cx: cx, cy: cy, radius: r * 1.15, value: 255)
        }

        guard euEdge > 0 || !badges.isEmpty, let out = pixels.makeImage() else { return image }
        return out
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

    private func detectSeals(_ px: PixelBuffer) -> [(cx: Double, cy: Double, r: Double)] {
        var candidates: [(kind: Kind, cx: Double, cy: Double, r: Double)] = []
        let h = Double(px.height), w = Double(px.width)
        let minDia = badgeMinDiameterFraction * h
        let maxDia = badgeMaxDiameterFraction * h

        // --- Colored seals (saturation-based; characters are grayscale) ---
        let coloredMask = px.coloredMask(satMin: 70, valMin: 60, hueRange: euHueRange)
        for (bbox, area) in connectedComponents(coloredMask, width: px.width, height: px.height, minArea: 30) {
            let bw = Double(bbox.width), bh = Double(bbox.height)
            let aspect = bw / max(1, bh)
            guard aspect >= 0.6, aspect <= 1.7 else { continue }
            let fill = Double(area) / max(1, bw * bh)
            guard fill >= 0.5 else { continue }
            let dia = max(bw, bh)
            guard minDia <= dia, dia <= maxDia else { continue }
            let cx = Double(bbox.minX) + bw / 2, cy = Double(bbox.minY) + bh / 2
            candidates.append((.colored, cx, cy, dia / 2))
        }

        // --- Gray seals (compact dark blobs; strictly gated) ---
        let darkMask = px.darkMask(structuringRadius: 2)
        for (bbox, area) in connectedComponents(darkMask, width: px.width, height: px.height, minArea: 40) {
            let bw = Double(bbox.width), bh = Double(bbox.height)
            let aspect = bw / max(1, bh)
            guard aspect >= 0.55, aspect <= 1.5 else { continue }
            let dia = max(bw, bh)
            guard minDia <= dia, dia <= maxDia else { continue }
            let fill = Double(area) / max(1, bw * bh)
            guard fill >= 0.55 else { continue }   // seals are solid, characters are strokes
            let cx = Double(bbox.minX) + bw / 2, cy = Double(bbox.minY) + bh / 2
            candidates.append((.gray, cx, cy, dia / 2))
        }

        // --- Gating ---
        var scored: [(iso: Double, white: Double, kind: Kind, cx: Double, cy: Double, r: Double)] = []
        for c in candidates {
            guard c.cx >= badgeMinCenterXFraction * w else { continue }
            let white = px.interiorWhiteFraction(cx: c.cx, cy: c.cy, radius: c.r * 0.6)
            guard white <= interiorWhiteMax else { continue }
            switch c.kind {
            case .colored:
                // Inherently safe: characters carry no saturation. Only geometry applies
                // (a seal cut off at the crop's top edge has its center near y = 0).
                guard c.cy >= 0, c.cy <= 0.9 * h else { continue }
                scored.append((0, white, c.kind, c.cx, c.cy, c.r))
            case .gray:
                guard c.cy >= grayCenterYRange.0 * h, c.cy <= grayCenterYRange.1 * h else { continue }
                let iso = px.gapIsolation(cx: c.cx, cy: c.cy, r: c.r)
                guard iso <= isolationDarkMax else { continue }
                scored.append((iso, white, c.kind, c.cx, c.cy, c.r))
            }
        }
        // Lowest isolation first (most badge-like); at most a stacked pair.
        scored.sort { ($0.iso, $0.white) < ($1.iso, $1.white) }
        return scored.prefix(maxBadges).map { ($0.cx, $0.cy, $0.r) }
    }

    private enum Kind { case colored, gray }
}

// MARK: - Pixel buffer

/// RGBA8 pixel access with the small image ops the cleaner needs.
/// Works on a downscaled copy (height ≤ targetHeight) — the ops are O(pixels) and the
/// corrector already upscales, so cleaning runs on ~460×128 images in well under 1 ms.
struct PixelBuffer {
    let width: Int
    let height: Int
    private var data: [UInt8]     // RGBA, 4 bytes/pixel

    init?(image: CGImage, targetHeight: CGFloat) {
        let scale = max(1, Double(targetHeight) / Double(image.height))
        let w = max(1, Int((Double(image.width) * scale).rounded()))
        let h = max(1, Int((Double(image.height) * scale).rounded()))
        guard let ctx = CGContext(data: nil, width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let base = ctx.data else { return nil }
        ctx.interpolationQuality = .medium
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        self.width = w
        self.height = h
        self.data = Array(UnsafeBufferPointer(start: base.assumingMemoryBound(to: UInt8.self), count: w * h * 4))
    }

    func makeImage() -> CGImage? {
        data.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return nil }
            let ctx = CGContext(data: UnsafeMutableRawPointer(mutating: base),
                                width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            return ctx?.makeImage()
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
        var mask = grays.map { Double($0) < threshold }
        return morphologicalClose(mask, radius: structuringRadius)
    }

    private func morphologicalClose(_ mask: [Bool], radius: Int) -> [Bool] {
        dilate(erode(mask, radius: radius), radius: radius)
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
                        if !(xx >= 0, xx < width) || !mask[yy * width + xx] { hit = false; break outer }
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

    func fillRect(x0: Int, x1: Int, y0: Int, y1: Int, value: UInt8) {
        for y in y0..<min(y1, height) {
            for x in x0..<min(x1, width) {
                let i = (y * width + x) * 4
                data[i] = value; data[i + 1] = value; data[i + 2] = value; data[i + 3] = 255
            }
        }
    }

    func fillCircle(cx: Double, cy: Double, radius: Double, value: UInt8) {
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
