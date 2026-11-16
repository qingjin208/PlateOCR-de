#if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc) || canImport(onnxruntime_objc)
/*
 License Plate OCR SDK — PaddleOCR text recognizer (ONNX Runtime)

 Abstract:
 Replaces Apple Vision as the character-reading stage. The detector / perspective
 corrector are deliberately untouched: this class receives an already rectified
 plate image and runs PaddleOCR's PP-OCRv5 mobile **recognition** head directly.
 No text detector (DB) is used — the pose keypoints already supply a tight,
 perspective-corrected text line, and running DB on a plate would split the EU
 strip, the round inspection seals and the character row into separate boxes.

 Pipeline (byte-for-byte aligned with `Scripts/paddleocr_rec_reference.py`):
   CGImage → resize to (resizedW, 48) → BGR NCHW float32 with (x/255 - 0.5)/0.5
           → right-pad to W with 0 (PaddleOCR's np.zeros, == 127.5 gray in pixel
             space, NOT black) → ORT session [1,3,48,W] → [1,T,18385] CTC probs
           → greedy decode + character-constrained CTC prefix beam search.

 Why a constrained beam: Vision exposes `topCandidates(10)`, and
 `PlatePostProcessor` / `GermanPlateValidator` use those alternatives for
 canonicalization and lexicon rescoring. PaddleOCR emits a single CTC matrix, so
 we regenerate the alternatives ourselves. Restricting the beam to the German
 plate alphabet (A-Z, 0-9, ÄÖÜ, '-', space, blank) also suppresses the Chinese
 glyphs that dominate the 18,383-character PP-OCRv5 dictionary.

 RecognitionMode mapping (PaddleOCR has no `.fast/.accurate` switch):
   .fast     — one inference, beam width 8
   .accurate — three crops (full / border-trimmed / 0.92x horizontally squeezed),
               beam width 16, plus a two-line plate pass when a horizontal ink
               band is detected. All readings are merged into ONE observation.
 */

import Foundation
import CoreGraphics
#if canImport(onnxruntime)
import onnxruntime
#elseif canImport(onnxruntime_mobile_objc)
import onnxruntime_mobile_objc
#elseif canImport(onnxruntime_objc)
import onnxruntime_objc
#endif

public final class PaddleOCRTextRecognizer: PlateTextRecognizing, @unchecked Sendable {

    // MARK: - Model / dictionary

    private struct Engine {
        let session: ORTSession
        let inputName: String
        let outputName: String
        /// CTC classes. Index 0 is blank, 1...18383 are the dictionary lines and
        /// 18384 is the appended space (`use_space_char=True` in imgocr).
        let alphabet: [String]
        /// Class indices the beam may emit: blank + German plate characters.
        let allowed: [Int]

        static let imageHeight = 48
        static let baseWidth = 320

        init() throws {
            guard PaddleOCRResources.dictionaryURL != nil else {
                throw LicensePlateError.paddleOCRUnavailable("缺少 ppocrv5_dict.txt")
            }
            let alphabet = try Self.loadAlphabet()
            guard let modelURL = PaddleOCRResources.modelURL else {
                throw LicensePlateError.paddleOCRUnavailable("缺少 PP-OCRv5_rec_mobile_infer.onnx")
            }

            // Same ORT ObjC API shape as `ONNXPoseLicensePlateDetector` (verified
            // against onnxruntime-objc 1.18.0). The recognition graph is tiny, so
            // two intra-op threads keep the camera's performance cores free.
            let env = try ORTEnv(loggingLevel: .warning)
            let options = try ORTSessionOptions()
            _ = try? options.setIntraOpNumThreads(2)
            let session = try ORTSession(env: env, modelPath: modelURL.path,
                                         sessionOptions: options)

            self.session = session
            self.inputName = (try? session.inputNames())?.first ?? "x"
            self.outputName = (try? session.outputNames())?.first ?? "fetch_name_0"
            self.alphabet = alphabet

            let plateCharacters = Set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-ÄÖÜ ".map { String($0) })
            var allowed = [0]
            for index in alphabet.indices where plateCharacters.contains(alphabet[index]) {
                allowed.append(index)
            }
            self.allowed = allowed

            // Prewarm with the canonical [1,3,48,320] shape: ORT's graph
            // optimization and arena allocation happen on the first run.
            if let zeros = NSMutableData(length: 3 * Self.imageHeight * Self.baseWidth
                                            * MemoryLayout<Float>.size),
               let warmup = try? ORTValue(tensorData: zeros, elementType: .float,
                                          shape: [1, 3, Self.imageHeight, Self.baseWidth]
                                              .map { NSNumber(value: $0) }) {
                _ = try? session.run(withInputs: [inputName: warmup],
                                     outputNames: Set([outputName]), runOptions: nil)
            }
        }

        private static func loadAlphabet() throws -> [String] {
            guard let url = PaddleOCRResources.dictionaryURL else {
                throw LicensePlateError.paddleOCRUnavailable("字典未打包")
            }
            let raw = try String(contentsOf: url, encoding: .utf8)
            var lines = raw.split(separator: "\n", omittingEmptySubsequences: false).map { line -> String in
                var text = String(line)
                if text.hasSuffix("\r") { text.removeLast() }
                return text
            }
            // Trailing newline produces one empty element; Python's readlines() does not.
            if lines.last == "" { lines.removeLast() }
            lines.append(" ")                 // use_space_char=True
            return ["blank"] + lines          // CTC blank at index 0
        }
    }

    private let engine: Engine?
    private let loadError: Error?

    /// Non-throwing so `OCREngine.makeRecognizer` keeps its signature. A failed
    /// load (missing model/dictionary, unsupported ORT build) is surfaced as a
    /// thrown error on the first recognition instead of a silent Vision fallback.
    public init() {
        do {
            self.engine = try Engine()
            self.loadError = nil
        } catch {
            self.engine = nil
            self.loadError = error
        }
    }

    // MARK: - PlateTextRecognizing

    public func recognize(in image: CGImage, mode: RecognitionMode) async throws -> [PlateObservation] {
        guard let engine = engine else {
            let detail: String
            if let loadError = loadError {
                detail = (loadError as? LocalizedError)?.errorDescription
                    ?? String(describing: loadError)
            } else {
                detail = "未知初始化错误"
            }
            throw LicensePlateError.paddleOCRUnavailable(detail)
        }
        try Task.checkCancellation()

        let beamWidth = mode == .accurate ? 16 : 8
        let candidateLimit = mode == .accurate ? 12 : 6

        // text → (CTC path log-probability, greedy confidence of its source crop)
        var ranked: [String: (score: Double, confidence: Float)] = [:]

        func record(_ text: String, score: Double, confidence: Float) {
            guard !text.isEmpty else { return }
            if let existing = ranked[text] {
                ranked[text] = (max(score, existing.score),
                                max(confidence, existing.confidence))
            } else {
                ranked[text] = (score, confidence)
            }
        }

        for variant in Self.variants(for: image, mode: mode) {
            try Task.checkCancellation()
            let prepared = try Self.prepare(variant)
            var (probs, time, classes) = try run(prepared, engine: engine)
            Self.normalizeProbabilities(&probs, time: time, classes: classes)

            let greedy = Self.greedyDecode(probs: probs, time: time, classes: classes,
                                           alphabet: engine.alphabet)
            let beams = Self.beamDecode(probs: probs, time: time, classes: classes,
                                        engine: engine, width: beamWidth, limit: candidateLimit)
            let topScore = beams.first?.logScore ?? -60
            for beam in beams { record(beam.text, score: beam.logScore, confidence: greedy.confidence) }
            // Greedy is a useful fallback when the constrained beam prunes the true path.
            record(greedy.text, score: topScore - 2, confidence: greedy.confidence)
        }

        // Two-line German plates (zweizeilige Kennzeichen): only in accurate mode and
        // only when an actual low-ink horizontal band proves the plate is two-row.
        if mode == .accurate, let split = Self.horizontalInkSplit(of: image),
           let top = image.cropping(to: CGRect(x: 0, y: 0,
                                               width: CGFloat(image.width), height: CGFloat(split))),
           let bottom = image.cropping(to: CGRect(x: 0, y: CGFloat(split),
                                                  width: CGFloat(image.width),
                                                  height: CGFloat(image.height - split))) {
            var pieces: [String] = []
            for piece in [top, bottom] {
                try Task.checkCancellation()
                let prepared = try Self.prepare(piece)
                var (probs, time, classes) = try run(prepared, engine: engine)
                Self.normalizeProbabilities(&probs, time: time, classes: classes)
                let reading = Self.greedyDecode(probs: probs, time: time, classes: classes,
                                                alphabet: engine.alphabet)
                if !reading.text.isEmpty { pieces.append(reading.text) }
            }
            if pieces.count == 2 {
                // Keep this speculative reading behind every regular CTC path: a
                // false positive from the ink-band gate must never outrank a valid
                // one-line plate. It still reaches `GermanPlateValidator`, which can
                // accept it when the plate really is two-row.
                let floor = ranked.values.map { $0.score }.min() ?? -6
                record(pieces.joined(separator: " "), score: floor - 1, confidence: 0.7)
            }
        }

        let ordered = ranked.sorted { $0.value.score > $1.value.score }
        guard let primary = ordered.first else { return [] }
        let topScore = primary.value.score
        let primaryConfidence = primary.value.confidence
        let alternatives: [(text: String, confidence: Float)] = ordered.dropFirst().prefix(9).map {
            let relative = Float(exp(min(0, $0.value.score - topScore)))
            return (text: $0.key,
                    confidence: max(0.01, min(1, $0.value.confidence) * relative * 0.98))
        }

        // One observation spanning the whole rectified crop. `PlatePostProcessor`
        // then treats the region as owned by this plate and keeps the detection box.
        return [PlateObservation(boundingBox: CGRect(x: 0, y: 0, width: 1, height: 1),
                                 string: primary.key,
                                 confidence: min(1, max(0, primaryConfidence)),
                                 candidates: alternatives)]
    }

    // MARK: - Inference

    private struct PreparedInput {
        let tensor: NSMutableData
        let width: Int
    }

    /// Renders the rectified plate into PP-OCRv5's `[1,3,48,W]` float32 NCHW tensor.
    ///
    /// Two details are load-bearing (both verified against the PC reference):
    /// 1. `CGContext(data:)` is bottom-left origin, so the drawing must be flipped
    ///    or the model sees a vertically mirrored plate (the exact bug already
    ///    documented for the pose detector's letterbox).
    /// 2. Right padding is **0 in normalized space**, not black (-1). PaddleOCR's
    ///    `np.zeros` corresponds to a 127.5 gray pixel after inverting the
    ///    `(x/255 - 0.5)/0.5` transform.
    private static func prepare(_ image: CGImage) throws -> PreparedInput {
        let imgH = Engine.imageHeight
        guard image.height > 0, image.width > 0 else {
            throw LicensePlateError.invalidInput("矫正后的车牌图像尺寸退化。")
        }
        let ratio = Double(image.width) / Double(image.height)
        let maxWHRatio = max(Double(Engine.baseWidth) / Double(imgH), ratio)
        let imgW = max(1, Int(Double(imgH) * maxWHRatio))
        let resizedW = min(max(1, Int(ceil(Double(imgH) * ratio))), imgW)

        let bytesPerRow = imgW * 4
        var pixels = [UInt8](repeating: 0, count: imgH * bytesPerRow)
        let rendered = pixels.withUnsafeMutableBytes { raw -> Bool in
            guard let base = raw.baseAddress,
                  let context = CGContext(
                      data: base, width: imgW, height: imgH,
                      bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                      space: CGColorSpaceCreateDeviceRGB(),
                      bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                          | CGBitmapInfo.byteOrder32Little.rawValue) else { return false }
            context.translateBy(x: 0, y: CGFloat(imgH))
            context.scaleBy(x: 1, y: -1)
            context.draw(image, in: CGRect(x: 0, y: 0,
                                           width: CGFloat(resizedW), height: CGFloat(imgH)))
            return true
        }
        guard rendered else {
            throw LicensePlateError.invalidInput("无法渲染 PaddleOCR 输入缓冲。")
        }

        // BGRA little-endian in memory == the BGR channel order cv2.imread produces.
        let plane = imgH * imgW
        var floats = [Float](repeating: 0, count: 3 * plane)
        for y in 0..<imgH {
            let row = y * bytesPerRow
            for x in 0..<resizedW {
                let o = row + x * 4
                let i = y * imgW + x
                floats[i] = (Float(pixels[o]) / 255.0 - 0.5) / 0.5
                floats[plane + i] = (Float(pixels[o + 1]) / 255.0 - 0.5) / 0.5
                floats[2 * plane + i] = (Float(pixels[o + 2]) / 255.0 - 0.5) / 0.5
            }
            // x >= resizedW deliberately stays 0.
        }
        return PreparedInput(tensor: NSMutableData(bytes: &floats,
                                                    length: floats.count * MemoryLayout<Float>.size),
                             width: imgW)
    }

    private func run(_ prepared: PreparedInput, engine: Engine) throws
        -> (probs: [Float], time: Int, classes: Int) {
        let shape = [1, 3, Engine.imageHeight, prepared.width].map { NSNumber(value: $0) }
        let input = try ORTValue(tensorData: prepared.tensor, elementType: .float, shape: shape)
        let outputs = try engine.session.run(withInputs: [engine.inputName: input],
                                             outputNames: Set([engine.outputName]),
                                             runOptions: nil)
        guard let value = outputs[engine.outputName] else {
            throw LicensePlateError.invalidModelOutput("PaddleOCR 未返回输出 \(engine.outputName)。")
        }
        let dims = try value.tensorTypeAndShapeInfo().shape.map { $0.intValue }
        guard dims.count >= 2 else {
            throw LicensePlateError.invalidModelOutput("PaddleOCR 输出形状异常：\(dims)")
        }
        let classes = dims[dims.count - 1]
        let time = dims[dims.count - 2]
        guard classes == engine.alphabet.count, time > 0 else {
            throw LicensePlateError.invalidModelOutput(
                "PaddleOCR 输出 \(dims) 与字典 \(engine.alphabet.count) 类不一致。")
        }
        let raw = try value.tensorData()
        let elementCount = time * classes
        if raw.length == elementCount * MemoryLayout<Float>.size {
            let pointer = raw.bytes.assumingMemoryBound(to: Float.self)
            return (Array(UnsafeBufferPointer(start: pointer, count: elementCount)), time, classes)
        }
        if raw.length == elementCount * MemoryLayout<Float16>.size {
            let pointer = raw.bytes.assumingMemoryBound(to: UInt16.self)
            var floats = [Float](repeating: 0, count: elementCount)
            for i in 0..<elementCount { floats[i] = Float(Float16(bitPattern: pointer[i])) }
            return (floats, time, classes)
        }
        throw LicensePlateError.invalidModelOutput(
            "PaddleOCR 输出字节数 \(raw.length) 无法解释为 float32/float16 张量。")
    }

    /// PP-OCRv5 ONNX exports emit softmax probabilities, but tolerate logits.
    private static func normalizeProbabilities(_ probs: inout [Float], time: Int, classes: Int) {
        var needsSoftmax = false
        for t in 0..<time {
            let base = t * classes
            var sum: Float = 0
            var negative = false
            for c in 0..<classes {
                let value = probs[base + c]
                if value < 0 { negative = true; break }
                sum += value
            }
            if negative || abs(sum - 1) > 0.05 { needsSoftmax = true; break }
        }
        guard needsSoftmax else { return }
        for t in 0..<time {
            let base = t * classes
            var maximum: Float = -.infinity
            for c in 0..<classes { maximum = max(maximum, probs[base + c]) }
            var sum: Float = 0
            for c in 0..<classes {
                let e = Float(exp(Double(probs[base + c] - maximum)))
                probs[base + c] = e
                sum += e
            }
            for c in 0..<classes { probs[base + c] /= sum }
        }
    }

    // MARK: - CTC decoding

    private static func greedyDecode(probs: [Float], time: Int, classes: Int,
                                     alphabet: [String]) -> (text: String, confidence: Float) {
        var text = ""
        var sum: Float = 0
        var count = 0
        var previous = 0
        for t in 0..<time {
            let base = t * classes
            var best = 0
            var bestValue: Float = -1
            for c in 0..<classes {
                let value = probs[base + c]
                if value > bestValue { bestValue = value; best = c }
            }
            if best != 0 && best != previous {
                text += alphabet[best]
                sum += bestValue
                count += 1
            }
            previous = best
        }
        return (text, count > 0 ? sum / Float(count) : 0)
    }

    private struct BeamState {
        var blankLog: Double
        var nonBlankLog: Double
        var score: Double { BeamState.logAdd(blankLog, nonBlankLog) }

        static func logAdd(_ a: Double, _ b: Double) -> Double {
            if a == -.infinity { return b }
            if b == -.infinity { return a }
            let high = max(a, b)
            return high + log1p(exp(min(a, b) - high))
        }
    }

    /// CTC prefix beam search over the German plate alphabet, in log space so a
    /// 40-step path never underflows. Mirrors the PC reference implementation.
    private static func beamDecode(probs: [Float], time: Int, classes: Int, engine: Engine,
                                   width: Int, limit: Int)
        -> [(text: String, logScore: Double)] {
        let negInf = -Double.infinity
        var beams: [String: BeamState] = ["": BeamState(blankLog: 0, nonBlankLog: negInf)]

        func merge(_ next: inout [String: BeamState], key: String,
                   blank: Double, nonBlank: Double) {
            var state = next[key] ?? BeamState(blankLog: negInf, nonBlankLog: negInf)
            state.blankLog = BeamState.logAdd(state.blankLog, blank)
            state.nonBlankLog = BeamState.logAdd(state.nonBlankLog, nonBlank)
            next[key] = state
        }

        for t in 0..<time {
            let base = t * classes
            var next: [String: BeamState] = [:]
            for (prefix, state) in beams {
                let total = BeamState.logAdd(state.blankLog, state.nonBlankLog)
                let blankProb = log(max(Double(probs[base]), 1e-12))
                merge(&next, key: prefix, blank: total + blankProb, nonBlank: negInf)

                let last = prefix.last.map { String($0) }
                for c in engine.allowed where c != 0 {
                    let character = engine.alphabet[c]
                    guard !character.isEmpty else { continue }
                    let probability = log(max(Double(probs[base + c]), 1e-12))
                    if last == character {
                        // Repeating a character requires an intervening blank; the
                        // non-blank path that stays on `prefix` is still valid.
                        merge(&next, key: prefix, blank: negInf,
                              nonBlank: state.nonBlankLog + probability)
                        merge(&next, key: prefix + character, blank: negInf,
                              nonBlank: state.blankLog + probability)
                    } else {
                        merge(&next, key: prefix + character, blank: negInf,
                              nonBlank: total + probability)
                    }
                }
            }
            if next.count > width {
                let top = next.sorted { $0.value.score > $1.value.score }.prefix(width)
                beams = Dictionary(uniqueKeysWithValues: top.map { ($0.key, $0.value) })
            } else {
                beams = next
            }
        }
        return beams.filter { !$0.key.isEmpty }
            .sorted { $0.value.score > $1.value.score }
            .prefix(limit)
            .map { (text: $0.key, logScore: $0.value.score) }
    }

    // MARK: - Recognition variants

    private static func variants(for image: CGImage, mode: RecognitionMode) -> [CGImage] {
        guard mode == .accurate else { return [image] }
        var result: [CGImage] = [image]
        // Plate borders / screw heads frequently sit at the very top and bottom of
        // the rectified quad. Trimming them changes none of the glyph strokes.
        let inset = max(1, Int(Double(image.height) * 0.06))
        if image.height - 2 * inset >= 16,
           let trimmed = image.cropping(to: CGRect(x: 0, y: CGFloat(inset),
                                                   width: CGFloat(image.width),
                                                   height: CGFloat(image.height - 2 * inset))) {
            result.append(trimmed)
        }
        // A slightly squeezed line produces a different CTC alignment — cheap and
        // effective when the perspective correction over-stretched the plate.
        let squeezedWidth = max(16, Int(Double(image.width) * 0.92))
        if let squeezed = image.resized(toWidth: squeezedWidth) { result.append(squeezed) }
        return result
    }

    /// Detects a genuine two-row plate from its horizontal ink projection. Returns
    /// the row at which to split, or `nil` for the overwhelmingly common one-row
    /// plate. Gate: both halves must contain ink and the central band must be empty.
    private static func horizontalInkSplit(of image: CGImage) -> Int? {
        let w = image.width, h = image.height
        guard w > 16, h > 40, Double(w) / Double(h) < 2.6 else { return nil }
        let bytesPerRow = w * 4
        var pixels = [UInt8](repeating: 255, count: h * bytesPerRow)
        let rendered = pixels.withUnsafeMutableBytes { raw -> Bool in
            guard let base = raw.baseAddress,
                  let context = CGContext(
                      data: base, width: w, height: h,
                      bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                      space: CGColorSpaceCreateDeviceRGB(),
                      bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                          | CGBitmapInfo.byteOrder32Little.rawValue) else { return false }
            context.translateBy(x: 0, y: CGFloat(h))
            context.scaleBy(x: 1, y: -1)
            context.draw(image, in: CGRect(x: 0, y: 0, width: CGFloat(w), height: CGFloat(h)))
            return true
        }
        guard rendered else { return nil }

        var ink = [Double](repeating: 0, count: h)
        for y in 0..<h {
            var dark = 0
            let row = y * bytesPerRow
            for x in 0..<w {
                let o = row + x * 4
                if pixels[o] < 150, pixels[o + 1] < 150, pixels[o + 2] < 150 { dark += 1 }
            }
            ink[y] = Double(dark) / Double(w)
        }
        let low = Int(Double(h) * 0.30)
        let high = Int(Double(h) * 0.70)
        let upperInk = (0..<low).map { ink[$0] }.max() ?? 0
        let lowerInk = (high..<h).map { ink[$0] }.max() ?? 0
        guard upperInk > 0.06, lowerInk > 0.06 else { return nil }

        var bestBand: (start: Int, end: Int)?
        var runStart: Int?
        for y in low...high {
            if ink[y] < 0.03 {
                if runStart == nil { runStart = y }
            } else if let start = runStart {
                if y - start >= 3,
                   bestBand == nil || (y - start) > (bestBand!.end - bestBand!.start) {
                    bestBand = (start: start, end: y)
                }
                runStart = nil
            }
        }
        if let start = runStart, high - start >= 3, bestBand == nil {
            bestBand = (start: start, end: high)
        }
        guard let band = bestBand else { return nil }
        return (band.start + band.end) / 2
    }
}

private extension CGImage {
    /// Re-renders this image at a new width, preserving the visual orientation
    /// (no manual flip: `CGContext.makeImage()` bakes in the default transform).
    func resized(toWidth width: Int) -> CGImage? {
        let height = max(1, Int(round(Double(self.height) * Double(width) / Double(self.width))))
        guard width > 0,
              let context = CGContext(data: nil, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                          | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        context.draw(self, in: CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
        return context.makeImage()
    }
}
#endif
