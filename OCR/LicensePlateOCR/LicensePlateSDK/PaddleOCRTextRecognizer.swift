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
           → compact [T,42] decoder view + full-dictionary greedy diagnostic.

 Why a constrained beam: Vision exposes `topCandidates(10)`, and
 `PlatePostProcessor` / `GermanPlateValidator` use those alternatives for
 canonicalization and lexicon rescoring. PaddleOCR emits a single CTC matrix, so
 we regenerate the alternatives ourselves. `PlateOCRAlphabet` is a DECODER
 CONSTRAINT, not a dictionary — the 18,383-line dictionary is bound to the
 model's fixed output head and must never be truncated.

 Performance notes (measured on PC with the reference script):
   decoding used to cost ~10.2 ms, as much as the 9.4 ms ORT inference, because
   three separate passes scanned all 18,385 classes and copied a ~2.9 MB tensor.
   `decode` now scans the full matrix ONCE and keeps only 42 columns per step.

 RecognitionMode mapping (PaddleOCR has no `.fast/.accurate` switch):
   .fast     — one inference, beam width 8
   .accurate — three crops (full / border-trimmed / 0.92x horizontally squeezed),
               beam width 16, plus a two-line plate pass when a horizontal ink
               band is detected. Measured ~4.7x the cost of `.fast`; keep it off
               the live-camera path.
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
        /// ORT's C++ `Session::Run` is thread-safe, but the ObjC wrapper makes no
        /// Sendable guarantee and `recognizeImage` calls this concurrently for
        /// every detection in a multi-plate frame. Serialize the run itself.
        let lock: NSLock

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

            var allowed = [0]
            for index in alphabet.indices where PlateOCRAlphabet.allowedSet.contains(alphabet[index]) {
                allowed.append(index)
            }
            self.allowed = allowed
            self.lock = NSLock()

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

    /// Loaded once per process, lazily and thread-safely. Switching OCR engines in
    /// the App used to rebuild this graph (16.5 MB + 18,383 dictionary lines + a
    /// new ORT session + warmup) on the main thread every time.
    ///
    /// `@unchecked Sendable` is required because `ORTSession` (ObjC) and the
    /// existential `Error` are not Sendable; both fields are immutable after init,
    /// and all mutable runtime state lives behind `Engine.lock` / the SDK's own
    /// `OSAllocatedUnfairLock`. A plain `Result<Engine, Error>` static let would
    /// fail Swift 6's concurrency-safety check.
    private final class SharedEngine: @unchecked Sendable {
        let engine: Engine?
        let failure: Error?

        init() {
            do {
                engine = try Engine()
                failure = nil
            } catch let error {
                engine = nil
                failure = error
            }
        }
    }

    private static let shared = SharedEngine()

    private let engine: Engine?
    private let loadError: Error?

    /// Non-throwing so `OCREngine.makeRecognizer` keeps its signature. A failed
    /// load (missing model/dictionary, unsupported ORT build) is surfaced as a
    /// thrown error on the first recognition instead of a silent Vision fallback.
    public init() {
        self.engine = Self.shared.engine
        self.loadError = Self.shared.failure
    }

    // MARK: - PlateTextRecognizing

    public func recognize(in image: CGImage, mode: RecognitionMode) async throws -> [PlateObservation] {
        guard let engine else {
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

        // text → best CTC path log-probability across all crops.
        var ranked: [String: Double] = [:]
        var greedyReadings: [String] = []
        var bestGreedyConfidence: Float = 0

        func record(_ text: String, score: Double) {
            guard !text.isEmpty else { return }
            ranked[text] = max(ranked[text] ?? -Double.infinity, score)
        }

        for variant in Self.variants(for: image, mode: mode) {
            try Task.checkCancellation()
            let prepared = try Self.prepare(variant)
            let output = try decode(prepared, engine: engine)
            let beams = Self.beamDecode(compact: output.compact, time: output.time,
                                        engine: engine, width: beamWidth, limit: candidateLimit)
            for beam in beams { record(beam.text, score: beam.logScore) }
            // The full-dictionary greedy reading is a CLEANING-QUALITY SIGNAL, not
            // a candidate. When it contains characters outside the plate alphabet
            // (e.g. '@' from a seal), the constrained beam would silently swallow
            // them and return a confident but wrong string. Keep it for diagnostics.
            if !output.greedyText.isEmpty { greedyReadings.append(output.greedyText) }
            bestGreedyConfidence = max(bestGreedyConfidence, output.greedyConfidence)
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
                let output = try decode(prepared, engine: engine)
                if !output.greedyText.isEmpty { pieces.append(output.greedyText) }
                greedyReadings.append(output.greedyText)
                bestGreedyConfidence = max(bestGreedyConfidence, output.greedyConfidence)
            }
            if pieces.count == 2 {
                // Keep this speculative reading behind every regular CTC path: a
                // false positive from the ink-band gate must never outrank a valid
                // one-line plate. It still reaches `GermanPlateValidator`.
                let floor = ranked.values.min() ?? -6
                record(pieces.joined(separator: " "), score: floor - 1)
            }
        }

        PaddleOCRDiagnostics.record(greedyReadings: greedyReadings)

        if ranked.isEmpty {
            // No legal CTC path survived (unusual plate format / heavy pruning).
            // Emit one character-filtered greedy reading so the validator still
            // gets something to judge instead of dropping the whole frame.
            let fallback = String((greedyReadings.first ?? "").filter {
                PlateOCRAlphabet.allowedSet.contains(String($0))
            })
            record(fallback, score: -60)
        }

        let ordered = ranked.sorted { $0.value > $1.value }
        guard let primary = ordered.first else { return [] }
        let topScore = primary.value
        // Absolute confidence anchor: the model's own greedy confidence. The vote
        // aggregator applies an absolute `provisionalConfidence` threshold, so a
        // relative-only score would make every reading look maximally confident.
        let primaryConfidence = min(1, max(0.01, bestGreedyConfidence))
        // Every alternative is now graded by its OWN CTC path score relative to the
        // winner, not by the source crop's greedy average (all previous candidates
        // shared that value). Scale back onto the primary's absolute confidence.
        let alternatives: [(text: String, confidence: Float)] = ordered.dropFirst().prefix(9).map {
            let relative = Float(exp(min(0, $0.value - topScore)))
            return (text: $0.key,
                    confidence: max(0.01, primaryConfidence * relative * 0.98))
        }

        // One observation spanning the whole rectified crop. `PlatePostProcessor`
        // then treats the region as owned by this plate and keeps the detection box.
        return [PlateObservation(boundingBox: CGRect(x: 0, y: 0, width: 1, height: 1),
                                 string: primary.key,
                                 confidence: primaryConfidence,
                                 candidates: alternatives)]
    }

    // MARK: - Inference

    private struct PreparedInput {
        let tensor: NSMutableData
        let width: Int
    }

    /// Only the columns the beam can emit, plus the full-dictionary greedy reading.
    /// Avoids materializing the `[T, 18385]` float array (~2.9 MB per inference).
    private struct DecoderOutput {
        let compact: [Float]
        let time: Int
        let greedyText: String
        let greedyConfidence: Float
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
        // CGContext(data:) does NOT clear the supplied buffer. Pre-fill it with
        // opaque 127.5 gray: perspective-corrected / cleaned crops can carry alpha,
        // and leaving transparent pixels at (0,0,0,0) normalizes to -1 — black
        // wedges at the plate corners. The right padding below stays 0 in float
        // space, which is exactly this same gray.
        var pixels = [UInt8](repeating: 0, count: imgH * bytesPerRow)
        for offset in stride(from: 0, to: pixels.count, by: 4) {
            pixels[offset] = 128
            pixels[offset + 1] = 128
            pixels[offset + 2] = 128
            pixels[offset + 3] = 255
        }
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

    /// Serializes just the ORT call; decoding happens outside the lock so unrelated
    /// detections do not contend for it.
    private func runSession(_ engine: Engine, _ input: ORTValue) throws -> ORTValue {
        engine.lock.lock()
        defer { engine.lock.unlock() }
        let outputs = try engine.session.run(withInputs: [engine.inputName: input],
                                             outputNames: Set([engine.outputName]),
                                             runOptions: nil)
        guard let value = outputs[engine.outputName] else {
            throw LicensePlateError.invalidModelOutput("PaddleOCR 未返回输出 \(engine.outputName)。")
        }
        return value
    }

    /// Runs the graph and reduces `[1,T,18385]` to `[T,42]` in a SINGLE full-matrix
    /// pass, while also building the full-dictionary greedy path. The old pipeline
    /// copied the whole tensor and then scanned it three times (normalize, greedy,
    /// beam); on PC that decoding cost ~10.2 ms, as much as the inference itself.
    private func decode(_ prepared: PreparedInput, engine: Engine) throws -> DecoderOutput {
        let shape = [1, 3, Engine.imageHeight, prepared.width].map { NSNumber(value: $0) }
        let input = try ORTValue(tensorData: prepared.tensor, elementType: .float, shape: shape)
        let value = try runSession(engine, input)

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
        let read: (Int) -> Float
        if raw.length == elementCount * MemoryLayout<Float>.size {
            let pointer = raw.bytes.assumingMemoryBound(to: Float.self)
            read = { pointer[$0] }
        } else if raw.length == elementCount * MemoryLayout<Float16>.size {
            let pointer = raw.bytes.assumingMemoryBound(to: UInt16.self)
            read = { Float(Float16(bitPattern: pointer[$0])) }
        } else {
            throw LicensePlateError.invalidModelOutput(
                "PaddleOCR 输出字节数 \(raw.length) 无法解释为 float32/float16 张量。")
        }

        let allowed = engine.allowed
        let width = allowed.count
        var compact = [Float](repeating: 0, count: time * width)
        var greedyIndices = [Int](repeating: 0, count: time)
        var greedyValues = [Float](repeating: 0, count: time)
        var rowSumTotal = Double(0)
        var hasNegative = false

        for t in 0..<time {
            let base = t * classes
            let out = t * width
            var best = 0
            var bestValue: Float = -1
            var rowSum: Float = 0
            for c in 0..<classes {
                let item = read(base + c)
                if item < 0 { hasNegative = true }
                rowSum += item
                if item > bestValue { bestValue = item; best = c }
            }
            rowSumTotal += Double(rowSum)
            greedyIndices[t] = best
            greedyValues[t] = bestValue
            for (j, c) in allowed.enumerated() { compact[out + j] = read(base + c) }
        }

        // PP-OCRv5 exports emit softmax. Use the PC reference's row-MEAN test
        // (Scripts/paddleocr_rec_reference.py), not the old per-row test. Only a
        // non-softmax export pays for the second pass.
        if hasNegative || abs(rowSumTotal / Double(time) - 1.0) > 0.05 {
            for t in 0..<time {
                let base = t * classes
                var maximum: Float = -.infinity
                for c in 0..<classes { maximum = max(maximum, read(base + c)) }
                var exponentSum: Float = 0
                for c in 0..<classes {
                    exponentSum += Float(exp(Double(read(base + c) - maximum)))
                }
                let out = t * width
                for (j, c) in allowed.enumerated() {
                    compact[out + j] = Float(exp(Double(read(base + c) - maximum))) / exponentSum
                }
            }
        }

        var greedyText = ""
        var confidenceSum: Float = 0
        var emitted = 0
        var previous = 0
        for t in 0..<time {
            let index = greedyIndices[t]
            if index != 0 && index != previous {
                greedyText += engine.alphabet[index]
                confidenceSum += greedyValues[t]
                emitted += 1
            }
            previous = index
        }

        return DecoderOutput(compact: compact, time: time, greedyText: greedyText,
                             greedyConfidence: emitted > 0 ? confidenceSum / Float(emitted) : 0)
    }

    // MARK: - CTC decoding

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

    /// Set to `false` to restore the unconstrained 42-class beam for A/B testing.
    private static let prefixPruningEnabled = true

    /// Conservative FZV-shaped prefix gate. Rejects only transitions that can
    /// never occur on a German plate; it must NOT reject a real reading. Umlauts
    /// are legal in the 1-3 letter city code but never in the identification
    /// number, and after the digit block only a terminal E/H suffix may appear.
    private static func isPlausiblePrefix(_ text: String) -> Bool {
        guard prefixPruningEnabled else { return true }
        let core = text.filter { $0 != "-" && $0 != " " }
        if core.isEmpty { return true }
        if core.count > 10 { return false }
        var sawDigit = false
        var letterRun = 0
        for (offset, character) in core.enumerated() {
            if character.isNumber {
                sawDigit = true
                letterRun = 0
            } else if character == "Ä" || character == "Ö" || character == "Ü" {
                if sawDigit { return false }
                letterRun += 1
            } else {
                letterRun += 1
                if sawDigit {
                    guard (character == "E" || character == "H"),
                          offset == core.count - 1 else { return false }
                }
            }
            if letterRun > 6 { return false }
        }
        return true
    }

    /// CTC prefix beam search over the German plate alphabet, in log space so a
    /// 40-step path never underflows. Mirrors the PC reference implementation.
    /// `compact` is `[T, allowed.count]` with the blank at column 0.
    private static func beamDecode(compact: [Float], time: Int, engine: Engine,
                                   width: Int, limit: Int)
        -> [(text: String, logScore: Double)] {
        let negInf = -Double.infinity
        var beams: [String: BeamState] = ["": BeamState(blankLog: 0, nonBlankLog: negInf)]
        let allowed = engine.allowed
        let columnCount = allowed.count
        let alphabet = engine.alphabet

        func merge(_ next: inout [String: BeamState], key: String,
                   blank: Double, nonBlank: Double) {
            var state = next[key] ?? BeamState(blankLog: negInf, nonBlankLog: negInf)
            state.blankLog = BeamState.logAdd(state.blankLog, blank)
            state.nonBlankLog = BeamState.logAdd(state.nonBlankLog, nonBlank)
            next[key] = state
        }

        for t in 0..<time {
            let base = t * columnCount
            var next: [String: BeamState] = [:]
            for (prefix, state) in beams {
                let total = BeamState.logAdd(state.blankLog, state.nonBlankLog)
                let blankProb = log(max(Double(compact[base]), 1e-12))
                merge(&next, key: prefix, blank: total + blankProb, nonBlank: negInf)

                let last = prefix.last.map { String($0) }
                for column in 1..<columnCount {
                    let character = alphabet[allowed[column]]
                    guard !character.isEmpty else { continue }
                    let probability = log(max(Double(compact[base + column]), 1e-12))
                    if last == character {
                        // Repeating a character requires an intervening blank; the
                        // non-blank path that stays on `prefix` is still valid.
                        merge(&next, key: prefix, blank: negInf,
                              nonBlank: state.nonBlankLog + probability)
                        let extended = prefix + character
                        guard isPlausiblePrefix(extended) else { continue }
                        merge(&next, key: extended, blank: negInf,
                              nonBlank: state.blankLog + probability)
                    } else {
                        let candidate = prefix + character
                        guard isPlausiblePrefix(candidate) else { continue }
                        merge(&next, key: candidate, blank: negInf,
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
        // Deliberately NO tail fallback: an unclosed low-ink run from some row all
        // the way to `high` is a washed-out single-line plate, not evidence of two
        // rows. The old fallback forced a split there and produced a bogus reading.
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
