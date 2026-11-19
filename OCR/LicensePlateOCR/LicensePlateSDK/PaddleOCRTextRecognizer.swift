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
              beam width 16. Measured ~4.7x the cost of `.fast`; keep it off
              the live-camera path.
  Both modes add a two-row plate pass (region code + identification row) when
  the horizontal ink-band gate proves the layout — CTC single-line decoding
  interleaves the rows and returns confident garbage there.
 */

import Foundation
import CoreGraphics
import ImageIO
import os
#if canImport(onnxruntime)
import onnxruntime
#elseif canImport(onnxruntime_mobile_objc)
import onnxruntime_mobile_objc
#elseif canImport(onnxruntime_objc)
import onnxruntime_objc
#endif

public final class PaddleOCRTextRecognizer: PlateTextRecognizing, @unchecked Sendable {

    // MARK: - Model / dictionary

    /// Immutable graph + dictionary. Each `session.run` stays on its engine's private
    /// serial queue, so the ObjC `ORTSession`'s missing concurrency annotation is harmless
    /// while different engines still decode in parallel.
    private struct Engine: @unchecked Sendable {
        let session: ORTSession
        let inputName: String
        let outputName: String
        /// CTC classes. Index 0 is blank, 1...18383 are the dictionary lines and
        /// 18384 is the appended space (`use_space_char=True` in imgocr).
        let alphabet: [String]
        /// Class indices the beam may emit: blank + German plate characters.
        let allowed: [Int]
        /// This session's PRIVATE serial queue. ORTSession is not annotated Sendable,
        /// so a single session is only ever run on this one queue. Multiple engines
        /// therefore decode different plates on different queues in parallel without
        /// ever calling one session from two threads.
        let queue: DispatchQueue

        static let imageHeight = 48
        static let baseWidth = 320

        init(queueIndex: Int) throws {
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
            self.queue = DispatchQueue(label: "de.plateocr.paddleocr.ort.\(queueIndex)",
                                       qos: .userInitiated)
            self.alphabet = alphabet

            var allowed = [0]
            for index in alphabet.indices where PlateOCRAlphabet.allowedSet.contains(alphabet[index]) {
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

    /// Number of independent ORT sessions kept for multi-plate OCR. Each session is
    /// pinned to its own private serial queue (see `Engine.queue`), so up to this many
    /// plates decode concurrently. A single session must never be run from two threads,
    /// which is why parallelism comes from MORE sessions rather than a concurrent queue.
    /// Two roughly doubles peak OCR memory (~16.5 MB graph + a dictionary copy each);
    /// still small next to the pose model, and it removes the serial OCR bottleneck that
    /// made multi-plate frames slow on older devices. Raise to 3-4 on stronger devices.
    private static let enginePoolSize = 2

    /// Dedicated queue for the (rare) lazy graph construction. Keeps the blocking
    /// init work off the cooperative pool. Each `Engine` then runs inference on its
    /// own queue, NOT here.
    private static let loadQueue = DispatchQueue(
        label: "de.plateocr.paddleocr.load", qos: .userInitiated)

    private enum LoadState {
        case idle
        case loaded([Engine])
        case failed(LicensePlateError)
    }

    /// Lazily builds a small POOL of engines once, then hands out a different engine
    /// to each caller (round-robin). Multi-plate frames issue one `recognize` per crop
    /// concurrently from `LicensePlateRecognizer`; with N engines their `session.run`
    /// calls no longer queue behind one another.
    ///
    /// Apple's "Reducing your app's launch time": UIKit draws the first frame on
    /// the main thread, so that thread must stay available, and expensive
    /// initialization belongs on first use. The pool is built once on `loadQueue`.
    private final class EngineStore: @unchecked Sendable {
        private let state = OSAllocatedUnfairLock(initialState: LoadState.idle)
        /// Monotonic ticket for round-robin engine selection; `Sendable` via the lock.
        private let nextIndex = OSAllocatedUnfairLock(initialState: 0)

        func get() async throws -> Engine {
            try await withCheckedThrowingContinuation { continuation in
                PaddleOCRTextRecognizer.loadQueue.async {
                    // Resolve the pool (or build it once) under the state lock, then
                    // pick the engine AFTER unlocking: avoids nesting the nextIndex
                    // lock inside the state lock and keeps the critical section short.
                    let resolved: Result<[Engine], LicensePlateError> = self.state.withLock { current in
                        switch current {
                        case .loaded(let engines):
                            return .success(engines)
                        case .failed(let error):
                            return .failure(error)
                        case .idle:
                            do {
                                var engines: [Engine] = []
                                engines.reserveCapacity(PaddleOCRTextRecognizer.enginePoolSize)
                                for index in 0..<PaddleOCRTextRecognizer.enginePoolSize {
                                    engines.append(try Engine(queueIndex: index))
                                }
                                current = .loaded(engines)
                                return .success(engines)
                            } catch {
                                let wrapped = LicensePlateError.paddleOCRUnavailable(
                                    (error as? LocalizedError)?.errorDescription
                                        ?? String(describing: error))
                                current = .failed(wrapped)
                                return .failure(wrapped)
                            }
                        }
                    }
                    let result: Result<Engine, LicensePlateError> = resolved.map { self.pick($0) }
                    // Resume explicitly: `CheckedContinuation` is parameterized on
                    // `any Error`, so a `Result<Engine, LicensePlateError>` cannot be
                    // passed to `resume(with:)` without an existential conversion.
                    switch result {
                    case .success(let engine): continuation.resume(returning: engine)
                    case .failure(let error): continuation.resume(throwing: error)
                    }
                }
            }
        }

        /// Picks the next engine by atomic ticket. Called under `loadQueue`, so no
        /// contention with graph construction; the lock is for Sendable correctness.
        private func pick(_ engines: [Engine]) -> Engine {
            let index = nextIndex.withLock { ticket -> Int in
                let chosen = ticket % engines.count
                ticket += 1
                return chosen
            }
            return engines[index]
        }
    }

    private static let store = EngineStore()

    /// Fire-and-forget warmup for the App to call AFTER the first frame is on screen.
    /// Swift static `let` is lazy, so simply constructing this class no longer loads
    /// anything. A failed load is cached and rethrown on the first recognition
    /// instead of silently downgrading to Apple Vision.
    public static func prewarm() async {
        _ = try? await store.get()
    }

    /// Allocation-only: `PaddleOCR.makeRecognizer` keeps its non-throwing signature
    /// and the App's `init` stays cheap.
    public init() {}

    // MARK: - PlateTextRecognizing

    public func recognize(in image: CGImage, mode: RecognitionMode) async throws -> [PlateObservation] {
        let engine = try await Self.store.get()
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

        // Classify the layout FIRST, then run only the matching OCR path.
        // Whole-crop OCR is now LAZY (see `needsWholeImage` below): it runs for a
        // single-row plate, or as a last-resort fallback when a CONFIRMED two-row
        // plate fails every split. It no longer runs unconditionally up front, so
        // on a two-row plate its interleaved garbage never enters the candidate
        // pool and no artificial score penalty ("-4") is needed to suppress it.

        // Two-row plates (zweizeilige Kennzeichen): CTC on the WHOLE crop interleaves
        // both rows and returns a plausible but wrong string (e.g. "TRD181" for
        // "TR / DL 81"). The horizontal ink projection proves the layout, then each
        // row is read on its own. The TOP K beams of both rows are combined into
        // candidate pairs and each pair is validated as a WHOLE plate, so a non-code
        // misread in one row's top beam cannot void the reading. The winner is
        // assembled straight into canonical `CITY-ID` form.
        var twoRowJoined: (text: String, confidence: Float)?
        // Read-only note for the debug panel: which split was used, what each row
        // read, and why a detected two-row layout may still have been rejected.
        var twoRowNote = "单行牌（未判定两行）"
        // Whether the whole-crop CTC pass must run. False on the common two-row
        // success path; flipped to true only for single-row plates or as a lazy
        // fallback after every two-row split failed.
        var needsWholeImage = false
        if let primarySplit = Self.horizontalInkSplit(of: image) {
            // Try the valley first. The ±step retries run ONLY when that split fails
            // to assemble, so the common case still costs two row decodes.
            let step = max(2, image.height / 40)
            var attempted = false
            for split in [primarySplit, primarySplit + step, primarySplit - step] {
                guard split > image.height / 4, split < image.height * 3 / 4,
                      let rawTop = image.cropping(to: CGRect(x: 0, y: 0,
                                                             width: CGFloat(image.width),
                                                             height: CGFloat(split))),
                      let rawBottom = image.cropping(to: CGRect(x: 0, y: CGFloat(split),
                                                                width: CGFloat(image.width),
                                                                height: CGFloat(image.height - split))),
                      let top = Self.paddedRow(rawTop),
                      let bottom = Self.paddedRow(rawBottom) else { continue }
                attempted = true
                try Task.checkCancellation()
                var rows: [RowReading] = []
                for piece in [top, bottom] {
                    try Task.checkCancellation()
                    let prepared = try Self.prepare(piece)
                    let output = try await Self.decode(prepared, engine: engine)
                    let beams = Self.beamDecode(compact: output.compact, time: output.time,
                                                engine: engine, width: beamWidth, limit: candidateLimit)
                    rows.append(RowReading(beams: beams, greedy: output.greedyText,
                                           confidence: output.greedyConfidence))
                    if !output.greedyText.isEmpty { greedyReadings.append(output.greedyText) }
                    bestGreedyConfidence = max(bestGreedyConfidence, output.greedyConfidence)
                }
                guard rows.count == 2 else { continue }
                let rowConfidence = min(rows[0].confidence, rows[1].confidence)
                let topText = rows[0].beams.first?.text ?? rows[0].greedy
                let bottomText = rows[1].beams.first?.text ?? rows[1].greedy
                // A confirmed two-row layout (the ink projection found two peaks and a
                // waist) is stronger evidence than a single row's CTC confidence. On
                // small/live camera crops one row often dips to 0.2–0.3 because the
                // glyphs are thin; gating at 0.3 discarded the whole two-row pass and
                // left only the interleaved whole-crop reading (e.g. "TRD181" for
                // TR/DL 81). 0.20 keeps genuinely unreadable rows out without throwing
                // away a correct pair. assembleTwoRows still validates the whole plate.
                guard rowConfidence >= 0.2 else {
                    twoRowNote = "split\(split) 行置信度\(String(format: "%.2f", Double(rowConfidence)))<0.20 top=\(topText) bottom=\(bottomText)"
                    continue
                }
                guard let reading = Self.assembleTwoRows(region: rows[0], identification: rows[1]) else {
                    twoRowNote = "split\(split) 结构校验失败 top=\(topText) bottom=\(bottomText)"
                    continue
                }
                // The two-row reading is recorded ALONE — no whole-crop text is in
                // the pool, so no artificial score penalty is needed to make it win.
                record(reading.text, score: reading.score)
                twoRowJoined = (reading.text, reading.confidence)
                twoRowNote = "split\(split) top=\(topText) bottom=\(bottomText) → \(reading.text)"
                break
            }
            if !attempted { twoRowNote = "split\(primarySplit) 无法裁剪" }

            if twoRowJoined == nil {
                // Every split failed on a plate we KNOW is two-row: fall back to the
                // whole-crop reading now. This is the ONLY case where the interleaved
                // reading is allowed into the pool — lazily, not unconditionally.
                twoRowNote += " → 两行组装失败，回退整图 OCR"
                needsWholeImage = true
            }
        } else {
            // Single-row plate: the whole-crop reading is the correct primary path.
            needsWholeImage = true
        }

        // WHOLE-CROP CTC PASS — inlined here (instead of a local async func that
        // captures and mutates the vars above) so Swift 6 strict-concurrency does
        // not flag "mutation of captured var in concurrently-executing code".
        if needsWholeImage {
            for variant in Self.variants(for: image, mode: mode) {
                try Task.checkCancellation()
                let prepared = try Self.prepare(variant)
                let output = try await Self.decode(prepared, engine: engine)
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
        }

        PaddleOCRDiagnostics.record(greedyReadings: greedyReadings, twoRow: twoRowNote)

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
        // Absolute confidence anchor. A joined two-row reading carries its own
        // honest per-row confidence; everything else anchors on the model's
        // greedy confidence as before.
        let primaryConfidence: Float
        if primary.key == twoRowJoined?.text, let joined = twoRowJoined {
            primaryConfidence = min(1, max(0.01, joined.confidence))
        } else {
            primaryConfidence = min(1, max(0.01, bestGreedyConfidence))
        }
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

    /// Diagnostics logging (visible via Console.app / `log collect --device`).
    enum PaddleOCRLog {
        static let tensor = Logger(subsystem: "de.plateocr", category: "paddleocr")
    }

    /// stderr diagnostic stream — captured by `devicectl device process launch
    /// --console`, which works without the root privileges `log collect --device`
    /// needs. Kept alongside the os.Logger entries so both channels stay in sync.
    private static func diag(_ line: String) {
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    private static let dumpLock = NSLock()
    private static var dumpCount = 0
    /// Cap per launch: an album recognition emits ~6 prepared inputs, live video
    /// emits them continuously — enough samples without filling the container.
    private static let dumpLimit = 60

    /// Writes prepared model inputs (and their source crops) to the app's
    /// Documents folder. Pull with `devicectl device copy from --domain-type
    /// appDataContainer` and replay through the PC reference script to compare
    /// device pixels against the byte-exact reference pipeline.
    private static func dumpPreparedIfNeeded(pixels: [UInt8], width: Int, height: Int,
                                             source: CGImage) {
        dumpLock.lock()
        let index = dumpCount
        if dumpCount < dumpLimit { dumpCount += 1 }
        dumpLock.unlock()
        guard index < dumpLimit else { return }

        func writePNG(_ image: CGImage, _ name: String) {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            guard let dir = docs else { return }
            let url = dir.appendingPathComponent(name)
            guard let dest = CGImageDestinationCreateWithURL(url as CFURL,
                                                             "public.png" as CFString, 1, nil)
            else { return }
            CGImageDestinationAddImage(dest, image, nil)
            CGImageDestinationFinalize(dest)
        }

        var buffer = pixels
        let bytesPerRow = width * 4
        if let context = CGContext(data: &buffer, width: width, height: height,
                                   bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                   space: SDKGraphics.sRGB,
                                   bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                       | CGBitmapInfo.byteOrder32Little.rawValue),
           let prepared = context.makeImage() {
            let tag = String(format: "%02d", index)
            writePNG(prepared, "ocr_dump_\(tag)_prepared_\(width)x\(height).png")
            writePNG(source, "ocr_dump_\(tag)_src_\(source.width)x\(source.height).png")
            diag("dump ocr_dump_\(tag) prepared=\(width)x\(height) src=\(source.width)x\(source.height)")
        }
    }

    /// Mean / std of the prepared tensor, for device-side sanity checks.
    private static func stat(floats: [Float]) -> (mean: Double, std: Double)? {
        guard !floats.isEmpty else { return nil }
        var sum = 0.0
        for f in floats { sum += Double(f) }
        let mean = sum / Double(floats.count)
        var acc = 0.0
        for f in floats { let d = Double(f) - mean; acc += d * d }
        let std = (acc / Double(floats.count)).squareRoot()
        return (mean, std)
    }

    /// Pure value type so it can cross the serial-queue boundary without an
    /// `ORTValue` / `NSMutableData` (both ObjC, neither `Sendable`).
    private struct PreparedInput: Sendable {
        let floats: [Float]
        let width: Int
    }

    /// Only the columns the beam can emit, plus the full-dictionary greedy reading.
    /// Avoids materializing the `[T, 18385]` float array (~2.9 MB per inference).
    private struct DecoderOutput: Sendable {
        let compact: [Float]
        let time: Int
        let greedyText: String
        let greedyConfidence: Float
        let log: DecodeLog?
    }

    /// Renders the rectified plate into PP-OCRv5's `[1,3,48,W]` float32 NCHW tensor.
    ///
    /// Two details are load-bearing:
    /// 1. Do NOT add a `translateBy`/`scaleBy(-1)` flip here. Commit 5994e1d removed it
    ///    after a Mac pixel-diff reported a plain `draw` closer to cv2.resize than the
    ///    flipped variant (37 vs 79), fixing mirrored readings (N→И, T→⊥, 255→522).
    ///    Caveat: Apple's docs fix the coordinate system of `CGContext(data:)` (Quartz
    ///    default, bottom-left) and the top-row-first storage of CGImage data
    ///    (`cropping(to:)`) — but neither, by itself, decides the row order of the
    ///    bitmap AFTER `draw`; the classic Quartz pitfall is the opposite convention.
    ///    Other SDK paths (`ONNXPose.legacyLetterbox`, `PlateImageCleaner`) do flip, so
    ///    this is an empirical convention, not a documented certainty. Re-verify on a
    ///    device with the `ocr_dump_*` prepared/source PNG pair before changing it.
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
                      space: SDKGraphics.sRGB,
                      bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                          | CGBitmapInfo.byteOrder32Little.rawValue) else { return false }
            // NO flip idiom here. For a `CGContext(data:)` bitmap the buffer's row 0
            // is the visual TOP row after a plain `draw` — verified on Mac by
            // pixel-diffing against cv2.resize of the same crop (plain draw matched
            // within interpolation tolerance; the translateBy/scaleBy(-1) idiom
            // produced an upside-down buffer, diff 79 vs 37). The idiom was the root
            // cause of every device-side recognition failure: the model was reading
            // vertically mirrored plates (N→И, T→⊥, 255→522).
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
        // Device-side diagnostic: input tensor statistics. A wrong channel order /
        // orientation / masking failure shows up here as an out-of-band mean or a
        // degraded std compared with the PC reference (~0.02 mean, ~0.75 std).
        if let stats = Self.stat(floats: floats) {
            PaddleOCRLog.tensor.info("prepare W=\(imgW) resizedW=\(resizedW) mean=\(stats.mean, format: .fixed(precision: 3)) std=\(stats.std, format: .fixed(precision: 3))")
            diag("prepare src=\(image.width)x\(image.height) W=\(imgW) resizedW=\(resizedW) mean=\(String(format: "%.3f", stats.mean)) std=\(String(format: "%.3f", stats.std))")
        }
        dumpPreparedIfNeeded(pixels: pixels, width: imgW, height: imgH, source: image)
        return PreparedInput(floats: floats, width: imgW)
    }

    /// Runs the graph and reduces `[1,T,18385]` to `[T,42]` in a SINGLE full-matrix
    /// pass, while also building the full-dictionary greedy path. The old pipeline
    /// copied the whole tensor and then scanned it three times (normalize, greedy,
    /// beam); on PC that decoding cost ~10.2 ms, as much as the inference itself.
    ///
    /// Both the blocking `Session::Run` and the ObjC tensor read happen on this
    /// engine's PRIVATE serial queue. Different engines use different queues, so
    /// multi-plate inference is parallel while each ORTSession stays single-threaded.
    /// Only the pure-Swift `DecoderOutput` crosses back, so the CTC beam search still
    /// runs concurrently for each plate in a multi-plate frame.
    private static func decode(_ prepared: PreparedInput, engine: Engine) async throws -> DecoderOutput {
        try await withCheckedThrowingContinuation { continuation in
            engine.queue.async {
                do {
                    let shape = [1, 3, Engine.imageHeight, prepared.width]
                        .map { NSNumber(value: $0) }
                    var floats = prepared.floats
                    let data = NSMutableData(bytes: &floats,
                                             length: floats.count * MemoryLayout<Float>.size)
                    let input = try ORTValue(tensorData: data, elementType: .float, shape: shape)
                    let outputs = try engine.session.run(
                        withInputs: [engine.inputName: input],
                        outputNames: Set([engine.outputName]), runOptions: nil)
                    guard let value = outputs[engine.outputName] else {
                        throw LicensePlateError.invalidModelOutput(
                            "PaddleOCR 未返回输出 \(engine.outputName)。")
                    }
                    let output = try Self.decodeTensor(value, engine: engine)
                    if let log = output.log {
                        PaddleOCRLog.tensor.info("decode T=\(log.time) C=\(log.classes) softmax=\(log.softmaxApplied) greedy=\"\(log.greedyText)\" conf=\(log.confidence, format: .fixed(precision: 3))")
                        diag("decode T=\(log.time) softmax=\(log.softmaxApplied) greedy=\"\(log.greedyText)\" conf=\(String(format: "%.3f", log.confidence))")
                    }
                    continuation.resume(returning: output)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Pure-CPU reduction of the ORT output; called on the engine's private queue.
    private static func decodeTensor(_ value: ORTValue, engine: Engine) throws -> DecoderOutput {
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
        let applySoftmax = hasNegative || abs(rowSumTotal / Double(time) - 1.0) > 0.05
        if applySoftmax {
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
                             greedyConfidence: emitted > 0 ? confidenceSum / Float(emitted) : 0,
                             log: .init(time: time, classes: classes,
                                        greedyText: greedyText,
                                        confidence: emitted > 0 ? confidenceSum / Float(emitted) : 0,
                                        softmaxApplied: hasNegative || abs(rowSumTotal / Double(time) - 1.0) > 0.05))
    }

    /// One-line diagnostic payload logged on the engine queue after every inference.
    struct DecodeLog: Sendable {
        let time: Int
        let classes: Int
        let greedyText: String
        let confidence: Float
        let softmaxApplied: Bool
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

    // MARK: - Two-row assembly (方案 B)

    /// One independently decoded row of a two-row plate.
    private struct RowReading {
        let beams: [(text: String, logScore: Double)]
        let greedy: String
        let confidence: Float
    }

    /// Adds symmetric white margins to a split row before it enters `prepare`.
    /// A region-code row with only 1–3 letters is a narrow strip; `prepare` maps its
    /// full width onto the 48 px-high tensor, so without padding its few glyphs are
    /// magnified and the CTC head loses the leading/trailing blank frames. Padding
    /// to roughly the aspect of a full identification row (~4:1) keeps glyph scale
    /// consistent with the model's training distribution. Rows already wide enough
    /// are returned untouched.
    private static func paddedRow(_ image: CGImage) -> CGImage? {
        let targetWidth = image.height * 4
        guard image.width < targetWidth else { return image }
        let pad = (targetWidth - image.width) / 2
        let width = image.width + 2 * pad
        guard let context = CGContext(data: nil, width: width, height: image.height,
                                      bitsPerComponent: 8, bytesPerRow: 0,
                                      space: SDKGraphics.sRGB,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                          | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        // White canvas: PaddleOCR normalizes white to +1, which is its blank/space
        // territory and never decodes as a plate glyph.
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: image.height))
        context.draw(image, in: CGRect(x: pad, y: 0,
                                       width: image.width, height: image.height))
        return context.makeImage()
    }

    /// Letter-only region-code candidates from the top row's beam, deduplicated and
    /// kept in CTC order. Each entry is (code, logScore). The greedy reading is added
    /// as a last-resort candidate when no beam produced a letter-only code.
    private static func regionCandidates(_ row: RowReading) -> [(text: String, score: Double)] {
        var out: [(text: String, score: Double)] = []
        var seen = Set<String>()
        for beam in row.beams.prefix(8) {
            let clean = beam.text.filter { $0.isLetter }
            guard (1...3).contains(clean.count), seen.insert(clean).inserted else { continue }
            out.append((clean, beam.logScore))
        }
        if out.isEmpty {
            let clean = row.greedy.filter { $0.isLetter }
            if (1...3).contains(clean.count) { out.append((clean, -60)) }
        }
        return out
    }

    /// Structurally legal identification candidates from the bottom row's beam, in
    /// CTC order, each normalized to canonical `SERIES DIGITS[SUFFIX]` form.
    private static func identificationCandidates(_ row: RowReading) -> [(text: String, score: Double)] {
        var out: [(text: String, score: Double)] = []
        var seen = Set<String>()
        for beam in row.beams.prefix(8) {
            let clean = beam.text.filter { PlateOCRAlphabet.allowedSet.contains(String($0)) }
            guard let body = canonicalBody(clean), seen.insert(body).inserted else { continue }
            out.append((body, beam.logScore))
        }
        return out
    }

    /// Validates an identification-row reading on its own and normalizes its shape.
    private static func canonicalBody(_ raw: String) -> String? {
        let clean = raw.uppercased().filter { $0 != " " && $0 != "-" }
        let pattern = "^[A-Z]{1,2}[0-9]{1,4}[EH]?$"
        guard clean.range(of: pattern, options: .regularExpression) != nil else { return nil }
        // Split at the first digit so the validator receives the canonical spacing,
        // e.g. `RY75H` → `RY 75H`. `isValid` also enforces the 8-char FZV limit.
        let cut = clean.firstIndex(where: { $0.isNumber })!
        let body = "\(clean[..<cut]) \(clean[cut...])"
        guard GermanPlateValidator.isValid("X-\(body)") else { return nil }
        return body
    }

    /// Picks the best full-plate reading from the two rows' candidate lists.
    ///
    /// The old code took each row's top-1 independently: when the city row's top beam
    /// was a non-code misread (`K8` for `KR`) or the bottom row's top beam was not a
    /// legal identification, the joined string failed validation and the entire
    /// two-row reading was discarded — falling back to the interleaved whole-crop
    /// garbage. Keeping the top-k of BOTH rows and validating each candidate pair as
    /// a WHOLE plate lets a lower-ranked but legal beam win, which is the common
    /// recoverable case.
    ///
    ///   score = mean(CTC top, CTC bottom) + lexicon bonus(city) − drift penalty
    /// Exact codes earn +4, edit-distance-1 codes +1.5, unknown codes 0 (the validator
    /// still accepts them — hint, never a hard filter). `canonicalize` returning the
    /// same text means the structure agrees, no penalty; a different text is penalized.
    ///
    /// Limit by design: if two candidates are BOTH real codes (`KK` vs `KR`) and the
    /// CTC misread ranks the wrong one higher, the bottom row carries no evidence to
    /// overturn it. Fixing that class requires the recognition model or real samples;
    /// this stage deliberately does not guess against the model's confidence.
    /// TEMP-EXPERIMENT [词表关闭]: when `false` (current), candidate pairs rank by
    /// PURE CTC scores — the lexicon bonus below is bypassed. Set `true` to restore.
    private static let lexiconBonusEnabled = false

    private static func assembleTwoRows(region: RowReading,
                                        identification: RowReading)
        -> (text: String, score: Double, confidence: Float)? {
        let cities = regionCandidates(region)
        let bodies = identificationCandidates(identification)
        guard !cities.isEmpty, !bodies.isEmpty else { return nil }

        var best: (text: String, score: Double)?
        for city in cities.prefix(6) {
            let bonus: Double = lexiconBonusEnabled
                ? (GermanPlateLexicon.contains(city.text) ? 4
                   : (GermanPlateLexicon.fuzzyMatches(for: city.text).isEmpty ? 0 : 1.5))
                : 0
            for body in bodies.prefix(6) {
                let joined = "\(city.text)-\(body.text)"
                guard let canonical = GermanPlateValidator.canonicalize(joined),
                      GermanPlateValidator.isValid(canonical) else { continue }
                let drift = canonical == joined ? 0.0 : -1.5
                // Mean of the two rows (not the sum) so the two-row reading stays on
                // the same CTC scale as a whole-crop candidate — otherwise adding two
                // log-probabilities would always lose to a single row's score.
                let score = (city.score + body.score) / 2 + bonus + drift
                if best == nil || score > best!.score {
                    best = (canonical, score)
                }
            }
        }
        guard let best else { return nil }
        return (best.text, best.score,
                (region.confidence + identification.confidence) / 2)
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

    /// Detects a genuine two-row plate from its horizontal ink profile and returns
    /// the row to split at, or `nil` for a one-row plate.
    ///
    /// Method: find a text peak in each half of the plate, then split at the longest
    /// low-ink run between them. Two details are load-bearing:
    ///   • Near-full-width dark rows are the plate frame / bumper, not text. They used
    ///     to win the lower-peak search (row ink ≈ 1.0) and destroyed the valley; they
    ///     are now excluded from both the peak and valley searches.
    ///   • The valley is RELATIVE to the weaker peak (≤ 60%, capped at 0.30). Real
    ///     two-row plates keep 0.2–0.3 ink in the gap because of screws, a divider or
    ///     a seal between the rows; the old 0.045 absolute floor rejected them all.
    ///
    /// A compact one-row plate still fails the aspect gate, and a single text band has
    /// no deep mid-height valley between two peaks. The old horizontal-span guard was
    /// removed: a city seal on the top row makes both rows span almost the full plate
    /// width, so it rejected genuine two-row plates (e.g. DA / H 220H). A false split
    /// now falls back safely because `assembleTwoRows` validates the whole plate.
    private static func horizontalInkSplit(of image: CGImage) -> Int? {
        let w = image.width, h = image.height
        // Keep the original 2.6 aspect gate: a compact single-row plate (FZV
        // 255×130 mm ≈ 1.96) is already below 3.0, so widening the gate would only
        // send more single-row plates into the split detector.
        guard w > 16, h > 40, Double(w) / Double(h) < 2.6 else { return nil }
        let bytesPerRow = w * 4
        var pixels = [UInt8](repeating: 255, count: h * bytesPerRow)
        let rendered = pixels.withUnsafeMutableBytes { raw -> Bool in
            guard let base = raw.baseAddress,
                  let context = CGContext(
                      data: base, width: w, height: h,
                      bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                      space: SDKGraphics.sRGB,
                      bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                          | CGBitmapInfo.byteOrder32Little.rawValue) else { return false }
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
        // Near-full-width dark rows are the plate frame / bumper, not glyphs. On the
        // real DA / H 220H crop 14 bottom rows have ink ≈ 1.0; the old lower-peak
        // search picked one of them, leaving no inter-row valley at all. Zero those
        // rows so both peak searches see text only.
        let borderInkLimit = 0.90
        var textInk = ink
        for y in 0..<h where ink[y] >= borderInkLimit { textInk[y] = 0 }
        // NOTE: no flip idiom during the draw above — same rule as `prepare`:
        // a plain `draw` leaves buffer row 0 as the visual top, so `ink[y]` indexes
        // rows top-down and the returned split row is directly usable for
        // `image.cropping`. The old flip made the profile bottom-up and mirrored
        // the split row vertically (two-row plates came out halves-swapped).
        // Peaks are searched in the outer bands (10–42% / 58–90%), the valley only
        // between them: this keeps a character's own strokes from being mistaken
        // for the inter-row gap and leaves room for a divider around mid-height.
        let upperLo = max(1, Int(Double(h) * 0.10))
        let upperHi = min(h, max(upperLo + 1, Int(Double(h) * 0.42)))
        let lowerLo = min(h - 1, max(upperHi, Int(Double(h) * 0.58)))
        let lowerHi = min(h, max(lowerLo + 1, Int(Double(h) * 0.90)))
        guard upperLo < upperHi, lowerLo < lowerHi else { return nil }
        let upperPeak = (upperLo..<upperHi).max { textInk[$0] < textInk[$1] } ?? upperLo
        let lowerPeak = (lowerLo..<lowerHi).max { textInk[$0] < textInk[$1] } ?? lowerLo
        guard textInk[upperPeak] > 0.05, textInk[lowerPeak] > 0.05 else { return nil }

        // Find the longest low-ink run between the two peaks and split at its centre.
        // The floor is RELATIVE to the weaker text peak, not an absolute "blank row"
        // threshold: screws, a printed divider or the city seal keep 0.2–0.3 ink in
        // a genuine inter-row gap. The old 0.045 floor saw no valley on either real
        // two-row sample and silently skipped the whole two-row pass.
        let peakFloor = min(textInk[upperPeak], textInk[lowerPeak])
        let threshold = min(peakFloor * 0.60, 0.30)
        var bestRun: (start: Int, end: Int)?
        var runStart: Int?
        for y in (upperPeak + 1)..<lowerPeak {
            if textInk[y] < threshold {
                if runStart == nil { runStart = y }
            } else if let start = runStart {
                if bestRun == nil || (y - start) > (bestRun!.end - bestRun!.start) {
                    bestRun = (start: start, end: y)
                }
                runStart = nil
            }
        }
        // Tail run (gap continues down to the lower peak's first dark row).
        if let start = runStart, lowerPeak > start,
           bestRun == nil || (lowerPeak - start) > (bestRun!.end - bestRun!.start) {
            bestRun = (start: start, end: lowerPeak)
        }
        guard let run = bestRun else { return nil }
        let valley = (run.start + run.end) / 2
        // A real inter-row gap sits at the plate's waist. A single-row plate can show
        // a deep gap between two characters, but it is not tied to mid-height (the EU
        // strip masks are also off-centre). The waist window was 40–60%, but a live
        // camera crop of TR/DL 81 put the valley at 57% — one frame's perspective or a
        // city seal in the gap can push it past 60%, which silently killed the whole
        // two-row pass and left the interleaved whole-crop reading. Widen to 36–64%:
        // still far from the outer text bands (10–42% / 58–90%), so a single-row
        // inter-character gap near an edge is rejected, while a genuine waist is kept.
        guard valley >= Int(Double(h) * 0.36), valley <= Int(Double(h) * 0.64) else { return nil }

        // No horizontal-span guard here. A city seal on the top row (DA / H 220H)
        // makes both halves span almost the entire plate width, so the old ≥0.12
        // span difference rejected real two-row plates. Full-plate validation in
        // `assembleTwoRows` already makes a false split harmless.
        return valley
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
                                      space: SDKGraphics.sRGB,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                          | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        context.draw(self, in: CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
        return context.makeImage()
    }
}
#endif
