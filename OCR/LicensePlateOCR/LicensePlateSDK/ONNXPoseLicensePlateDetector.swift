#if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc) || canImport(onnxruntime_objc)
/*
 License Plate OCR SDK — ONNX Runtime pose detector

 Abstract:
 Runs a trained German-plate corner model through ONNX Runtime (Microsoft
 `onnxruntime-mobile-objc`) and decodes both the detection box and the 4 plate
 corners (left_top, right_top, right_bottom, left_bottom).

 Two export layouts are supported, auto-detected from the model's output names:
 - Ultralytics YOLOv8-pose (older exports, e.g. `De-plate-pose-500.onnx`): single
   output `[1, 17, 8400]` = 5 (cx, cy, w, h, conf) + 4 * 3 (x, y, visibility).
 - LibreYOLO YOLO-NAS-pose (`De-plate-pose.onnx`, the default): four
   outputs `boxes`/`scores`/`keypoints_xy`/`keypoints_conf`, 512 input,
   bottom-right letterbox, 127 gray, BGR — see `decodeLibreYOLO`.
 The decode / NMS / remap are shared with the Core ML detector via `YOLOOutputDecoder`,
 which is now keypoint-aware.

 Requires adding the ONNX Runtime framework to the target (see POSE_INTEGRATION.md).
 When the framework is absent this whole file is excluded by the `canImport` guard
 (matches `onnxruntime` or `onnxruntime_mobile_objc`, whichever CocoaPods exposes).
 */

import Foundation
import CoreGraphics
import CoreImage
import CoreML
import Accelerate
#if canImport(onnxruntime)
import onnxruntime
#elseif canImport(onnxruntime_mobile_objc)
import onnxruntime_mobile_objc
#elseif canImport(onnxruntime_objc)
import onnxruntime_objc
#endif

public final class ONNXPoseLicensePlateDetector: LicensePlateDetecting, @unchecked Sendable {

    private let session: ORTSession
    private let config: YOLOConfig
    private let inputName: String
    private let outputName: String
    private let inputSize: Int
    /// Export-layout auto-detection (at init, from the model's own output names):
    /// - `false` = Ultralytics YOLOv8-pose, single output `[1, 17, 8400]`.
    /// - `true` = LibreYOLO YOLO-NAS-pose (`De-plate-pose.onnx`), four
    ///   outputs (`boxes`/`scores`/`keypoints_xy`/`keypoints_conf`), 512×512 input,
    ///   **bottom-right** letterbox with **127** gray, **BGR** channel order. See
    ///   `YOLOConfig.germanPlatePoseLibre` and `Scripts/libreyolo_onnx_probe.py`.
    private let libreLayout: Bool
    private let libreOutputNames: [String]
    /// Channel order fed to the model. `true` = RGB, which is what Ultralytics trains on
    /// and what the proven PC reference (`Scripts/pose_pipeline_reference.py`) feeds:
    /// `img = np.array(Image.open(path).convert("RGB"))` → `/255` → NCHW. A mismatch here
    /// is the #1 cause of "great on PC, poor on device", so the default follows the
    /// reference rather than guesswork. Still runtime-settable via
    /// `LicensePlateRecognizer.useRGB` so a differently trained export can be handled
    /// without recompiling. Lock-guarded in `Tunables` (the UI may
    /// flip it while an inference pass is reading it).
    var useRGB: Bool {
        get { tunables.useRGB }
        set { tunables.useRGB = newValue }
    }
    /// Whether the letterbox buffer is written top-row-first (`true`), like every
    /// Ultralytics / OpenCV / PIL pipeline — including the PC reference
    /// (`canvas[pad_y:pad_y+nh] = resized`, row 0 = visual top).
    ///
    /// This is NOT cosmetic. A `CGContext` created with `CGContext(data:...)` uses the
    /// Quartz default coordinate system with its origin at the **bottom-left**, so drawing
    /// a `CGImage` into it without a flip transform writes the picture **upside down** into
    /// the buffer — the model then sees a vertically mirrored scene. The PC pipeline has
    /// no such flip, which is exactly why "the PC detects everything, the phone does not".
    /// The flip is kept runtime-settable so the two behaviours can be A/B-tested on device
    /// without rebuilding. With the vImage letterbox this is implemented by reversing the
    /// scaled rows after the resample (see `CGImage.letterboxedUpright`); the legacy
    /// CGContext fallback honours it as the original drawing transform. Lock-guarded in
    /// `Tunables` for the same reason as `useRGB`.
    var flipY: Bool {
        get { tunables.flipY }
        set { tunables.flipY = newValue }
    }

    /// Mutable runtime state in a lock-guarded box: a `Sendable` class may not have
    /// mutable stored properties, and every access must be synchronized because the UI
    /// thread may flip a switch while an inference task is mid-`detect` — the letterbox
    /// geometry (`flipY`) and channel order (`useRGB`) must not change between the two
    /// reads of one pass, and `topScore` is written by the inference task and read by the UI.
    private final class Tunables: @unchecked Sendable {
        private let lock = NSLock()
        private var _confidence: Float
        private var _iou: Float
        private var _captureSnapshot = false
        /// Highest raw anchor score of the most recent pass, BEFORE the confidence
        /// threshold. The one number that splits "the model saw nothing like a plate in
        /// this input" (≈0.01-0.1) from "the model sees the plate but decode/validation
        /// drops it" (≥0.25) on a device where the PC pipeline works — no snapshot
        /// comparison needed.
        private var _topScore: Float?
        private var _useRGB: Bool
        private var _flipY: Bool

        init(confidence: Float, iou: Float, useRGB: Bool, flipY: Bool) {
            _confidence = confidence
            _iou = iou
            _useRGB = useRGB
            _flipY = flipY
        }

        private func withLock<T>(_ body: () -> T) -> T {
            lock.lock()
            defer { lock.unlock() }
            return body()
        }

        var confidence: Float {
            get { withLock { _confidence } }
            set { withLock { _confidence = newValue } }
        }
        var iou: Float {
            get { withLock { _iou } }
            set { withLock { _iou = newValue } }
        }
        var captureSnapshot: Bool {
            get { withLock { _captureSnapshot } }
            set { withLock { _captureSnapshot = newValue } }
        }
        var topScore: Float? {
            get { withLock { _topScore } }
            set { withLock { _topScore = newValue } }
        }
        var useRGB: Bool {
            get { withLock { _useRGB } }
            set { withLock { _useRGB = newValue } }
        }
        var flipY: Bool {
            get { withLock { _flipY } }
            set { withLock { _flipY = newValue } }
        }
    }
    private let tunables: Tunables
    /// `CGImage` is not `Sendable`, so the snapshot lives in its own box too.
    private final class SnapshotBox: @unchecked Sendable { var image: CGImage? }
    private let snapshotBox = SnapshotBox()

    // MARK: - Runtime tunables / diagnostics

    public var confidenceThreshold: Float {
        get { tunables.confidence }
        set { tunables.confidence = newValue }
    }
    public var nmsIoUThreshold: Float {
        get { tunables.iou }
        set { tunables.iou = newValue }
    }
    public var debugSnapshot: CGImage? { snapshotBox.image }
    /// Highest raw anchor score of the last `detect` pass (threshold-independent). `nil`
    /// until the LibreYOLO path has run at least once.
    public var lastTopScore: Float? { tunables.topScore }
    public var captureDebugSnapshot: Bool {
        get { tunables.captureSnapshot }
        set { tunables.captureSnapshot = newValue }
    }

    /// The LibreYOLO export bakes BGR into its training preprocessing (see `feedsRGB`),
    /// so `useRGB` only has an effect on the Ultralytics layout.
    public var honorsChannelOrder: Bool { !libreLayout }

    /// The base config with the current runtime thresholds applied.
    private func currentConfig() -> YOLOConfig {
        var cfg = config
        cfg.confidenceThreshold = tunables.confidence
        cfg.iouThreshold = tunables.iou
        return cfg
    }

    public init(modelName: String = "De-plate-pose",
                config: YOLOConfig = .germanPlatePoseLibre,
                useRGB: Bool = true,
                flipY: Bool = true,
                intraOpThreads: Int32 = 2) throws {
        self.config = config
        self.inputSize = config.inputSize
        self.tunables = Tunables(confidence: config.confidenceThreshold, iou: config.iouThreshold,
                                 useRGB: useRGB, flipY: flipY)

        #if SWIFT_PACKAGE
        let bundle = Bundle.module
        #else
        let bundle = Bundle.main
        #endif
        guard let url = bundle.url(forResource: modelName, withExtension: "onnx") else {
            throw LicensePlateError.modelNotFound(modelName + ".onnx")
        }

        // ORT Objective-C API (verified against onnxruntime-mobile 1.18.0):
        //   ORTEnv(loggingLevel:), ORTSessionOptions(), ORTSession(env:modelPath:sessionOptions:),
        //   ORTValue(tensorData: NSMutableData, elementType: .float, shape: [NSNumber]),
        //   session.run(withInputs:outputNames:runOptions:) -> [String: ORTValue].
        let env = try ORTEnv(loggingLevel: .warning)
        let opts = try ORTSessionOptions()
        // ORT defaults to one intra-op thread per core, which keeps every performance core
        // busy for the whole session — the dominant heat source once the camera has been
        // running for a few minutes. A 512² YOLO-NAS-pose model does not need that many:
        // 2 threads keep latency close while leaving headroom for AVCapture + Vision.
        // Pass 0 to restore ORT's own default when A/B-testing on device.
        if intraOpThreads > 0 {
            _ = try? opts.setIntraOpNumThreads(intraOpThreads)
        }
        let session = try ORTSession(env: env, modelPath: url.path, sessionOptions: opts)
        self.session = session
        // Read tensor names from the model itself (Ultralytics uses "images"/"output0",
        // but querying avoids hardcoding assumptions across export versions).
        let names = (try? session.outputNames()) ?? []
        if names.contains("boxes"), names.contains("keypoints_xy") {
            // LibreYOLO YOLO-NAS-pose export: four named outputs, bottom-right
            // letterbox, 127 gray, BGR. The channel order is fixed by the training
            // pipeline (libreyolo `preprocess_pose_image` reverses RGB→BGR), so the
            // legacy `useRGB` runtime toggle is intentionally ignored for this layout.
            self.libreLayout = true
            self.libreOutputNames = ["boxes", "scores", "keypoints_xy", "keypoints_conf"]
                .filter { names.contains($0) }
            self.outputName = names.first ?? "boxes"
        } else {
            self.libreLayout = false
            self.libreOutputNames = []
            self.outputName = names.first ?? "output0"
        }
        self.inputName = (try? session.inputNames())?.first ?? "images"

        // Prewarm (standard practice in well-performing ONNX Runtime mobile
        // deployments): graph optimisation, arena allocation and kernel selection all
        // happen on the FIRST `run` and can cost hundreds of milliseconds on device.
        // Without this, the first real frame pays that bill — which reads as "aimed at
        // the plate and nothing happens for a second". A zero-filled tensor is enough;
        // any failure here is non-fatal (`try?`), real inference still works.
        let warmupOutputs = libreOutputNames.isEmpty ? [outputName] : libreOutputNames
        if let zeros = NSMutableData(length: 3 * inputSize * inputSize * MemoryLayout<Float>.size) {
            if let warmup = try? ORTValue(tensorData: zeros, elementType: .float,
                                          shape: [1, 3, inputSize, inputSize].map { NSNumber(value: $0) }) {
                _ = try? session.run(withInputs: [inputName: warmup],
                                     outputNames: Set(warmupOutputs), runOptions: nil)
            }
        }
    }

    /// The channel order actually fed to the model: BGR for the LibreYOLO layout,
    /// `useRGB` for the Ultralytics one.
    private var feedsRGB: Bool { !libreLayout && useRGB }

    public func detect(in image: CGImage) async throws -> [PlateDetection] {
        try Task.checkCancellation()

        let cfg = currentConfig()
        // One consistent snapshot of the runtime toggles for the whole pass: reading them
        // separately for the letterbox and the tensor could otherwise straddle a UI flip.
        let flipY = self.flipY
        let feedsRGB = self.feedsRGB
        let captureSnapshot = tunables.captureSnapshot
        // Letterbox geometry follows the export family: Ultralytics centers the resized
        // image on a 114-gray canvas; LibreYOLO YOLO-NAS-pose anchors it top-left
        // ("bottom-right" padding) on a 127-gray canvas. `padX/padY` are returned in
        // both cases so `YOLOOutputDecoder.remap` needs no per-family special case.
        let prepared = try image.letterboxedUpright(size: inputSize, flipY: flipY,
                                                    snapshot: captureSnapshot,
                                                    bottomRightPad: libreLayout,
                                                    padValue: libreLayout ? 127 : 114)
        if let snapshot = prepared.image { snapshotBox.image = snapshot }
        let tensorData = try Self.inputTensorData(from: prepared.buffer, useRGB: feedsRGB)
        // `shape` is NSArray<NSNumber*>: build it explicitly instead of relying on an
        // `[Int] as [NSNumber]` bridging cast.
        let shape = [1, 3, inputSize, inputSize].map { NSNumber(value: $0) }
        let inputValue = try ORTValue(
            tensorData: tensorData,
            elementType: .float,
            shape: shape
        )

        if libreLayout {
            return try detectLibreYOLO(
                inputValue: inputValue, cfg: cfg,
                prepared: (scale: prepared.scale, padX: prepared.padX, padY: prepared.padY),
                imageSize: CGSize(width: image.width, height: image.height))
        }

        let outputs = try session.run(
            withInputs: [inputName: inputValue],
            // `runWithInputs:outputNames:` takes an `NSSet<NSString*>*` — Swift `Set<String>`,
            // NOT an array. An array literal here is a compile error.
            outputNames: Set([outputName]),
            runOptions: nil
        )
        guard let outValue = outputs[outputName] else {
            throw LicensePlateError.invalidModelOutput("ONNX 推理未返回输出 \(outputName)。")
        }

        // Read the output using its REAL shape and element type. The ONNX export can
        // lay out proposals as [1, C, N] or [1, N, C]; hard-coding [1, channels, N]
        // silently scrambles the tensor for the [1, N, C] case — which is exactly the
        // "great on PC, poor on device" symptom, because the PC reference auto-detects
        // the layout. `YOLOOutputDecoder` already honors both via tensor strides, so we
        // just feed the actual shape. Float16 exports are upcast to float32.
        let array = try Self.outputMultiArray(from: outValue, config: cfg)

        let decoded = try YOLOOutputDecoder.decode(array, config: cfg)
        let mapped = decoded.map {
            YOLOOutputDecoder.remap($0, scale: prepared.scale, padX: prepared.padX, padY: prepared.padY,
                imageSize: CGSize(width: image.width, height: image.height))
        }.filter { $0.rect.width > 0 && $0.rect.height > 0 }
        return YOLOOutputDecoder.suppress(mapped, threshold: cfg.iouThreshold)
    }

    // MARK: - LibreYOLO (YOLO-NAS-pose) decode path

    /// Runs the four-output LibreYOLO graph and decodes it. The graph already decodes
    /// anchors into canvas-pixel xyxy boxes and applies sigmoid to scores/keypoint
    /// confidence, so there is no per-anchor math here — only thresholding, a pre-NMS
    /// top-K cap (upstream convention), and the shared remap + NMS.
    private func detectLibreYOLO(inputValue: ORTValue, cfg: YOLOConfig,
                                 prepared: (scale: CGFloat, padX: CGFloat, padY: CGFloat),
                                 imageSize: CGSize) throws -> [PlateDetection] {
        let outputs = try session.run(
            withInputs: [inputName: inputValue],
            outputNames: Set(libreOutputNames),
            runOptions: nil
        )
        func floats(_ name: String) throws -> [Float] {
            guard let value = outputs[name] else {
                throw LicensePlateError.invalidModelOutput("ONNX 推理未返回输出 \(name)。")
            }
            let data = try value.tensorData()
            let count = data.length / MemoryLayout<Float>.size
            return Array(UnsafeBufferPointer(start: data.bytes.assumingMemoryBound(to: Float.self),
                                             count: count))
        }
        let boxValues = try floats("boxes")
        let scoreValues = try floats("scores")
        // Diagnostic: the best raw anchor score, regardless of threshold. Surfaced via
        // `lastTopScore` so a "detects nothing on device" report can be triaged in one run.
        tunables.topScore = scoreValues.max()
        let keypointXYValues = try floats("keypoints_xy")
        let keypointConfValues = try floats("keypoints_conf")

        let anchorCount = scoreValues.count
        let decoded = YOLOOutputDecoder.decodeLibreYOLO(
            boxes: boxValues, scores: scoreValues,
            keypointsXY: keypointXYValues, keypointsConf: keypointConfValues,
            anchorCount: anchorCount, config: cfg)
        let mapped = decoded.map {
            YOLOOutputDecoder.remap($0, scale: prepared.scale, padX: prepared.padX, padY: prepared.padY,
                imageSize: imageSize)
        }.filter { $0.rect.width > 0 && $0.rect.height > 0 }
        return YOLOOutputDecoder.suppress(mapped, threshold: cfg.iouThreshold)
    }

    // MARK: - Preprocessing: upright letterbox -> RGB/255 NCHW float32

    private static func inputTensorData(from buffer: CVPixelBuffer, useRGB: Bool) throws -> NSMutableData {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw LicensePlateError.invalidInput("无法读取 letterbox 缓冲。")
        }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let bpr = CVPixelBufferGetBytesPerRow(buffer)
        let plane = w * h
        var floats = [Float](repeating: 0, count: 3 * plane)
        for y in 0..<h {
            let row = base.advanced(by: y * bpr).assumingMemoryBound(to: UInt8.self)
            for x in 0..<w {
                let o = x * 4
                let b = Float(row[o]) / 255.0
                let g = Float(row[o + 1]) / 255.0
                let r = Float(row[o + 2]) / 255.0
                let i = y * w + x
                if useRGB {
                    floats[i] = r; floats[plane + i] = g; floats[2 * plane + i] = b
                } else {
                    floats[i] = b; floats[plane + i] = g; floats[2 * plane + i] = r
                }
            }
        }
        return NSMutableData(bytes: &floats, length: floats.count * MemoryLayout<Float>.size)
    }

    /// Builds a float32 `MLMultiArray` from the raw ORT output, honoring its real
    /// shape and upcasting float16 if the export uses it (otherwise the decoder would
    /// read the raw bytes as float32 and produce garbage detections).
    private static func outputMultiArray(from outValue: ORTValue, config: YOLOConfig) throws -> MLMultiArray {
        let typeInfo = try outValue.tensorTypeAndShapeInfo()
        let dims = typeInfo.shape.map { $0.intValue }
        let channels = 5 + config.numKeypoints * 3
        guard dims.count == 3, dims[0] == 1,
              (dims[1] == channels || dims[2] == channels) else {
            throw LicensePlateError.invalidModelOutput(
                "意外的 ONNX 输出形状 \(dims)（期望 [1, \(channels), N] 或 [1, N, \(channels)]）。")
        }
        let rawData = try outValue.tensorData()
        let elementCount = dims.reduce(1, *)

        // Float16 export → upcast to float32 (length is half of the float32 size).
        if rawData.length == elementCount * 2 {
            // `NSMutableData.bytes` is a non-optional `UnsafeRawPointer` (unlike
            // `Data`), so it is bound with `let`, not `guard let`.
            let bytes = rawData.bytes
            let src = bytes.assumingMemoryBound(to: UInt16.self)
            var floats = [Float](repeating: 0, count: elementCount)
            for i in 0..<elementCount { floats[i] = Float(Float16(bitPattern: src[i])) }
            let array = try MLMultiArray(shape: dims.map { NSNumber(value: $0) }, dataType: .float32)
            let byteCount = floats.count * MemoryLayout<Float>.size
            floats.withUnsafeBytes { buffer in
                guard let base = buffer.baseAddress else { return }
                memcpy(array.dataPointer, base, byteCount)
            }
            return array
        }

        let array = try MLMultiArray(shape: dims.map { NSNumber(value: $0) }, dataType: .float32)
        // `bytes` is non-optional here too — it is always valid for the data's length.
        memcpy(array.dataPointer, rawData.bytes,
               min(rawData.length, array.count * MemoryLayout<Float>.size))
        return array
    }
}

extension CGImage {
    /// Letterbox into a `size`×`size` buffer for the pose model, resampling with
    /// **Accelerate vImage**.
    ///
    /// Why vImage: the reference pipelines (LibreYOLO `preprocess_numpy`, the PC
    /// scripts) resize with PIL `Image.resize(..., BILINEAR)`, which antialiases on
    /// downscale (area-weighted averaging). The previous `CGContext.draw` path used
    /// Core Graphics' plain linear interpolation with no antialiasing, so large
    /// downscales aliased thin character strokes away — exactly the features the pose
    /// model's corner heads latch onto. `vImageScale_ARGB8888` +
    /// `kvImageHighQualityResampling` is an area-averaged, antialiased resample with
    /// the same character as PIL's downscale.
    ///
    /// The destination is always written **top-row-first** (vImage row 0 = first row
    /// in memory, matching the PC / OpenCV / PIL layout), which removes the Quartz
    /// bottom-left-origin flip hazard the CGContext path had to work around.
    /// `flipY == false` remains available for on-device A/B testing and emulates the
    /// old bottom-first write by reversing the scaled rows after the resample.
    ///
    /// - Parameter snapshot: also return the buffer as a `CGImage`, i.e. exactly what
    ///   the model sees. Comparing it with the PC pipeline's input settles "is the
    ///   tensor wrong, or is the decode wrong?" in one step.
    /// - Parameter bottomRightPad: `true` = LibreYOLO YOLO-NAS-pose convention — anchor
    ///   the resized image at the canvas top-left (`padX = padY = 0`) and leave all pad
    ///   pixels at the right/bottom; `false` = Ultralytics convention — center the image.
    /// - Parameter padValue: gray fill (114 Ultralytics, 127 LibreYOLO pose).
    /// - Returns: geometry (scale/pad) consistent with `YOLOOutputDecoder.remap`.
    func letterboxedUpright(size: Int, flipY: Bool = true, snapshot: Bool = false,
                            bottomRightPad: Bool = false, padValue: UInt8 = 114) throws -> (buffer: CVPixelBuffer, scale: CGFloat, padX: CGFloat, padY: CGFloat, image: CGImage?) {
        let side = CGFloat(size)
        let scale = min(side / CGFloat(width), side / CGFloat(height))
        let resizedWidth = Int(round(CGFloat(width) * scale))
        let resizedHeight = Int(round(CGFloat(height) * scale))
        let padX = bottomRightPad ? 0 : floor((side - CGFloat(resizedWidth)) / 2)
        let padY = bottomRightPad ? 0 : floor((side - CGFloat(resizedHeight)) / 2)
        guard resizedWidth > 0, resizedHeight > 0 else {
            throw LicensePlateError.invalidInput("源图像尺寸退化，无法 letterbox。")
        }
        var pixel: CVPixelBuffer?
        let attributes = [kCVPixelBufferCGImageCompatibilityKey: true,
                          kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, size, size, kCVPixelFormatType_32BGRA, attributes, &pixel) == kCVReturnSuccess,
              let buffer = pixel else {
            throw LicensePlateError.invalidInput("无法分配 letterbox 缓冲。")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw LicensePlateError.invalidInput("无法读取 letterbox 缓冲。")
        }
        let bpr = CVPixelBufferGetBytesPerRow(buffer)

        // Fill the whole canvas with the pad gray (BGRA little-endian: B = G = R = pad).
        // The tensor builder only reads B/G/R, but the alpha bytes are set opaque so an
        // optional snapshot renders without a surprise 50% dim.
        memset(base, Int32(padValue), bpr * size)
        for y in 0..<size {
            let row = base.advanced(by: y * bpr).assumingMemoryBound(to: UInt8.self)
            for x in 0..<size { row[x * 4 + 3] = 255 }
        }

        // Decode the source into a little-endian ARGB8888 vImage buffer — the same
        // in-memory layout as `kCVPixelFormatType_32BGRA`, so scaling moves bytes
        // without touching channel order. `vImageBuffer_InitWithCGImage` converts any
        // CGImage flavour (16-bit, grayscale, Display-P3) the way the old CGContext
        // render did. The temporary buffer is one uncompressed copy of the source and
        // is freed right after the scale.
        // Apple ships a Swift-specific failable initializer that takes the `CGColorSpace`
        // object directly. The raw C-struct memberwise initializer declares `colorSpace`
        // as `Unmanaged<CGColorSpace>!`, so a bare `CGColorSpace` would NOT type-check
        // against it — use the overlay initializer.
        guard let format = vImage_CGImageFormat(
                  bitsPerComponent: 8, bitsPerPixel: 32,
                  colorSpace: CGColorSpaceCreateDeviceRGB(),
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue
                                            | CGBitmapInfo.byteOrder32Little.rawValue),
                  renderingIntent: .defaultIntent) else {
            return try legacyLetterbox(size: size, flipY: flipY, snapshot: snapshot,
                                       bottomRightPad: bottomRightPad, padValue: padValue)
        }
        var src = vImage_Buffer()
        if vImageBuffer_InitWithCGImage(&src, &format, nil, self, vImage_Flags(kvImageNoFlags)) != kvImageNoError {
            return try legacyLetterbox(size: size, flipY: flipY, snapshot: snapshot,
                                       bottomRightPad: bottomRightPad, padValue: padValue)
        }
        defer { if let p = src.data { free(p) } }

        // Resample straight into the canvas region the decoder's geometry expects:
        // top-left anchored for the LibreYOLO layout (padX = padY = 0), centered
        // otherwise. vImage writes row 0 first, so no flip is involved anywhere.
        var dst = vImage_Buffer(data: base.advanced(by: Int(padY) * bpr + Int(padX) * 4),
                                height: vImagePixelCount(resizedHeight),
                                width: vImagePixelCount(resizedWidth),
                                rowBytes: bpr)
        let scaleError = vImageScale_ARGB8888(&src, &dst, nil, vImage_Flags(kvImageHighQualityResampling))
        if scaleError != kvImageNoError {
            // Scaling should not fail for 8-bit same-layout buffers, but a silent
            // all-gray canvas is the worst possible failure mode (detections just
            // vanish), so fall back to the proven CGContext path instead.
            return try legacyLetterbox(size: size, flipY: flipY, snapshot: snapshot,
                                       bottomRightPad: bottomRightPad, padValue: padValue)
        }

        if !flipY {
            // A/B diagnostic mode: emulate the old bottom-first (upside-down) write so
            // the flip question can still be re-tested on device without rebuilding.
            var scratch = [UInt8](repeating: 0, count: resizedWidth * 4)
            let regionStart = base.advanced(by: Int(padY) * bpr + Int(padX) * 4)
            for i in 0..<resizedHeight / 2 {
                let top = regionStart.advanced(by: i * bpr)
                let bottom = regionStart.advanced(by: (resizedHeight - 1 - i) * bpr)
                memcpy(&scratch, top, resizedWidth * 4)
                memcpy(top, bottom, resizedWidth * 4)
                memcpy(bottom, &scratch, resizedWidth * 4)
            }
        }

        let snap = snapshot ? Self.uprightSnapshot(base: base, side: size, bytesPerRow: bpr) : nil
        return (buffer, scale, padX, padY, snap)
    }

    /// The pre-vImage `CGContext` implementation, kept as a fallback for sources
    /// `vImageBuffer_InitWithCGImage` cannot convert, and as the ground truth for
    /// A/B-testing the resampler on device.
    private func legacyLetterbox(size: Int, flipY: Bool, snapshot: Bool,
                                 bottomRightPad: Bool, padValue: UInt8) throws -> (buffer: CVPixelBuffer, scale: CGFloat, padX: CGFloat, padY: CGFloat, image: CGImage?) {
        let side = CGFloat(size)
        let scale = min(side / CGFloat(width), side / CGFloat(height))
        let resizedWidth = round(CGFloat(width) * scale)
        let resizedHeight = round(CGFloat(height) * scale)
        let padX = bottomRightPad ? 0 : floor((side - resizedWidth) / 2)
        let padY = bottomRightPad ? 0 : floor((side - resizedHeight) / 2)
        var pixel: CVPixelBuffer?
        let attributes = [kCVPixelBufferCGImageCompatibilityKey: true,
                          kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, size, size, kCVPixelFormatType_32BGRA, attributes, &pixel) == kCVReturnSuccess,
              let buffer = pixel else {
            throw LicensePlateError.invalidInput("无法分配 letterbox 缓冲。")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let context = CGContext(data: base, width: size, height: size, bitsPerComponent: 8,
                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: CGColorSpaceCreateDeviceRGB(),
                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw LicensePlateError.invalidInput("无法渲染 letterbox 图像。")
        }
        let gray = CGFloat(padValue) / 255.0
        context.setFillColor(red: gray, green: gray, blue: gray, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: side, height: side))
        if flipY {
            // `CGContext(data:...)` starts in the Quartz default system (origin
            // bottom-left), which would put the picture into the buffer upside down.
            // Flipping first makes row 0 the visual top and `padY` a top offset —
            // byte-for-byte the layout the PC reference produces.
            context.translateBy(x: 0, y: side)
            context.scaleBy(x: 1, y: -1)
        }
        context.draw(self, in: CGRect(x: padX, y: padY, width: resizedWidth, height: resizedHeight))
        // `makeImage()` bakes in the current transform, i.e. it returns the buffer exactly
        // as laid out in memory — what the model actually sees, flip and all.
        return (buffer, scale, padX, padY, snapshot ? context.makeImage() : nil)
    }

    /// Wraps an already-written BGRA buffer (top row first in memory) as a `CGImage`.
    /// No drawing happens on this context, so no transform is baked in: the returned
    /// image is the raw memory, i.e. exactly what the model sees.
    private static func uprightSnapshot(base: UnsafeMutableRawPointer, side: Int, bytesPerRow: Int) -> CGImage? {
        guard let ctx = CGContext(data: base, width: side, height: side, bitsPerComponent: 8,
                                  bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                            | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        return ctx.makeImage()
    }
}
#endif
