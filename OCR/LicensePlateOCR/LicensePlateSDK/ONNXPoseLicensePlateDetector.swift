#if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc)
/*
 License Plate OCR SDK — ONNX Runtime pose detector

 Abstract:
 Runs the trained German-plate YOLOv8-pose model (`De-plate-pose-500.onnx`) through
 ONNX Runtime (Microsoft `onnxruntime-mobile-objc`) and decodes both the detection
 box and the 4 plate corners (left_top, right_top, right_bottom, left_bottom).

 Output layout: `[1, 17, 8400]` = 5 (cx, cy, w, h, conf) + 4 * 3 (x, y, visibility).
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
#if canImport(onnxruntime)
import onnxruntime
#elseif canImport(onnxruntime_mobile_objc)
import onnxruntime_mobile_objc
#endif

public final class ONNXPoseLicensePlateDetector: LicensePlateDetecting, @unchecked Sendable {

    private let session: ORTSession
    private let config: YOLOConfig
    private let inputName: String
    private let outputName: String
    private let inputSize: Int
    /// Channel order fed to the model. `true` = RGB, which is what Ultralytics trains on
    /// and what the proven PC reference (`Scripts/pose_pipeline_reference.py`) feeds:
    /// `img = np.array(Image.open(path).convert("RGB"))` → `/255` → NCHW. A mismatch here
    /// is the #1 cause of "great on PC, poor on device", so the default follows the
    /// reference rather than guesswork. Still runtime-settable via
    /// `LicensePlateRecognizer.useRGB` so a differently trained export can be handled
    /// without recompiling.
    var useRGB: Bool
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
    /// without rebuilding.
    var flipY: Bool

    /// Mutable thresholds in a box: a `Sendable` class may not have mutable stored
    /// properties, and `@unchecked` on the box keeps that warning off the detector.
    private final class Tunables: @unchecked Sendable {
        var confidence: Float
        var iou: Float
        var captureSnapshot = false
        init(confidence: Float, iou: Float) { self.confidence = confidence; self.iou = iou }
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
    public var captureDebugSnapshot: Bool {
        get { tunables.captureSnapshot }
        set { tunables.captureSnapshot = newValue }
    }

    /// The base config with the current runtime thresholds applied.
    private func currentConfig() -> YOLOConfig {
        var cfg = config
        cfg.confidenceThreshold = tunables.confidence
        cfg.iouThreshold = tunables.iou
        return cfg
    }

    public init(modelName: String = "De-plate-pose-500",
                config: YOLOConfig = .germanPlatePose,
                useRGB: Bool = true,
                flipY: Bool = true) throws {
        self.config = config
        self.inputSize = config.inputSize
        self.useRGB = useRGB
        self.flipY = flipY
        self.tunables = Tunables(confidence: config.confidenceThreshold, iou: config.iouThreshold)

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
        let session = try ORTSession(env: env, modelPath: url.path, sessionOptions: opts)
        self.session = session
        // Read tensor names from the model itself (Ultralytics uses "images"/"output0",
        // but querying avoids hardcoding assumptions across export versions).
        self.inputName = (try? session.inputNames())?.first ?? "images"
        self.outputName = (try? session.outputNames())?.first ?? "output0"
    }

    public func detect(in image: CGImage) async throws -> [PlateDetection] {
        try Task.checkCancellation()

        let cfg = currentConfig()
        let prepared = try image.letterboxedUpright(size: inputSize, flipY: flipY,
                                                    snapshot: tunables.captureSnapshot)
        if let snapshot = prepared.image { snapshotBox.image = snapshot }
        let tensorData = try Self.inputTensorData(from: prepared.buffer, useRGB: useRGB)
        // `shape` is NSArray<NSNumber*>: build it explicitly instead of relying on an
        // `[Int] as [NSNumber]` bridging cast.
        let shape = [1, 3, inputSize, inputSize].map { NSNumber(value: $0) }
        let inputValue = try ORTValue(
            tensorData: tensorData,
            elementType: .float,
            shape: shape
        )
        let outputs = try session.run(
            withInputs: [inputName: inputValue],
            outputNames: [outputName],
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
    /// Letterbox into a 640x640 buffer for the pose model.
    ///
    /// - Parameter flipY: write the picture top-row-first (`true`, the Ultralytics /
    ///   OpenCV / PC-reference layout: row 0 = visual top, `padY` measured from the top)
    ///   instead of letting the Quartz default (bottom-left origin) mirror it vertically.
    ///   See `ONNXPoseLicensePlateDetector.flipY`.
    /// - Parameter snapshot: also return the buffer as a `CGImage`, i.e. exactly what the
    ///   model sees. Comparing it with the PC pipeline's input settles "is the tensor
    ///   wrong, or is the decode wrong?" in one step.
    /// - Returns: geometry (scale/pad) consistent with `YOLOOutputDecoder.remap`.
    func letterboxedUpright(size: Int, flipY: Bool = true, snapshot: Bool = false) throws -> (buffer: CVPixelBuffer, scale: CGFloat, padX: CGFloat, padY: CGFloat, image: CGImage?) {
        let side = CGFloat(size)
        let scale = min(side / CGFloat(width), side / CGFloat(height))
        let resizedWidth = round(CGFloat(width) * scale)
        let resizedHeight = round(CGFloat(height) * scale)
        let padX = floor((side - resizedWidth) / 2)
        let padY = floor((side - resizedHeight) / 2)
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
        context.setFillColor(red: 114 / 255, green: 114 / 255, blue: 114 / 255, alpha: 1)
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
}
#endif
