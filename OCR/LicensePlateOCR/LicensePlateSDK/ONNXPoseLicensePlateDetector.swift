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
    /// Ultralytics ONNX expects RGB normalized by /255. If detections come out wrong
    /// on first device run, flip this to `false` (BGR) — a one-line change.
    private let useRGB: Bool

    public init(modelName: String = "De-plate-pose-500",
                config: YOLOConfig = .germanPlatePose,
                useRGB: Bool = true) throws {
        self.config = config
        self.inputSize = config.inputSize
        self.useRGB = useRGB

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

        let prepared = try image.letterboxedUpright(size: inputSize)
        let tensorData = try Self.inputTensorData(from: prepared.buffer, useRGB: useRGB)
        let inputValue = try ORTValue(
            tensorData: tensorData,
            elementType: .float,
            shape: [1, 3, inputSize, inputSize] as [NSNumber]
        )
        let outputs = try session.run(
            withInputs: [inputName: inputValue],
            outputNames: [outputName],
            runOptions: nil
        )
        guard let outValue = outputs[outputName] else {
            throw LicensePlateError.invalidModelOutput("ONNX 推理未返回输出 \(outputName)。")
        }
        let outData = try outValue.tensorData()

        // Rebuild the [1, 17, 8400] (or [1, 8400, 17]) array as an MLMultiArray so the
        // shared decoder can read it with real strides/types.
        let channels = 5 + config.numKeypoints * 3
        let total = outData.length / MemoryLayout<Float>.size
        let proposals = total / channels           // N = 8400 for a 640 model
        let array = try Self.makeMultiArray(data: outData, shape: [1, channels, proposals])

        let decoded = try YOLOOutputDecoder.decode(array, config: config)
        let mapped = decoded.map {
            YOLOOutputDecoder.remap($0, scale: prepared.scale, padX: prepared.padX, padY: prepared.padY,
                imageSize: CGSize(width: image.width, height: image.height))
        }.filter { $0.rect.width > 0 && $0.rect.height > 0 }
        return YOLOOutputDecoder.suppress(mapped, threshold: config.iouThreshold)
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

    private static func makeMultiArray(data: NSMutableData, shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        if let src = data.bytes {
            memcpy(array.dataPointer, src, data.length)
        }
        return array
    }
}

extension CGImage {
    /// Upright (OpenCV-style, top-row-first) letterbox. YOLOv8/Ultralytics expects the
    /// image in this orientation; do NOT reuse the flipped `letterboxed` used by the old
    /// Core ML detector. Returns geometry (scale/pad) consistent with `YOLOOutputDecoder.remap`.
    func letterboxedUpright(size: Int) throws -> (buffer: CVPixelBuffer, scale: CGFloat, padX: CGFloat, padY: CGFloat) {
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
        // No y-flip: keep the image upright so row 0 is the visual top (OpenCV convention).
        context.draw(self, in: CGRect(x: padX, y: padY, width: resizedWidth, height: resizedHeight))
        return (buffer, scale, padX, padY)
    }
}
#endif
