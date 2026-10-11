/*
 License Plate OCR SDK
 
 Abstract:
 YOLO object-detection wrapper that locates German license plates inside an image.
 The detector is intentionally model-agnostic: it loads any Core ML YOLO export at
 runtime and decodes the most common Ultralytics YOLOv8 output layout
 (`[1, N, 4 + numClasses]`, normalized center-x/center-y/width/height).
 Tune `YOLOConfig` if your export uses a different layout.
 */

import Foundation
import CoreGraphics
import CoreImage
@preconcurrency import CoreML
import Accelerate

// MARK: - Detection result

/// A single plate detection, in normalized coordinates (origin = top-left, range 0...1).
public struct PlateDetection: Sendable {
    /// Normalized bounding box (x, y, width, height) with origin at the top-left.
    public let rect: CGRect
    /// Detection confidence in 0...1.
    public let confidence: Float
    /// Class label from the model (usually `"license_plate"`).
    public let label: String

    public init(rect: CGRect, confidence: Float, label: String) {
        self.rect = rect
        self.confidence = confidence
        self.label = label
    }
}

// MARK: - Detector protocol

/// Abstraction over a license-plate localizer so the SDK can be tested or swapped.
public protocol LicensePlateDetecting: Sendable {
    /// Detects plates in a Core Graphics image.
    func detect(in image: CGImage) async throws -> [PlateDetection]
}

// MARK: - YOLO configuration

/// Describes how to interpret a YOLO Core ML export.
public struct YOLOConfig: Sendable {
    /// Square input size the model expects (e.g. 640).
    public var inputSize: Int
    /// Number of classes the model predicts.
    public var numClasses: Int
    /// Human-readable class names, indexed by class id.
    public var classNames: [String]
    /// Minimum class confidence to keep a proposal.
    public var confidenceThreshold: Float
    /// IoU threshold for non-maximum suppression.
    public var iouThreshold: Float
    /// Coordinate convention of the model output.
    public var format: OutputFormat
    /// Only keep detections whose class id is in this set. Empty = keep all.
    /// For a generic COCO model used as a placeholder, set this to `[2]` (car).
    public var targetClassIndices: [Int]

    public enum OutputFormat: Sendable {
        /// `[x_center, y_center, width, height]`, normalized to 0...1 of the input.
        case xywhCenterNormalized
        /// `[x_min, y_min, x_max, y_max]`, normalized to 0...1 of the input.
        case xyxyCornerNormalized
    }

    public init(
        inputSize: Int = 640,
        numClasses: Int = 1,
        classNames: [String] = ["license_plate"],
        confidenceThreshold: Float = 0.25,
        iouThreshold: Float = 0.45,
        format: OutputFormat = .xywhCenterNormalized,
        targetClassIndices: [Int] = []
    ) {
        self.inputSize = inputSize
        self.numClasses = numClasses
        self.classNames = classNames
        self.confidenceThreshold = confidenceThreshold
        self.iouThreshold = iouThreshold
        self.format = format
        self.targetClassIndices = targetClassIndices
    }

    /// Sensible defaults for a single-class German license-plate detector.
    public static var germanPlate: YOLOConfig {
        YOLOConfig(inputSize: 640, numClasses: 1, classNames: ["license_plate"],
                   confidenceThreshold: 0.25, iouThreshold: 0.45,
                   format: .xywhCenterNormalized, targetClassIndices: [0])
    }

    /// Defaults for the official Ultralytics `yolov8n` COCO model used as a placeholder.
    /// Keep only the `car` class (id 2); input is 640×640.
    public static var cocoCar: YOLOConfig {
        YOLOConfig(inputSize: 640, numClasses: 80, classNames: YOLOConfig.cocoNames,
                   confidenceThreshold: 0.30, iouThreshold: 0.45,
                   format: .xywhCenterNormalized, targetClassIndices: [2])
    }

    /// The 80 COCO class names, indexed by class id (matches Ultralytics YOLOv8).
    public static let cocoNames: [String] = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
}

// MARK: - Core ML YOLO detector

/// Loads a Core ML YOLO model from the main bundle and runs inference.
///
/// `@unchecked Sendable`: the detector only holds immutable references (`MLModel` is
/// not itself `Sendable`) and is used read-only, so it is safe to cross actors.
public final class CoreMLLicensePlateDetector: LicensePlateDetecting, @unchecked Sendable {
    private let model: MLModel
    private let config: YOLOConfig

    /// - Parameters:
    ///   - modelName: Resource name of the compiled model (without extension), e.g. `"license_plate_yolo"`.
    ///   - config: Output-layout description for your specific export.
    public init(modelName: String = "license_plate_yolo", config: YOLOConfig = .germanPlate) throws {
        let url = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc")
            ?? Bundle.main.url(forResource: modelName, withExtension: "mlpackage")
        guard let modelURL = url else {
            throw LicensePlateError.modelNotFound(modelName)
        }
        self.model = try MLModel(contentsOf: modelURL)
        self.config = config
    }

    public func detect(in image: CGImage) async throws -> [PlateDetection] {
        let inputSize = CGFloat(config.inputSize)

        // 1. Letterbox the image into a square pixel buffer.
        let prepared = image.preparedForYOLO(size: inputSize)
        guard let pixelBuffer = prepared.buffer else {
            throw LicensePlateError.invalidInput("无法为 YOLO 预处理图像")
        }

        // 2. Build the model input feature provider.
        let inputKey = model.modelDescription.inputDescriptionsByName.first?.key
            ?? "image"
        let inputFeature = try MLDictionaryFeatureProvider(
            dictionary: [inputKey: MLFeatureValue(pixelBuffer: pixelBuffer)]
        )

        // 3. Run inference. (On current Xcode/Core ML SDKs `prediction` is async;
        //    on older Xcode 16 / iOS 18 SDKs it is synchronous — drop `await` there.)
        let outputFeatures = try await model.prediction(from: inputFeature, options: MLPredictionOptions())
        guard let multiArray = outputFeatures.featureValue(for: outputFeatures.featureNames.first ?? inputKey)?
            .multiArrayValue else {
            throw LicensePlateError.invalidModelOutput("未找到 MLMultiArray 输出")
        }

        // 4. Decode proposals -> detections (still in model/letterbox space).
        var proposals = decode(multiArray: multiArray, inputSize: inputSize)

        // 5. Map letterbox coords back to the original image, then NMS.
        proposals = proposals.map { remap($0, scale: prepared.scale, padX: prepared.padX, padY: prepared.padY,
                                          imageSize: CGSize(width: image.width, height: image.height)) }
        proposals = nonMaxSuppression(proposals, iouThreshold: config.iouThreshold)

        return proposals
    }

    // MARK: Decoding

    private func decode(multiArray: MLMultiArray, inputSize: CGFloat) -> [PlateDetection] {
        let shape = multiArray.shape.map { $0.intValue }
        guard shape.count == 3 else { return [] }
        let numClasses = config.numClasses
        let c = 4 + numClasses

        // YOLOv8 exports the raw tensor as `[1, C, N]` (channels-first, e.g.
        // `[1, 84, 8400]`). Some exports are `[1, N, C]`. Detect which dimension
        // carries the `4 + numClasses` channels so we read it correctly either way.
        let dim1 = shape[1], dim2 = shape[2]
        let channelsFirst: Bool
        if dim1 == c && dim2 > c { channelsFirst = true }
        else if dim2 == c && dim1 > c { channelsFirst = false }
        else { return [] }

        let s1 = multiArray.strides[1].intValue   // stride of dim1
        let s2 = multiArray.strides[2].intValue   // stride of dim2
        let ptr = multiArray.dataPointer.bindMemory(
            to: Float.self, capacity: shape[0] * dim1 * dim2)

        // Feature address = proposal * proposalStride + featureIndex * featureStride.
        let proposalStride: Int
        let featureStride: Int
        if channelsFirst {
            // [1, C, N]: feature lives on dim1, proposal on dim2.
            proposalStride = s2
            featureStride = s1
        } else {
            // [1, N, C]: proposal lives on dim1, feature on dim2.
            proposalStride = s1
            featureStride = s2
        }

        let numProposals = channelsFirst ? dim2 : dim1
        let target = config.targetClassIndices

        var out: [PlateDetection] = []
        out.reserveCapacity(numProposals)

        for p in 0..<numProposals {
            let cx = ptr[p * proposalStride + 0 * featureStride]
            let cy = ptr[p * proposalStride + 1 * featureStride]
            let w  = ptr[p * proposalStride + 2 * featureStride]
            let h  = ptr[p * proposalStride + 3 * featureStride]

            // Class confidence: max over class channels.
            var bestScore: Float = 0
            var bestClass = 0
            for k in 0..<numClasses {
                let score = ptr[p * proposalStride + (4 + k) * featureStride]
                if score > bestScore { bestScore = score; bestClass = k }
            }
            guard bestScore >= config.confidenceThreshold else { continue }
            guard target.isEmpty || target.contains(bestClass) else { continue }

            // Convert to pixel coordinates inside the (square) model input.
            let (x1, y1, bw, bh): (CGFloat, CGFloat, CGFloat, CGFloat)
            switch config.format {
            case .xywhCenterNormalized:
                let px = CGFloat(cx) * inputSize
                let py = CGFloat(cy) * inputSize
                let pw = CGFloat(w)  * inputSize
                let ph = CGFloat(h)  * inputSize
                x1 = px - pw / 2
                y1 = py - ph / 2
                bw = pw; bh = ph
            case .xyxyCornerNormalized:
                x1 = CGFloat(cx) * inputSize
                y1 = CGFloat(cy) * inputSize
                bw = CGFloat(w) * inputSize - x1
                bh = CGFloat(h) * inputSize - y1
            }

            let rect = CGRect(x: x1, y: y1, width: max(0, bw), height: max(0, bh))
            let label = config.classNames[safe: bestClass] ?? "class \(bestClass)"
            out.append(PlateDetection(rect: rect, confidence: bestScore, label: label))
        }
        return out
    }

    // MARK: Coordinate remap (letterbox -> original image, normalized)

    private func remap(_ det: PlateDetection, scale: CGFloat, padX: CGFloat, padY: CGFloat,
                       imageSize: CGSize) -> PlateDetection {
        let x = (det.rect.origin.x - padX) / scale
        let y = (det.rect.origin.y - padY) / scale
        let w = det.rect.width / scale
        let h = det.rect.height / scale

        let nx = max(0, min(1, x / imageSize.width))
        let ny = max(0, min(1, y / imageSize.height))
        let nw = max(0, min(1 - nx, w / imageSize.width))
        let nh = max(0, min(1 - ny, h / imageSize.height))
        return PlateDetection(rect: CGRect(x: nx, y: ny, width: nw, height: nh),
                              confidence: det.confidence, label: det.label)
    }

    // MARK: Non-maximum suppression

    private func nonMaxSuppression(_ detections: [PlateDetection], iouThreshold: Float) -> [PlateDetection] {
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var kept: [PlateDetection] = []
        var suppressed = [Bool](repeating: false, count: sorted.count)

        for i in 0..<sorted.count {
            guard !suppressed[i] else { continue }
            kept.append(sorted[i])
            for j in (i + 1)..<sorted.count where !suppressed[j] {
                if iou(sorted[i].rect, sorted[j].rect) > iouThreshold {
                    suppressed[j] = true
                }
            }
        }
        return kept
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        guard !inter.isNull, inter.width > 0, inter.height > 0 else { return 0 }
        let interArea = inter.width * inter.height
        let union = a.width * a.height + b.width * b.height - interArea
        guard union > 0 else { return 0 }
        return Float(interArea / union)
    }
}

// MARK: - CGImage YOLO helpers

private extension CGImage {
    /// Letterboxes the image into a `size x size` pixel buffer (gray background),
    /// returning the scale and padding used so detections can be remapped.
    func preparedForYOLO(size: CGFloat) -> (buffer: CVPixelBuffer?, scale: CGFloat, padX: CGFloat, padY: CGFloat) {
        let w = CGFloat(width), h = CGFloat(height)
        let scale = min(size / w, size / h)
        let resizedW = floor(w * scale)
        let resizedH = floor(h * scale)
        let padX = (size - resizedW) / 2
        let padY = (size - resizedH) / 2

        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(size), Int(size),
                                        kCVPixelFormatType_32BGRA, attrs, &buffer)
        guard status == kCVReturnSuccess, let buf = buffer else {
            return (nil, scale, padX, padY)
        }
        CVPixelBufferLockBaseAddress(buf, .init(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(buf, .init(rawValue: 0)) }
        guard let base = CVPixelBufferGetBaseAddress(buf) else {
            return (buf, scale, padX, padY)
        }
        let ctx = CGContext(data: base,
                            width: Int(size), height: Int(size),
                            bitsPerComponent: 8,
                            bytesPerRow: CVPixelBufferGetBytesPerRow(buf),
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue)
        // Gray background (YOLO training padding color is ~114).
        ctx?.setFillColor(red: 114/255, green: 114/255, blue: 114/255, alpha: 1)
        ctx?.fill(CGRect(x: 0, y: 0, width: size, height: size))
        ctx?.draw(self, in: CGRect(x: padX, y: padY, width: resizedW, height: resizedH))
        return (buf, scale, padX, padY)
    }
}

// MARK: - Small safe-access helpers

private extension Collection {
    subscript(safe index: Index) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
