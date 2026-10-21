import Foundation
import CoreGraphics
import CoreImage
@preconcurrency import CoreML

/// A plate box. When the model is a pose model, `keypoints` carries the plate
/// corners (e.g. 4 corners for the German-plate pose model) in the same
/// coordinate space as `rect`: 640-input pixels before `YOLOOutputDecoder.remap`,
/// normalized 0...1 (top-left origin) after it.
public struct PlateDetection: Sendable {
    public let rect: CGRect
    public let confidence: Float
    public let label: String
    /// Optional pose keypoints (corner points) mirroring `rect`'s coordinate space.
    public let keypoints: [CGPoint]?
    /// Per-keypoint visibility/confidence. For the Ultralytics YOLOv8-pose ONNX used here this
    /// is a Sigmoid output in [0, 1] (verified: max ~0.998), NOT the COCO label convention
    /// (0 absent / 1 occluded / 2 visible). Do not compare it against values > 1.
    public let keypointConfidences: [Float]?
    public init(rect: CGRect, confidence: Float, label: String,
                keypoints: [CGPoint]? = nil, keypointConfidences: [Float]? = nil) {
        self.rect = rect; self.confidence = confidence; self.label = label
        self.keypoints = keypoints; self.keypointConfidences = keypointConfidences
    }
}

public protocol LicensePlateDetecting: Sendable {
    func detect(in image: CGImage) async throws -> [PlateDetection]
}

/// The bundled model produces [1, 5, 8400], with xywh in 640x640 input pixels.
public struct YOLOConfig: Sendable {
    public var inputSize: Int
    public var numClasses: Int
    public var classNames: [String]
    public var confidenceThreshold: Float
    public var iouThreshold: Float
    public var format: OutputFormat
    public var targetClassIndices: [Int]
    /// Number of pose keypoints per detection (0 for a plain detector model).
    /// The pose output layout is `[1, 5 + numKeypoints*3, N]`.
    public var numKeypoints: Int
    public enum OutputFormat: Sendable, Equatable {
        case xywhCenterPixels
        case xywhCenterNormalized
        case xyxyCornerNormalized
    }
    public init(inputSize: Int = 640, numClasses: Int = 1, classNames: [String] = ["license_plate"],
                confidenceThreshold: Float = 0.45, iouThreshold: Float = 0.45,
                format: OutputFormat = .xywhCenterPixels, targetClassIndices: [Int] = [0],
                numKeypoints: Int = 0) {
        self.inputSize = inputSize; self.numClasses = numClasses; self.classNames = classNames
        self.confidenceThreshold = confidenceThreshold; self.iouThreshold = iouThreshold
        self.format = format; self.targetClassIndices = targetClassIndices
        self.numKeypoints = numKeypoints
    }
    public static var germanPlate: YOLOConfig { YOLOConfig() }
    /// German-plate keypoint (pose) model: `De-plate-pose-500`. Output is
    /// `[1, 17, 8400]` = 5 (box + conf) + 4 corners * 3 (x, y, visibility).
    /// Corner order is fixed by the training config: left_top, right_top, right_bottom, left_bottom.
    public static var germanPlatePose: YOLOConfig {
        YOLOConfig(inputSize: 640, numClasses: 1, classNames: ["plate"],
                   confidenceThreshold: 0.45, iouThreshold: 0.45,
                   format: .xywhCenterPixels, targetClassIndices: [0], numKeypoints: 4)
    }
}

/// Executes the original NB graph and weights through Core ML's eligible compute units.
public final class CoreMLLicensePlateDetector: LicensePlateDetecting, @unchecked Sendable {
    private let model: MLModel
    private let config: YOLOConfig
    private let inputName: String
    private let outputName: String

    public init(modelName: String = "license_plate_yolo", config: YOLOConfig = .germanPlate,
                computeUnits: MLComputeUnits = .all) throws {
        #if SWIFT_PACKAGE
        let bundle = Bundle.module
        #else
        let bundle = Bundle.main
        #endif
        guard config.inputSize > 0, config.numClasses > 0 else {
            throw LicensePlateError.invalidInput("Invalid detector configuration.")
        }
        guard let url = bundle.url(forResource: modelName, withExtension: "mlmodelc")
                ?? bundle.url(forResource: modelName, withExtension: "mlmodel")
                ?? bundle.url(forResource: modelName, withExtension: "mlpackage") else {
            throw LicensePlateError.modelNotFound(modelName)
        }
        let compiled: URL
        if url.pathExtension == "mlmodelc" { compiled = url }
        else { compiled = try MLModel.compileModel(at: url) }
        let settings = MLModelConfiguration()
        settings.computeUnits = computeUnits
        model = try MLModel(contentsOf: compiled, configuration: settings)
        self.config = config
        guard let input = model.modelDescription.inputDescriptionsByName.first(where: { $0.value.type == .image }),
              let constraint = input.value.imageConstraint,
              constraint.pixelsWide == config.inputSize, constraint.pixelsHigh == config.inputSize else {
            throw LicensePlateError.invalidModelOutput("The model image size does not match the detector configuration.")
        }
        inputName = input.key
        guard let output = model.modelDescription.outputDescriptionsByName.first(where: {
            guard let shape = $0.value.multiArrayConstraint?.shape.map(\.intValue) else { return false }
            let channels = 4 + config.numClasses
            let poseChannels = 5 + config.numKeypoints * 3
            return shape.count == 3 && shape[0] == 1
                && (shape[1] == channels || shape[2] == channels
                    || (config.numKeypoints > 0 && (shape[1] == poseChannels || shape[2] == poseChannels)))
        }) else { throw LicensePlateError.invalidModelOutput("Expected raw YOLO box / pose output.") }
        outputName = output.key
    }

    public func detect(in image: CGImage) async throws -> [PlateDetection] {
        try Task.checkCancellation()
        let prepared = try image.letterboxed(size: config.inputSize)
        let array = try await prediction(on: prepared.buffer)
        let proposals = try YOLOOutputDecoder.decode(array, config: config)
        let mapped = proposals.map {
            YOLOOutputDecoder.remap($0, scale: prepared.scale, padX: prepared.padX, padY: prepared.padY,
                imageSize: CGSize(width: image.width, height: image.height))
        }.filter { $0.rect.width > 0 && $0.rect.height > 0 }
        return YOLOOutputDecoder.suppress(mapped, threshold: config.iouThreshold)
    }

    /// Exposes the raw graph output to package regression tests before NMS/cropping.
    func prediction(on pixelBuffer: CVPixelBuffer) async throws -> MLMultiArray {
        let features = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: pixelBuffer)])
        let output = try await model.prediction(from: features, options: MLPredictionOptions())
        guard let array = output.featureValue(for: outputName)?.multiArrayValue else {
            throw LicensePlateError.invalidModelOutput("The model did not return its configured YOLO output.")
        }
        return array
    }
}

extension CGImage {
    /// Preserves the original iOS detector's centered 114-gray padding and top-left pixels.
    func letterboxed(size: Int) throws -> (buffer: CVPixelBuffer, scale: CGFloat, padX: CGFloat, padY: CGFloat) {
        let side = CGFloat(size)
        let scale = min(side / CGFloat(width), side / CGFloat(height))
        let resizedWidth = round(CGFloat(width) * scale), resizedHeight = round(CGFloat(height) * scale)
        let padX = floor((side - resizedWidth) / 2), padY = floor((side - resizedHeight) / 2)
        var pixel: CVPixelBuffer?
        let attributes = [kCVPixelBufferCGImageCompatibilityKey: true,
                          kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, size, size, kCVPixelFormatType_32BGRA, attributes, &pixel) == kCVReturnSuccess,
              let buffer = pixel else { throw LicensePlateError.invalidInput("Could not allocate the detector image.") }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let context = CGContext(data: base, width: size, height: size, bitsPerComponent: 8,
                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: CGColorSpaceCreateDeviceRGB(),
                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw LicensePlateError.invalidInput("Could not render the detector image.")
        }
        context.setFillColor(red: 114 / 255, green: 114 / 255, blue: 114 / 255, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: side, height: side))
        context.translateBy(x: 0, y: side); context.scaleBy(x: 1, y: -1)
        context.interpolationQuality = .medium
        context.draw(self, in: CGRect(x: padX, y: padY, width: resizedWidth, height: resizedHeight))
        return (buffer, scale, padX, padY)
    }
}
