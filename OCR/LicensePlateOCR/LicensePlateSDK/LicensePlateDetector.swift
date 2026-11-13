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

/// Tunables + diagnostics every detector exposes.
///
/// The thresholds are runtime-settable because the working value can only be confirmed
/// on device against a known-good PC run. The PC pipeline that "detects everything"
/// (`截取车牌区域并保存-2.py`) calls `model(image)` with no overrides, so it runs on the
/// Ultralytics defaults **conf = 0.25, iou = 0.7** — those are the values mirrored here.
/// (The video script passes `conf=0.7` explicitly; that is a *stricter* run, not the
/// reference for recall.)
public protocol LicensePlateDetecting: AnyObject, Sendable {
    var confidenceThreshold: Float { get set }
    var nmsIoUThreshold: Float { get set }
    /// The exact image handed to the model (640x640 letterbox) when snapshotting is on.
    /// Comparing this against the PC input settles "is the tensor wrong?" in one step.
    var debugSnapshot: CGImage? { get }
    var captureDebugSnapshot: Bool { get set }
    /// Whether this detector's preprocessing actually consumes a configurable channel
    /// order. `false` means the channel order is fixed by the model/export (LibreYOLO
    /// YOLO-NAS-pose is BGR-only; Core ML normalizes the buffer itself), so a UI toggle
    /// for RGB/BGR would be a no-op and should be disabled rather than shown as active.
    var honorsChannelOrder: Bool { get }
    func detect(in image: CGImage) async throws -> [PlateDetection]
}

public extension LicensePlateDetecting {
    var confidenceThreshold: Float { get { 0.25 } set {} }
    var nmsIoUThreshold: Float { get { 0.7 } set {} }
    var debugSnapshot: CGImage? { nil }
    var captureDebugSnapshot: Bool { get { false } set {} }
    /// Opt-in: a detector that ignores the channel order says so by overriding this.
    var honorsChannelOrder: Bool { false }
}

/// Tiled detection for large still images.
///
/// The German-plate pose model exports a **fixed 640x640 input**, so a single pass
/// letterboxes the whole picture and every plate shrinks with it — on multi-plate
/// collages / parking-lot photos most plates drop below the confidence threshold
/// (the "detects everything on the PC, only a couple of boxes in the app" case).
/// Slicing the image into overlapping tiles keeps each region at a resolution the
/// model actually sees; the per-tile results are mapped back to full-image
/// normalized coordinates and merged with NMS. Small images fall through to one
/// plain `detect` pass, so the common case pays no extra cost.
public extension LicensePlateDetecting {
    /// - Parameters:
    ///   - tileSize: maximum tile edge in pixels (each tile is letterboxed to the model
    ///     input by `detect`).
    ///   - overlap: fractional overlap between neighbouring tiles so a plate cut off at
    ///     one tile edge is fully inside the next one.
    ///   - nmsThreshold: IoU threshold for the final cross-tile merge (a plate seen by
    ///     two tiles must not be reported twice). Matches the PC reference's 0.7; lower
    ///     values merge genuinely distinct plates that sit close together.
    func detectTiled(in image: CGImage, tileSize: CGFloat = 1024, overlap: CGFloat = 0.3,
                     nmsThreshold: Float = 0.7) async throws -> [PlateDetection] {
        let width = CGFloat(image.width), height = CGFloat(image.height)
        let tile = CGSize(width: min(tileSize, width), height: min(tileSize, height))
        guard max(width, height) > max(tile.width, tile.height) else {
            return try await detect(in: image)
        }
        // Tile origins along one axis: step until the next tile would overshoot, then
        // always pin the last tile to the far edge so the border is fully covered.
        func offsets(length: CGFloat, tileLength: CGFloat) -> [CGFloat] {
            guard length > tileLength else { return [0] }
            let step = tileLength * (1 - overlap)
            var result: [CGFloat] = []
            var offset: CGFloat = 0
            while offset + tileLength < length - 1 {
                result.append(offset)
                offset += step
            }
            result.append(length - tileLength)
            return result
        }

        var proposals: [PlateDetection] = []
        for tileY in offsets(length: height, tileLength: tile.height) {
            for tileX in offsets(length: width, tileLength: tile.width) {
                let crop = CGRect(x: tileX, y: tileY, width: tile.width, height: tile.height).integral
                guard let tileImage = image.cropping(to: crop) else { continue }
                // `detect` returns boxes/keypoints normalized to the tile it was given.
                for detection in try await detect(in: tileImage) {
                    func mapX(_ nx: CGFloat) -> CGFloat { min(1, max(0, (crop.minX + nx * crop.width) / width)) }
                    func mapY(_ ny: CGFloat) -> CGFloat { min(1, max(0, (crop.minY + ny * crop.height) / height)) }
                    let rect = CGRect(x: mapX(detection.rect.minX), y: mapY(detection.rect.minY),
                                      width: detection.rect.width * crop.width / width,
                                      height: detection.rect.height * crop.height / height)
                    let keypoints = detection.keypoints?.map { CGPoint(x: mapX($0.x), y: mapY($0.y)) }
                    proposals.append(PlateDetection(rect: rect, confidence: detection.confidence,
                                                    label: detection.label, keypoints: keypoints,
                                                    keypointConfidences: detection.keypointConfidences))
                }
            }
        }
        return YOLOOutputDecoder.suppress(proposals, threshold: nmsThreshold)
    }
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
    /// German-plate keypoint (pose) model: Ultralytics YOLOv8-pose export (e.g. the
    /// earlier `De-plate-pose-500`). Output is
    /// `[1, 17, 8400]` = 5 (box + conf) + 4 corners * 3 (x, y, visibility).
    /// Corner order is fixed by the training config: left_top, right_top, right_bottom, left_bottom.
    public static var germanPlatePose: YOLOConfig {
        // Thresholds mirror the PC reference exactly (`截取车牌区域并保存-2.py` calls
        // `model(image)` with no overrides, i.e. Ultralytics defaults conf=0.25/iou=0.7):
        //   • conf 0.25 — the PC run that "detects everything" uses this. A 0.45 gate
        //     silently discards every plate the model scores between 0.25 and 0.45.
        //   • iou 0.7 — must match. At 0.45 two neighbouring plates whose boxes overlap
        //     by ~half are merged into one: the "2-3 plates in frame, 1 reported" symptom.
        // Both stay runtime-tunable so the working pair can be confirmed on device.
        YOLOConfig(inputSize: 640, numClasses: 1, classNames: ["plate"],
                   confidenceThreshold: 0.25, iouThreshold: 0.7,
                   format: .xywhCenterPixels, targetClassIndices: [0], numKeypoints: 4)
    }
    /// LibreYOLO (YOLO-NAS-pose) German-plate corner model: `De-plate-pose.onnx`.
    /// Four separate outputs — `boxes [1,N,4]` (x1y1x2y2, input-canvas pixels),
    /// `scores [1,N,1]` (post-sigmoid), `keypoints_xy [1,N,4,2]`, `keypoints_conf [1,N,4]`
    /// — with a **512×512** input. Preprocessing differs from Ultralytics on every axis:
    /// **bottom-right** letterbox (top-left anchored, no centering), **127** gray fill,
    /// **BGR** channel order, /255. All verified against the training val split
    /// (IoU 0.84–0.93 with GT; see `Scripts/libreyolo_onnx_probe.py`). The detector
    /// auto-detects this layout from the model's output names, so both configs work
    /// through the same `ONNXPoseLicensePlateDetector`.
    /// Corner order is fixed by the training config: left_top, right_top, right_bottom,
    /// left_bottom (`data.yaml` / `pose_config.yaml` in the training folder).
    public static var germanPlatePoseLibre: YOLOConfig {
        // conf 0.25 mirrors the eval convention; the LibreYOLO head scores lower than
        // the Ultralytics one (val max 0.36–0.67), so this gate is tuned looser than it
        // looks — every remaining value stays runtime-tunable for on-device A/B.
        YOLOConfig(inputSize: 512, numClasses: 1, classNames: ["plate"],
                   confidenceThreshold: 0.25, iouThreshold: 0.7,
                   format: .xywhCenterPixels, targetClassIndices: [0], numKeypoints: 4)
    }
}
