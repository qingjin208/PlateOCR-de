import Foundation
import CoreGraphics
import CoreImage
import UIKit
import UltralyticsYOLO

/// A plate box. When the model is a pose model, `keypoints` carries the plate
/// corners (4 corners for the German-plate pose model) in the same coordinate
/// space as `rect`: normalized 0...1 with a top-left origin, relative to the
/// image that was passed to `detect(in:)`.
public struct PlateDetection: Sendable {
    public let rect: CGRect
    public let confidence: Float
    public let label: String
    /// Optional pose keypoints (corner points) mirroring `rect`'s coordinate space.
    public let keypoints: [CGPoint]?
    /// Per-keypoint confidence in [0, 1] (Ultralytics pose exports a Sigmoid head,
    /// NOT the COCO 0/1/2 label convention).
    public let keypointConfidences: [Float]?
    public init(rect: CGRect, confidence: Float, label: String,
                keypoints: [CGPoint]? = nil, keypointConfidences: [Float]? = nil) {
        self.rect = rect; self.confidence = confidence; self.label = label
        self.keypoints = keypoints; self.keypointConfidences = keypointConfidences
    }
}

// MARK: - Geometry helpers

/// Corner ordering and quad utilities shared by the detector and the overlay.
public enum PlateGeometry {

    /// Reorders 4 points to [leftTop, rightTop, rightBottom, leftBottom] using the
    /// centroid quadrants. The pose model is trained with a fixed corner order, but
    /// trusting it blindly turns a mis-ordered quad into a bow-tie warp, so the order
    /// is always re-derived from geometry.
    public static func orderedCorners(_ points: [CGPoint]) -> [CGPoint] {
        guard points.count == 4 else { return points }
        let cx = points.reduce(0) { $0 + $1.x } / 4
        let cy = points.reduce(0) { $0 + $1.y } / 4
        var lt = CGPoint.zero, rt = CGPoint.zero, rb = CGPoint.zero, lb = CGPoint.zero
        for p in points {
            switch (p.x < cx, p.y < cy) {
            case (true, true): lt = p
            case (false, true): rt = p
            case (false, false): rb = p
            case (true, false): lb = p
            }
        }
        return [lt, rt, rb, lb]
    }

    /// Axis-aligned bounds of a point set, clamped to the unit square.
    public static func boundingBox(_ points: [CGPoint]) -> CGRect? {
        guard let first = points.first else { return nil }
        var minX = first.x, minY = first.y, maxX = first.x, maxY = first.y
        for p in points.dropFirst() {
            minX = min(minX, p.x); minY = min(minY, p.y)
            maxX = max(maxX, p.x); maxY = max(maxY, p.y)
        }
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
            .intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
    }

    public static func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let overlap = a.intersection(b)
        guard !overlap.isNull, overlap.width > 0, overlap.height > 0 else { return 0 }
        let union = a.width * a.height + b.width * b.height - overlap.width * overlap.height
        guard union > 0 else { return 0 }
        return (overlap.width * overlap.height) / union
    }
}

/// Greedy IoU non-maximum suppression over normalized detection rectangles.
public enum PlateNMS {
    public static func suppress(_ detections: [PlateDetection], threshold: Float) -> [PlateDetection] {
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var kept: [PlateDetection] = []
        for candidate in sorted {
            let dominated = kept.contains { PlateGeometry.iou($0.rect, candidate.rect) > CGFloat(threshold) }
            if !dominated { kept.append(candidate) }
        }
        return kept
    }
}

// MARK: - Detector protocol

/// Tunables every detector exposes.
public protocol LicensePlateDetecting: AnyObject, Sendable {
    var confidenceThreshold: Float { get set }
    var nmsIoUThreshold: Float { get set }
    /// The exact image handed to the model (640x640 letterbox) when snapshotting is on.
    var debugSnapshot: CGImage? { get }
    var captureDebugSnapshot: Bool { get set }
    func detect(in image: CGImage) async throws -> [PlateDetection]
}

public extension LicensePlateDetecting {
    var confidenceThreshold: Float { get { 0.25 } set {} }
    var nmsIoUThreshold: Float { get { 0.7 } set {} }
    var debugSnapshot: CGImage? { nil }
    var captureDebugSnapshot: Bool { get { false } set {} }
}

/// Tiled detection for large still images.
///
/// The pose model has a fixed 640x640 input, so a single pass letterboxes the whole
/// picture and every plate shrinks with it. Slicing into overlapping tiles keeps each
/// region at a resolution the model actually sees; per-tile results are mapped back to
/// full-image normalized coordinates and merged with NMS. Small images fall through to
/// one plain `detect` pass.
public extension LicensePlateDetecting {
    func detectTiled(in image: CGImage, tileSize: CGFloat = 1024, overlap: CGFloat = 0.3,
                     nmsThreshold: Float = 0.7) async throws -> [PlateDetection] {
        let width = CGFloat(image.width), height = CGFloat(image.height)
        let tile = CGSize(width: min(tileSize, width), height: min(tileSize, height))
        guard max(width, height) > max(tile.width, tile.height) else {
            return try await detect(in: image)
        }
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
        return PlateNMS.suppress(proposals, threshold: nmsThreshold)
    }
}

// MARK: - Official UltralyticsYOLO Core ML pose detector

/// Plate corner detector backed by the **official Ultralytics YOLO iOS package**.
///
/// Detection quality on device matches the official app because preprocessing
/// (Vision `scaleFit` letterbox), the pose decoder, NMS and the thresholds all come
/// from that package instead of a hand-rolled pipeline. The model is the Core ML
/// `.mlpackage` exported from `De-plate-pose-500.pt` (`Scripts/export_pose_coreml.py`);
/// the package compiles it on first use, so the raw `.mlpackage` can ship in Resources.
///
/// Output contract (unchanged for the rest of the SDK): rect and keypoints are
/// normalized 0...1 with a top-left origin, relative to the image passed in.
public final class UltralyticsPosePlateDetector: LicensePlateDetecting, @unchecked Sendable {

    public let modelName: String

    private let yolo: YOLO
    private let tunables: Tunables
    private let loadState = LoadState()

    private final class Tunables: @unchecked Sendable {
        var confidence: Float
        var iou: Float
        init(confidence: Float, iou: Float) { self.confidence = confidence; self.iou = iou }
    }

    private final class LoadState: @unchecked Sendable {
        var failure: Error?
    }

    public var confidenceThreshold: Float {
        get { tunables.confidence }
        set { tunables.confidence = newValue; yolo.setConfidenceThreshold(Double(newValue)) }
    }

    public var nmsIoUThreshold: Float {
        get { tunables.iou }
        set { tunables.iou = newValue; yolo.setIouThreshold(Double(newValue)) }
    }

    public var debugSnapshot: CGImage? { nil }
    public var captureDebugSnapshot: Bool { get { false } set {} }

    /// - Parameter modelName: bundle resource name of the exported `.mlpackage` / `.mlmodelc`.
    public init(modelName: String = "De-plate-pose-500",
                confidence: Float = 0.25,
                iou: Float = 0.7) throws {
        let bundle = Self.resourceBundle
        guard bundle.url(forResource: modelName, withExtension: "mlpackage") != nil
                || bundle.url(forResource: modelName, withExtension: "mlmodelc") != nil
                || bundle.url(forResource: modelName, withExtension: "mlmodel") != nil else {
            throw LicensePlateError.modelNotFound(modelName)
        }
        self.modelName = modelName
        self.tunables = Tunables(confidence: confidence, iou: iou)

        let state = LoadState()
        let predictor = YOLO(modelName, task: .pose) { result in
            if case .failure(let error) = result { state.failure = error }
        }
        predictor.setConfidenceThreshold(Double(confidence))
        predictor.setIouThreshold(Double(iou))
        self.yolo = predictor
        self.loadState = state
    }

    private static var resourceBundle: Bundle {
        #if SWIFT_PACKAGE
        return Bundle.module
        #else
        return Bundle.main
        #endif
    }

    public var detectorDescription: String { "UltralyticsPosePlateDetector(\(modelName))" }

    public func detect(in image: CGImage) async throws -> [PlateDetection] {
        try await waitUntilLoaded()
        let result = yolo(CIImage(cgImage: image))
        return Self.convert(result)
    }

    /// Model loading is asynchronous inside the package; inference before it completes
    /// returns an empty result, so the first call waits (with a bound) instead of
    /// silently reporting "no plate".
    private func waitUntilLoaded() async throws {
        for _ in 0..<600 {  // ~6 s
            if yolo.isLoaded { return }
            if let failure = loadState.failure { throw failure }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        if let failure = loadState.failure { throw failure }
        throw LicensePlateError.modelNotFound(modelName)
    }

    /// Converts an official `YOLOResult` into SDK detections.
    ///
    /// The rectangle is derived from the four corners rather than the box head: the
    /// pose head's box output can sit slightly off the keypoints, and every consumer
    /// downstream (perspective correction, overlays) uses the corners anyway.
    static func convert(_ result: YOLOResult) -> [PlateDetection] {
        guard !result.keypointsList.isEmpty else { return [] }
        return result.keypointsList.enumerated().compactMap { index, keypoints in
            guard keypoints.xyn.count >= 4 else { return nil }
            let raw = keypoints.xyn.prefix(4).map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
            let corners = PlateGeometry.orderedCorners(raw)
            guard let rect = PlateGeometry.boundingBox(corners), rect.width > 0, rect.height > 0 else {
                return nil
            }
            let confidence: Float
            if index < result.boxes.count {
                confidence = result.boxes[index].conf
            } else if keypoints.conf.isEmpty {
                confidence = 0
            } else {
                confidence = keypoints.conf.reduce(0, +) / Float(keypoints.conf.count)
            }
            return PlateDetection(rect: rect, confidence: confidence, label: "plate",
                                  keypoints: corners,
                                  keypointConfidences: Array(keypoints.conf.prefix(4)))
        }
    }
}
