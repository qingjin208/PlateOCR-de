import XCTest
import CoreML
@testable import LicensePlateSDK

final class YOLODecoderTests: XCTestCase {
    private func array(first: Bool, type: MLMultiArrayDataType = .float32) throws -> MLMultiArray {
        let values: [[Float]] = [[320, 320, 160, 40, 0.9], [322, 320, 160, 40, 0.8],
                                [120, 280, 80, 30, 0.7], [320, 320, 160, 40, 0.2]]
        let output = try MLMultiArray(shape: first ? [1, 5, 4] : [1, 4, 5], dataType: type)
        for p in 0..<4 {
            for c in 0..<5 {
                output[first ? [0, NSNumber(value: c), NSNumber(value: p)] : [0, NSNumber(value: p), NSNumber(value: c)]] = NSNumber(value: values[p][c])
            }
        }
        return output
    }

    func testPixelCoordinatesBothLayoutsAndStorageTypes() throws {
        for first in [true, false] {
            for type in [MLMultiArrayDataType.float32, .float16, .double] {
                let proposals = try YOLOOutputDecoder.decode(array(first: first, type: type), config: .germanPlate)
                XCTAssertEqual(proposals.count, 3)
                XCTAssertEqual(proposals[0].rect.minX, 240, accuracy: 0.01)
                XCTAssertEqual(proposals[0].rect.minY, 300, accuracy: 0.01)
                let boxes = proposals.map {
                    YOLOOutputDecoder.remap($0, scale: 1.0 / 3, padX: 0, padY: 140,
                        imageSize: CGSize(width: 1920, height: 1080))
                }
                let kept = YOLOOutputDecoder.suppress(boxes, threshold: 0.45)
                XCTAssertEqual(kept.count, 2)
                XCTAssertEqual(kept[0].rect.minX, 0.375, accuracy: 0.0001)
                XCTAssertEqual(kept[0].rect.minY, 160.0 / 360, accuracy: 0.0001)
            }
        }
    }

    func testClipsEdgesAndRejectsPaddingOnlyBoxes() {
        let box = PlateDetection(rect: CGRect(x: -40, y: 100, width: 80, height: 80), confidence: 0.9, label: "license_plate")
        let clipped = YOLOOutputDecoder.remap(box, scale: 1.0 / 3, padX: 0, padY: 140,
            imageSize: CGSize(width: 1920, height: 1080))
        XCTAssertEqual(clipped.rect.minX, 0)
        XCTAssertEqual(clipped.rect.minY, 0)
        XCTAssertEqual(clipped.rect.width, 0.0625, accuracy: 0.0001)
        let padding = PlateDetection(rect: CGRect(x: 100, y: 20, width: 40, height: 40), confidence: 0.9, label: "license_plate")
        XCTAssertEqual(YOLOOutputDecoder.remap(padding, scale: 1.0 / 3, padX: 0, padY: 140,
            imageSize: CGSize(width: 1920, height: 1080)).rect.height, 0)
    }

    // MARK: - Pose (keypoint) decoding — parity with the official Ultralytics iOS app
    //
    // The official `PostProcessPose` derives `featureCount = shape[1] - 5`,
    // `numKeypoints = featureCount / 3`, and reads each keypoint at channel `5 + 3*k`
    // (x, y, visibility). This test locks that exact behaviour for our 4-corner German
    // plate model (`[1, 17, N]`) so a future refactor cannot silently break keypoints.

    /// Builds a `[1, 17, N]` (or `[1, N, 17]`) tensor for the pose model.
    /// Proposal 0 is confident (conf 0.9); proposal 1 is below the 0.25 gate and must be
    /// dropped — exercising both the keypoint read and the confidence filter.
    private func poseArray(first: Bool, type: MLMultiArrayDataType = .float32) throws -> MLMultiArray {
        // cx, cy, w, h, conf, k0(x,y,v), k1(x,y,v), k2(x,y,v), k3(x,y,v)
        let proposals: [[Float]] = [
            [320, 320, 400, 200, 0.9, 120, 220, 0.99, 520, 220, 0.98, 520, 420, 0.97, 120, 420, 0.96],
            [100, 100, 50, 50, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        let C = 17, N = proposals.count
        let output = try MLMultiArray(
            shape: first ? [1, NSNumber(value: C), NSNumber(value: N)]
                        : [1, NSNumber(value: N), NSNumber(value: C)],
            dataType: type)
        for p in 0..<N {
            for c in 0..<C {
                output[first ? [0, NSNumber(value: c), NSNumber(value: p)]
                              : [0, NSNumber(value: p), NSNumber(value: c)]] = NSNumber(value: proposals[p][c])
            }
        }
        return output
    }

    func testPoseDecodeReadsFourKeypointsAndRemapsCorrectly() throws {
        for first in [true, false] {
            let decoded = try YOLOOutputDecoder.decode(try poseArray(first: first), config: .germanPlatePose)
            XCTAssertEqual(decoded.count, 1, "only the conf=0.9 proposal survives the 0.25 gate")
            let d = decoded[0]
            // Decode produces keypoints in the 640 input-pixel space, same as the box — the
            // official `PostProcessPose` reads x,y at `5 + 3*k` and never normalizes them.
            guard let raw = d.keypoints, raw.count == 4 else { XCTFail("expected 4 raw keypoints"); return }
            XCTAssertEqual(raw[0].x, 120, accuracy: 1e-3)   // left_top
            XCTAssertEqual(raw[0].y, 220, accuracy: 1e-3)
            XCTAssertEqual(raw[1].x, 520, accuracy: 1e-3)   // right_top
            XCTAssertEqual(raw[3].y, 420, accuracy: 1e-3)   // left_bottom
            XCTAssertEqual(d.keypointConfidences?[0], 0.99, accuracy: 1e-3)

            // Remap with gain 0.5, padX 0, padY 140 on a 1280x720 original (mirrors the
            // PC reference's synthetic check): the official `inputPoint(fromModelPoint:)`
            // does exactly `(p - pad) / gain`, then `/ imageSize`.
            let remapped = YOLOOutputDecoder.remap(d, scale: 0.5, padX: 0, padY: 140,
                imageSize: CGSize(width: 1280, height: 720))
            guard let kpts = remapped.keypoints, kpts.count == 4 else { XCTFail("expected 4 remapped keypoints"); return }
            // Order must be left_top, right_top, right_bottom, left_bottom.
            XCTAssertEqual(kpts[0].x, 0.1875, accuracy: 1e-4)  // LT  (120/0.5/1280)
            XCTAssertEqual(kpts[0].y, 0.2222, accuracy: 1e-4)  //      ((220-140)/0.5/720)
            XCTAssertEqual(kpts[1].x, 0.8125, accuracy: 1e-4)  // RT
            XCTAssertEqual(kpts[2].x, 0.8125, accuracy: 1e-4)  // RB
            XCTAssertEqual(kpts[2].y, 0.7778, accuracy: 1e-4)  //      ((420-140)/0.5/720)
            XCTAssertEqual(kpts[3].y, 0.7778, accuracy: 1e-4)  // LB
        }
    }

    func testCornerOrderReordersToLeftTopRightTopRightBottomLeftBottom() throws {
        // Same four positions but fed in a scrambled order (RT, RB, LB, LT). `cornerOrder`
        // re-derives the canonical LT→RT→RB→LB permutation from the centroid, exactly like
        // the PC reference's `determine_relative_positions`, so a mis-ordered model read is
        // corrected instead of producing a bow-tie quad.
        let scrambled: [CGPoint] = [
            CGPoint(x: 0.8125, y: 0.2222),  // right_top
            CGPoint(x: 0.8125, y: 0.7778),  // right_bottom
            CGPoint(x: 0.1875, y: 0.7778),  // left_bottom
            CGPoint(x: 0.1875, y: 0.2222),  // left_top
        ]
        let order = try XCTUnwrap(YOLOOutputDecoder.cornerOrder(scrambled))
        XCTAssertEqual(order, [3, 0, 1, 2])  // maps scrambled -> [LT, RT, RB, LB]
        let sorted = order.map { scrambled[$0] }
        XCTAssertEqual(sorted[0].x, 0.1875, accuracy: 1e-6)  // LT first
        XCTAssertEqual(sorted[0].y, 0.2222, accuracy: 1e-6)
        XCTAssertEqual(sorted[3].y, 0.7778, accuracy: 1e-6)  // LB last
    }

    func testMalformedOutputFailsExplicitly() throws {
        let malformed = try MLMultiArray(shape: [1, 4, 8400], dataType: .float32)
        XCTAssertThrowsError(try YOLOOutputDecoder.decode(malformed, config: .germanPlate))
    }

    func testOriginalConvertedModelCompilesAndLoads() throws {
        XCTAssertNoThrow(try CoreMLLicensePlateDetector(computeUnits: .cpuOnly))
    }

    func testAppleCPUPredictionAgainstOriginalGraphFixtures() async throws {
        let detector = try CoreMLLicensePlateDetector(computeUnits: .cpuOnly)
        for (name, gray) in [("nb-reference-black", UInt8(0)), ("nb-reference-gray", UInt8(114))] {
            var pixel: CVPixelBuffer?
            XCTAssertEqual(CVPixelBufferCreate(kCFAllocatorDefault, 640, 640, kCVPixelFormatType_32BGRA,
                nil, &pixel), kCVReturnSuccess)
            let buffer = try XCTUnwrap(pixel)
            CVPixelBufferLockBaseAddress(buffer, [])
            if let base = CVPixelBufferGetBaseAddress(buffer)?.assumingMemoryBound(to: UInt8.self) {
                let row = CVPixelBufferGetBytesPerRow(buffer)
                for y in 0..<640 {
                    for x in 0..<640 {
                        let offset = y * row + x * 4
                        base[offset] = gray; base[offset + 1] = gray; base[offset + 2] = gray; base[offset + 3] = 255
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "f32"))
            let bytes = try Data(contentsOf: url)
            XCTAssertEqual(bytes.count, 5 * 8400 * 4)
            let output = try await detector.prediction(on: buffer)
            XCTAssertEqual(output.shape.map(\.intValue), [1, 5, 8400])
            var maxCoordinateError: Float = 0, maxScoreError: Float = 0
            bytes.withUnsafeBytes { raw in
                for channel in 0..<5 {
                    for anchor in 0..<8400 {
                        let offset = (channel * 8400 + anchor) * 4
                        let bits = UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
                        let expected = Float(bitPattern: bits)
                        let actual = output[[0, NSNumber(value: channel), NSNumber(value: anchor)]].floatValue
                        if channel == 4 { maxScoreError = max(maxScoreError, abs(actual - expected)) }
                        else { maxCoordinateError = max(maxCoordinateError, abs(actual - expected)) }
                    }
                }
            }
            XCTAssertLessThan(maxCoordinateError, 0.02, name)
            XCTAssertLessThan(maxScoreError, 0.0001, name)
        }
    }
}
