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
