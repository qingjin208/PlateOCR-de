import XCTest
import CoreImage
import CoreVideo
@testable import LicensePlateSDK

private struct MockDetector: LicensePlateDetecting {
    func detect(in image: CGImage) async throws -> [PlateDetection] {
        [PlateDetection(rect: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.1), confidence: 0.9, label: "license_plate"),
         PlateDetection(rect: CGRect(x: 0.6, y: 0.5, width: 0.3, height: 0.1), confidence: 0.8, label: "license_plate")]
    }
}
private struct MockOCR: PlateTextRecognizing {
    func recognize(in image: CGImage, mode: RecognitionMode) async throws -> [PlateObservation] {
        // OCR must receive each plate crop rather than the full input frame.
        XCTAssertLessThan(image.width, 400)
        XCTAssertLessThan(image.height, 100)
        return [PlateObservation(boundingBox: CGRect(x: 0, y: 0, width: 1, height: 1),
            string: "HH-AB 1234", confidence: 0.95, candidates: [])]
    }
}
final class FramePipelineTests: XCTestCase {
    private func buffer(width: Int = 1000, height: Int = 600, format: OSType = kCVPixelFormatType_32BGRA) throws -> CVPixelBuffer {
        var result: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, format,
            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &result)
        XCTAssertEqual(status, kCVReturnSuccess)
        let pixel = try XCTUnwrap(result)
        CIContext().render(CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: width, height: height)), to: pixel)
        return pixel
    }
    func testMultiplePlateCrops() async throws {
        let sdk = LicensePlateRecognizer(detector: MockDetector(), recognizer: MockOCR())
        let result = try await sdk.recognizeFrame(buffer())
        XCTAssertEqual(result.detections.count, 2)
        XCTAssertEqual(result.plates.count, 2)
        XCTAssertEqual(result.plates.first?.text, "HH-AB 1234")
        XCTAssertEqual(result.imageSize, CGSize(width: 1000, height: 600))
    }
    func testNV12AndOrientation() async throws {
        let sdk = LicensePlateRecognizer(detector: MockDetector(), recognizer: MockOCR())
        for format in [kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange] {
            let result = try await sdk.recognizeFrame(buffer(width: 200, height: 100, format: format), orientation: .right)
            XCTAssertEqual(result.imageSize, CGSize(width: 100, height: 200))
            XCTAssertEqual(result.plates.count, 2)
        }
    }
    func testRejectOversizedFrame() async throws {
        let sdk = LicensePlateRecognizer(detector: MockDetector(), recognizer: MockOCR())
        do {
            _ = try await sdk.recognizeFrame(buffer(width: 3842, height: 2160))
            XCTFail("Oversized frame should fail validation.")
        } catch LicensePlateError.invalidInput { }
    }
    func testPreservesCitySeparatorAndJoinsOCRFragments() {
        XCTAssertEqual(GermanPlateValidator.canonicalize("HH A 1234"), "HH-A 1234")
        let parts = ["HH", "AB", "1234"].enumerated().map { index, text in
            PlateObservation(boundingBox: CGRect(x: Double(index) / 3, y: 0, width: 0.3, height: 1),
                string: text, confidence: 0.9, candidates: [])
        }
        XCTAssertEqual(PlatePostProcessor().process(observations: parts, region: .zero,
            imageSize: CGSize(width: 100, height: 50))?.text, "HH-AB 1234")
    }
}
