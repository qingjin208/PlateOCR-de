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
        let parsed = PlatePostProcessor().process(observations: parts, region: .zero,
            imageSize: CGSize(width: 100, height: 50))
        XCTAssertEqual(parsed.first?.text, Optional("HH-AB 1234"))
    }

    func testCompletePlateWinsOverHigherConfidenceFragment() {
        let parts = [
            PlateObservation(boundingBox: CGRect(x: 0, y: 0.1, width: 0.2, height: 0.7), string: "HH", confidence: 0.8, candidates: []),
            PlateObservation(boundingBox: CGRect(x: 0.3, y: 0.2, width: 0.6, height: 0.7), string: "AB1234", confidence: 0.99, candidates: [])
        ]
        let result = PlatePostProcessor().process(observations: parts, region: .zero, imageSize: CGSize(width: 100, height: 50))
        XCTAssertEqual(result.first?.text, Optional("HH-AB 1234"))
        XCTAssertEqual(result.first?.confidence ?? 0, 0.8, accuracy: 0.0001)
    }

    func testEUStripAndDresdenPrefixAreDistinguished() {
        XCTAssertEqual(GermanPlateValidator.canonicalize("D HH AB 1234"), "HH-AB 1234")
        XCTAssertEqual(GermanPlateValidator.canonicalize("EU HH-AB 1234"), "HH-AB 1234")
        XCTAssertEqual(GermanPlateValidator.canonicalize("D AB 1234"), "D-AB 1234")
        XCTAssertNil(GermanPlateValidator.canonicalize("ABCDEF1234"))
        XCTAssertFalse(GermanPlateValidator.isValid("HH-AB 1234\n"))
    }

    func testProcessReturnsEveryDistinctPlateInOneCrop() {
        // Two separate plates read from a single detection crop must both be returned,
        // each with its own sub-box (not collapsed into one result).
        let parts = [
            PlateObservation(boundingBox: CGRect(x: 0.05, y: 0.2, width: 0.4, height: 0.6), string: "B-AB 1234", confidence: 0.9, candidates: []),
            PlateObservation(boundingBox: CGRect(x: 0.55, y: 0.2, width: 0.4, height: 0.6), string: "M-XY 99", confidence: 0.92, candidates: [])
        ]
        let results = PlatePostProcessor().process(observations: parts, region: CGRect(x: 0.1, y: 0.1, width: 0.8, height: 0.8),
                                                  imageSize: CGSize(width: 100, height: 50))
        XCTAssertEqual(results.count, 2)
        let texts = results.map(\.text).sorted()
        XCTAssertEqual(texts, ["B-AB 1234", "M-XY 99"])
        // Each result gets its own sub-box derived from its observation, not the full region.
        for r in results {
            XCTAssertNotEqual(r.boundingBox, CGRect(x: 0.1, y: 0.1, width: 0.8, height: 0.8))
        }
    }

    func testProcessDropsFragmentThatIsSubstringOfCompletePlate() {
        // "HH" + "AB1234" should assemble to "HH-AB 1234"; the fragment "A-B 1234"
        // must not be emitted alongside it.
        let parts = [
            PlateObservation(boundingBox: CGRect(x: 0, y: 0.1, width: 0.2, height: 0.7), string: "HH", confidence: 0.8, candidates: []),
            PlateObservation(boundingBox: CGRect(x: 0.3, y: 0.2, width: 0.6, height: 0.7), string: "AB1234", confidence: 0.99, candidates: [])
        ]
        let results = PlatePostProcessor().process(observations: parts, region: .zero, imageSize: CGSize(width: 100, height: 50))
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results.first?.text, "HH-AB 1234")
    }

    // MARK: - Confusion folding & registration-code preference

    func testConfusionFoldingIsPositionAware() {
        // A letter in the numeric part folds to a digit; a digit in the letter part folds
        // to a letter. Both must survive as the *same* plate.
        XCTAssertEqual(GermanPlateValidator.canonicalize("B AB I234"), "B-AB 1234")
        XCTAssertEqual(GermanPlateValidator.canonicalize("B AB 12S4"), "B-AB 1254")
        // …but folding must never be used to squeeze an over-long digit run into the format.
        XCTAssertNil(GermanPlateValidator.canonicalize("B AB 12345"))
        // The suffix is limited to the legal E / H markers.
        XCTAssertFalse(GermanPlateValidator.isValid("K-AB 123Z"))
        XCTAssertTrue(GermanPlateValidator.isValid("K-AB 123E"))
    }

    // MARK: - Same-frame duplicate boxes

    func testOverlappingDuplicatesOfOnePlateCollapse() {
        func plate(_ box: CGRect, _ confidence: Float) -> PlateResult {
            PlateResult(text: "HH-AB 1234", confidence: confidence, boundingBox: box,
                        imageSize: CGSize(width: 100, height: 50), candidates: [])
        }
        // A merged detection and the single plate inside it: IoU is low, containment is high.
        let merged = plate(CGRect(x: 0.10, y: 0.10, width: 0.40, height: 0.20), 0.80)
        let inner = plate(CGRect(x: 0.12, y: 0.12, width: 0.36, height: 0.16), 0.90)
        // A second plate of the same text somewhere else in the frame must survive.
        let elsewhere = plate(CGRect(x: 0.60, y: 0.60, width: 0.20, height: 0.10), 0.70)

        let kept = PlatePostProcessor.deduplicateOverlapping([merged, inner, elsewhere])
        XCTAssertEqual(kept.count, 2)
        // Highest confidence wins the collapsed pair, and the caller's order is preserved.
        XCTAssertEqual(kept.first?.confidence ?? 0, 0.90, accuracy: 0.0001)
        XCTAssertEqual(kept.last?.boundingBox, elsewhere.boundingBox)
    }

    // MARK: - Enhanced mode (opt-in, `RecognitionEnhancements`)

    func testEnhancedScoringKeepsShortNumericPlatesWithSuffixIntact() {
        // Legacy: `AB 123E` and `A` + `8123E` score the same, and the shorter series is
        // generated first — so the second series letter is folded into the digits.
        XCTAssertEqual(GermanPlateValidator.canonicalize("B AB 123E"), "B-A 8123E")
        // Enhanced: every fold costs points, so the literal reading wins.
        XCTAssertEqual(GermanPlateValidator.canonicalize("B AB 123E", enhancements: .on), "B-AB 123E")
        XCTAssertEqual(GermanPlateValidator.canonicalize("K AB 123H", enhancements: .on), "K-AB 123H")
        XCTAssertEqual(GermanPlateValidator.canonicalize("M AB 12H", enhancements: .on), "M-AB 12H")
        XCTAssertEqual(GermanPlateValidator.canonicalize("HH AB 123E", enhancements: .on), "HH-AB 123E")
    }

    func testEnhancedScoringLeavesAlreadyCorrectReadingsAlone() {
        for raw in ["B AB 1234", "HH AB 1234", "MAB1234", "D AB 1234", "D HH AB 1234",
                    "EU HH-AB 1234", "B AB 1234E", "B AB I234", "B AB 12S4", "SB AB 1234"] {
            XCTAssertEqual(GermanPlateValidator.canonicalize(raw, enhancements: .on),
                           GermanPlateValidator.canonicalize(raw),
                           "enhanced mode changed the reading of \(raw)")
        }
    }

    func testEnhancedStrictNumericPartRejectsLeadingZero() {
        // FZV §10 numbers run 1…9999: a leading zero is always a misread glyph.
        XCTAssertEqual(GermanPlateValidator.canonicalize("B AB 0123"), "B-AB 0123")
        XCTAssertNil(GermanPlateValidator.canonicalize("B AB 0123", enhancements: .on))
        XCTAssertEqual(GermanPlateValidator.canonicalize("B AB 1000", enhancements: .on), "B-AB 1000")
    }

    func testEnhancedMasterSwitchGatesEverySubSwitch() {
        // Sub-switches alone must not change anything while the master switch is off.
        let gated = RecognitionEnhancements(isEnabled: false, foldAwareScoring: true,
                                            strictNumericPart: true, keypointQuads: true)
        XCTAssertEqual(GermanPlateValidator.canonicalize("B AB 123E", enhancements: gated), "B-A 8123E")
        XCTAssertEqual(GermanPlateValidator.canonicalize("B AB 0123", enhancements: gated), "B-AB 0123")
        XCTAssertFalse(gated.effectiveFoldAwareScoring)
        XCTAssertFalse(gated.effectiveStrictNumericPart)
        XCTAssertFalse(gated.effectiveKeypointQuads)
        XCTAssertEqual(RecognitionEnhancements.off.summary, "关闭")
    }

    func testEnhancedKeypointQuadsAttachToPlateThatOwnsTheCrop() {
        let corners = [CGPoint(x: 0.12, y: 0.12), CGPoint(x: 0.58, y: 0.10),
                       CGPoint(x: 0.58, y: 0.30), CGPoint(x: 0.12, y: 0.32)]
        // Vision reports one box filling the crop (slightly inset, as it usually does).
        let observation = PlateObservation(boundingBox: CGRect(x: 0.02, y: 0.05, width: 0.95, height: 0.9),
                                           string: "HH-AB 1234", confidence: 0.9, candidates: [])
        let region = CGRect(x: 0.1, y: 0.1, width: 0.5, height: 0.2)
        let size = CGSize(width: 1000, height: 500)

        let legacy = PlatePostProcessor().process(observations: [observation], region: region,
                                                  imageSize: size, corners: corners)
        XCTAssertNil(legacy.first?.corners)

        let enhanced = PlatePostProcessor().process(observations: [observation], region: region,
                                                    imageSize: size, corners: corners,
                                                    enhancements: .on)
        XCTAssertEqual(enhanced.first?.corners, corners)
    }

    func testEnhancedKeepsDetectionBoxWhenOCRInputWasRectified() throws {
        // OCR on a perspective-corrected image: the observation box has no linear relation
        // to the frame, so the region owner must keep the detection's own box.
        let context = CIContext()
        let extent = CGRect(x: 0, y: 0, width: 40, height: 12)
        let corrected = try XCTUnwrap(context.createCGImage(
            CIImage(color: CIColor(red: 1, green: 1, blue: 1)).cropped(to: extent), from: extent))
        let observation = PlateObservation(boundingBox: CGRect(x: 0.05, y: 0.1, width: 0.9, height: 0.8),
                                           string: "HH-AB 1234", confidence: 0.9, candidates: [])
        let region = CGRect(x: 0.2, y: 0.2, width: 0.4, height: 0.12)
        let results = PlatePostProcessor().process(observations: [observation], region: region,
                                                   imageSize: CGSize(width: 1000, height: 500),
                                                   correctedImage: corrected, enhancements: .on)
        XCTAssertEqual(results.first?.boundingBox, region)
        XCTAssertEqual(results.first?.text, "HH-AB 1234")
    }
}
