import Foundation
import CoreGraphics
#if SWIFT_PACKAGE
import NativePlateDetector
#endif

/// Uses the original VINRecognizeSDK plate model; OCR is handled separately by Vision.
public final class PaddleLicensePlateDetector: LicensePlateDetecting, @unchecked Sendable {
    private let native: NativePlateDetector

    public init() throws {
        #if SWIFT_PACKAGE
        let bundle = Bundle.module
        #else
        let bundle = Bundle.main
        #endif
        guard let path = bundle.path(forResource: "yolo", ofType: "nb") else {
            throw LicensePlateError.modelNotFound("yolo.nb")
        }
        native = try NativePlateDetector(modelPath: path)
    }

    public func detect(in image: CGImage) async throws -> [PlateDetection] {
        try native.detectImage(image).map {
            PlateDetection(rect: $0.rect, confidence: $0.confidence, label: "license_plate")
        }
    }
}
