#if canImport(PaddleOCR)
/*
 License Plate OCR SDK — PaddleOCR recognizer (integration seam)

 Abstract:
 A `PlateTextRecognizing` implementation that reads the (already rectified) plate with
 PaddleOCR-iOS. It is disabled unless the PaddleOCR framework is linked, in which case
 `OCREngine.paddleOCR` uses it. The exact PaddleOCR-iOS class/method names differ by
 release, so the actual inference lives in `runPaddleOCR(on:)` — fill that in against
 your framework and replace the integration seam. See POSE_INTEGRATION.md.
 */

import Foundation
import CoreGraphics
import PaddleOCR

public final class PaddleOCRTextRecognizer: PlateTextRecognizing {

    public init() {}

    public func recognize(in image: CGImage, mode: RecognitionMode) async throws -> [PlateObservation] {
        // The plate image is already rectified by the keypoint corrector. Run PaddleOCR
        // detection + recognition on it and return the strongest reading.
        let (text, confidence, candidates) = try runPaddleOCR(on: image, mode: mode)

        // Box spans the whole rectified plate (normalized, top-left origin).
        let box = CGRect(x: 0, y: 0, width: 1, height: 1)
        return [PlateObservation(boundingBox: box, string: text, confidence: confidence, candidates: candidates)]
    }

    // MARK: - Integration seam

    /// Runs PaddleOCR on a rectified plate image.
    /// Contract: return the best line's text, its confidence, and alternative candidates.
    /// Replace the body with your PaddleOCR-iOS calls (det + rec + cls models from the bundle).
    private func runPaddleOCR(on image: CGImage, mode: RecognitionMode) throws -> (String, Float, [(String, Float)]) {
        // -------------------------------------------------------------------------
        // EXAMPLE SHAPE (adapt to your PaddleOCR-iOS version — names vary):
        //
        //   let predictor = PaddleOCR.Predictor(
        //       detModel: Bundle.main.url(forResource: "ch_PP-OCRv4_det", withExtension: "onnx")!,
        //       recModel: Bundle.main.url(forResource: "ch_PP-OCRv4_rec", withExtension: "onnx")!,
        //       clsModel: Bundle.main.url(forResource: "ch_ppocr_mobile_v2.0_cls", withExtension: "onnx")!
        //   )
        //   let lines = predictor.predict(image)        // -> [(text, score, box)]
        //   guard let best = lines.max(by: { $0.score < $1.score }) else {
        //       throw LicensePlateError.recognitionFailed("PaddleOCR 未识别到文本。")
        //   }
        //   let candidates = lines.map { ($0.text, Float($0.score)) }
        //   return (best.text, Float(best.score), candidates)
        // -------------------------------------------------------------------------
        throw LicensePlateError.recognitionFailed(
            "PaddleOCR 尚未完成集成：请在 PaddleOCRTextRecognizer.runPaddleOCR(on:) 中接入你的 PaddleOCR-iOS API，并替换 PaddleOCR.Predictor 类型与调用方式。详见 POSE_INTEGRATION.md。"
        )
    }
}
#endif
