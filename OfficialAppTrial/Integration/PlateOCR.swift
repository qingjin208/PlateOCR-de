//
//  PlateOCR.swift
//  Drop-in OCR for the official Ultralytics YOLO iOS app (YOLOiOSApp target).
//
//  Pipeline: 4 pose keypoints (plate corners) -> CIPerspectiveCorrection -> Apple Vision OCR
//            -> German plate normalisation/validation.
//
//  Self-contained: depends only on UIKit / Vision / CoreImage (no ONNX, no external SDK).
//
//  Corner order expected: [leftTop, rightTop, rightBottom, leftBottom], in the
//  pixel coordinate space of the frame you pass in, with a TOP-LEFT origin
//  (i.e. Keypoints.xy converted to the frame's pixel size). The model emits its
//  4 keypoints in exactly this order (left_top, right_top, right_bottom, left_bottom).
//

import CoreImage
import Foundation
import UIKit
import Vision

/// One OCR pass over a perspective-corrected plate.
public struct PlateOCRResult {
    public let text: String
    public let confidence: Float
    public let correctedImage: UIImage?
}

/// Normalises/validates German plate text. Simplified vs. LicensePlateSDK's full
/// validator, but follows the same shape rules:
///   [city 1-3 letters] - [1-2 letters][1-4 digits][optional E/H]
public enum GermanPlateValidator {

    private static let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ"
    private static let allowed = letters + "0123456789-"

    /// Returns the canonical "CITY-NUMBER" form, or nil if it does not look like a plate.
    public static func normalize(_ raw: String) -> String? {
        var s = raw.uppercased()
        s = s.replacingOccurrences(of: " ", with: "")
        s = String(s.filter { allowed.contains($0) })
        guard !s.isEmpty else { return nil }

        // 1) Trust an explicit dash if both halves validate.
        if s.contains("-") {
            let parts = s.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false).map(String.init)
            if parts.count == 2, isValid(city: parts[0], rest: parts[1]) {
                return "\(parts[0])-\(parts[1])"
            }
        }

        // 2) No (or invalid) dash: try every city length 1...3.
        let compact = s.replacingOccurrences(of: "-", with: "")
        guard compact.count >= 3 else { return nil }
        for cityLen in 1...min(3, compact.count - 2) {
            let idx = compact.index(compact.startIndex, offsetBy: cityLen)
            let city = String(compact[..<idx])
            let rest = String(compact[idx...])
            if isValid(city: city, rest: rest) {
                return "\(city)-\(rest)"
            }
        }
        return nil
    }

    private static func isValid(city: String, rest: String) -> Bool {
        guard (1...3).contains(city.count),
              city.allSatisfy({ letters.contains($0) }),
              !rest.isEmpty,
              rest.range(of: "^[A-ZÄÖÜ]{1,2}[0-9]{1,4}[EH]?$", options: .regularExpression) != nil
        else { return false }

        // FZV: city + identification (excluding the E/H suffix) <= 8 characters.
        let hasSuffix = (rest.last == "E" || rest.last == "H")
        let restCore = hasSuffix ? String(rest.dropLast()) : rest
        return city.count + restCore.count <= 8
    }
}

/// Perspective-corrects the plate quad and runs Apple Vision OCR, throttled.
public final class PlateOCRService {

    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private let queue = DispatchQueue(label: "plate.ocr", qos: .userInitiated)
    private var lastRun = Date.distantPast
    private let throttle: TimeInterval

    public init(throttle: TimeInterval = 0.3) {
        self.throttle = throttle
    }

    /// corners: [leftTop, rightTop, rightBottom, leftBottom] in `frame` pixel space
    /// (top-left origin). `completion` is delivered on an arbitrary queue; hop to main
    /// before touching UI.
    public func recognize(frame: UIImage, corners: [CGPoint],
                          completion: @escaping (PlateOCRResult?) -> Void) {
        guard corners.count == 4 else { completion(nil); return }
        let now = Date()
        guard now.timeIntervalSince(lastRun) >= throttle else { completion(nil); return }
        lastRun = now

        queue.async { [weak self] in
            guard let self else { return }
            guard let corrected = self.perspectiveCorrect(frame: frame, corners: corners) else {
                completion(nil); return
            }
            self.runVisionOCR(on: corrected) { text, conf in
                guard let text, let plate = GermanPlateValidator.normalize(text) else {
                    completion(nil); return
                }
                completion(PlateOCRResult(text: plate, confidence: conf, correctedImage: corrected))
            }
        }
    }

    // MARK: - Perspective correction

    private func perspectiveCorrect(frame: UIImage, corners: [CGPoint]) -> UIImage? {
        guard let cg = frame.cgImage else { return nil }
        let ci = CIImage(cgImage: cg)
        let height = ci.extent.height

        // UIKit top-left origin -> Core Image bottom-left origin.
        func ciPoint(_ p: CGPoint) -> CIVector {
            CIVector(x: p.x, y: height - p.y)
        }

        guard let filter = CIFilter(name: "CIPerspectiveCorrection") else { return nil }
        filter.setValue(ci, forKey: kCIInputImageKey)
        filter.setValue(ciPoint(corners[0]), forKey: "inputTopLeft")
        filter.setValue(ciPoint(corners[1]), forKey: "inputTopRight")
        filter.setValue(ciPoint(corners[3]), forKey: "inputBottomLeft")
        filter.setValue(ciPoint(corners[2]), forKey: "inputBottomRight")

        guard let output = filter.outputImage else { return nil }
        let extent = output.extent
        // Reject degenerate quads (self-intersecting / collapsed) rather than warp garbage.
        guard extent.width >= 24, extent.height >= 8 else { return nil }

        // Upscale small crops so Vision sees enough pixels for text.
        let scale = max(1.0, 320.0 / extent.width)
        let scaled = output.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let cgOut = ciContext.createCGImage(scaled, from: scaled.extent) else { return nil }
        return UIImage(cgImage: cgOut)
    }

    // MARK: - Vision OCR

    private func runVisionOCR(on image: UIImage, completion: @escaping (String?, Float) -> Void) {
        guard let cg = image.cgImage else { completion(nil, 0); return }

        let request = VNRecognizeTextRequest { request, _ in
            guard let observations = request.results as? [VNRecognizedTextObservation],
                  !observations.isEmpty
            else { completion(nil, 0); return }

            let candidates = observations.compactMap { $0.topCandidates(1).first }
            let joined = candidates.map(\.string).joined(separator: " ")
            let bestConf = candidates.map(\.confidence).max() ?? 0
            completion(joined.isEmpty ? nil : joined, bestConf)
        }
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = false
        request.recognitionLanguages = ["de-DE", "en-US"]
        request.minimumTextHeight = 0.05

        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        do {
            try handler.perform([request])
        } catch {
            completion(nil, 0)
        }
    }
}
