import Foundation
import CoreGraphics
import CoreML

/// Reads tensor storage using its real type and strides instead of assuming float32.
private struct TensorValues {
    let array: MLMultiArray
    subscript(_ offset: Int) -> Float {
        switch array.dataType {
        case .float32: return array.dataPointer.assumingMemoryBound(to: Float.self)[offset]
        case .float16: return Float(array.dataPointer.assumingMemoryBound(to: Float16.self)[offset])
        case .double: return Float(array.dataPointer.assumingMemoryBound(to: Double.self)[offset])
        case .int32: return Float(array.dataPointer.assumingMemoryBound(to: Int32.self)[offset])
        @unknown default: return array[offset].floatValue
        }
    }
}

enum YOLOOutputDecoder {
    static func decode(_ array: MLMultiArray, config: YOLOConfig) throws -> [PlateDetection] {
        let shape = array.shape.map(\.intValue), channels = 4 + config.numClasses
        guard shape.count == 3, shape[0] == 1, shape[1] == channels || shape[2] == channels else {
            throw LicensePlateError.invalidModelOutput("Expected [1, C, N] or [1, N, C] YOLO output.")
        }
        let first = shape[1] == channels
        let count = first ? shape[2] : shape[1]
        let proposalStride = array.strides[first ? 2 : 1].intValue
        let channelStride = array.strides[first ? 1 : 2].intValue
        let values = TensorValues(array: array)
        var results: [PlateDetection] = []
        for index in 0..<count {
            func read(_ channel: Int) -> Float { values[index * proposalStride + channel * channelStride] }
            let coords = (0..<4).map(read)
            guard coords.allSatisfy(\.isFinite) else { continue }
            var score: Float = 0, classIndex = 0
            for label in 0..<config.numClasses {
                let value = read(4 + label)
                if value.isFinite, value > score, value <= 1 { score = value; classIndex = label }
            }
            guard score >= config.confidenceThreshold,
                  config.targetClassIndices.isEmpty || config.targetClassIndices.contains(classIndex) else { continue }
            let rect: CGRect
            switch config.format {
            case .xywhCenterPixels, .xywhCenterNormalized:
                let factor: CGFloat = config.format == .xywhCenterNormalized ? CGFloat(config.inputSize) : 1
                let cx = CGFloat(coords[0]) * factor, cy = CGFloat(coords[1]) * factor
                let width = CGFloat(coords[2]) * factor, height = CGFloat(coords[3]) * factor
                rect = CGRect(x: cx - width / 2, y: cy - height / 2, width: width, height: height)
            case .xyxyCornerNormalized:
                let side = CGFloat(config.inputSize)
                rect = CGRect(x: CGFloat(coords[0]) * side, y: CGFloat(coords[1]) * side,
                    width: CGFloat(coords[2] - coords[0]) * side, height: CGFloat(coords[3] - coords[1]) * side)
            }
            guard rect.width > 0, rect.height > 0 else { continue }
            results.append(PlateDetection(rect: rect, confidence: score,
                label: config.classNames.indices.contains(classIndex) ? config.classNames[classIndex] : "class \(classIndex)"))
        }
        return results
    }

    static func remap(_ detection: PlateDetection, scale: CGFloat, padX: CGFloat, padY: CGFloat,
                      imageSize: CGSize) -> PlateDetection {
        let x1 = max(0, min(1, (detection.rect.minX - padX) / scale / imageSize.width))
        let y1 = max(0, min(1, (detection.rect.minY - padY) / scale / imageSize.height))
        let x2 = max(0, min(1, (detection.rect.maxX - padX) / scale / imageSize.width))
        let y2 = max(0, min(1, (detection.rect.maxY - padY) / scale / imageSize.height))
        return PlateDetection(rect: CGRect(x: x1, y: y1, width: max(0, x2 - x1), height: max(0, y2 - y1)),
                              confidence: detection.confidence, label: detection.label)
    }

    static func suppress(_ proposals: [PlateDetection], threshold: Float) -> [PlateDetection] {
        var kept: [PlateDetection] = []
        for candidate in proposals.sorted(by: { $0.confidence > $1.confidence }) {
            if !kept.contains(where: { $0.label == candidate.label && iou($0.rect, candidate.rect) > threshold }) {
                kept.append(candidate)
            }
        }
        return kept
    }

    private static func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let intersection = a.intersection(b)
        guard !intersection.isNull, intersection.width > 0, intersection.height > 0 else { return 0 }
        let area = intersection.width * intersection.height
        let total = a.width * a.height + b.width * b.height - area
        return total > 0 ? Float(area / total) : 0
    }
}
