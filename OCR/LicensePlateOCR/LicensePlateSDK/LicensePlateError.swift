/*
 License Plate OCR SDK
 
 Abstract:
 Shared error types for the license plate detection + recognition pipeline.
 */

import Foundation

/// Errors that can be thrown by the license plate SDK.
public enum LicensePlateError: LocalizedError {
    /// The YOLO ONNX pose model could not be found in the app bundle.
    /// Keep `De-plate-pose.onnx` in `LicensePlateOCR/LicensePlateSDK/Resources/`
    /// and rebuild (Xcode 16's synchronized folders copy it automatically).
    case modelNotFound(String)
    /// The model's output could not be interpreted as a YOLO tensor.
    case invalidModelOutput(String)
    /// Text recognition (Vision) failed.
    case recognitionFailed(String)
    /// The provided media could not be read.
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "找不到车牌检测模型「\(name)」。请确认 LicensePlateOCR/LicensePlateSDK/Resources/ 下的 ONNX 模型已随 App 打包（Xcode 16 同步文件夹会自动拷贝）。"
        case .invalidModelOutput(let detail):
            return "YOLO 模型输出格式无法解析：\(detail)"
        case .recognitionFailed(let detail):
            return "文本识别失败：\(detail)"
        case .invalidInput(let detail):
            return "无效的输入媒体：\(detail)"
        }
    }
}
