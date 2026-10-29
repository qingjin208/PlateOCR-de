/*
 License Plate OCR SDK
 
 Abstract:
 Shared error types for the license plate detection + recognition pipeline.
 */

import Foundation

/// Errors that can be thrown by the license plate SDK.
public enum LicensePlateError: LocalizedError {
    /// The YOLO Core ML model could not be found in the app bundle.
    /// Drop your exported `*.mlmodelc` (or `*.mlpackage`) into
    /// `LicensePlateOCR/LicensePlateSDK/Resources/` and rebuild.
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
            return "找不到车牌检测模型「\(name)」。请在 Mac 上用 Scripts/export_pose_coreml.py 导出 De-plate-pose-500.mlpackage 并放进 LicensePlateSDK/Resources/（文件夹同步目录会自动打包）。"
        case .invalidModelOutput(let detail):
            return "YOLO 模型输出格式无法解析：\(detail)"
        case .recognitionFailed(let detail):
            return "文本识别失败：\(detail)"
        case .invalidInput(let detail):
            return "无效的输入媒体：\(detail)"
        }
    }
}
