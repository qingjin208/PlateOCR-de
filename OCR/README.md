# 德国车牌识别：车牌检测 + Apple Vision OCR

默认使用从 `VINRecognizeSDK` 项目 `feature/german-plate-external-frame-demo` 分支复制的 **`yolo.nb` 车牌检测模型**。检测后逐个裁剪车牌区域，交给 Apple Vision `RecognizeTextRequest` 识别字符，再校验和规范化德国车牌格式。

## 使用

用 Xcode 16 或更新版本打开 `LicensePlateOCR.xcodeproj`，确认签名配置，选择运行 iOS 18 或更新版本的 **arm64 iPhone 真机**运行。

- **图片**：选择图片后自动识别，按图片方向处理并显示车牌框和结果。
- **视频**：选择视频后通过 `AVAssetReader` 解码，默认最多采样 30 帧，返回去重后的车牌结果。
- **相机**：进入页面即可实时检测；先显示车牌检测框，再显示 OCR 文字和置信度，并在控制台打印结果。无需点击拍摄按钮。
- **模式**：`fast` / `accurate` 控制 Apple OCR 的识别级别，适用于图片、视频和实时相机。

实时相机默认使用 1080p NV12，最多每秒处理 5 帧。同一时间只处理一帧，忙时丢弃新帧，避免积压。实际速度取决于设备和每帧车牌数量；界面显示每次完整处理的耗时。离开相机页面或进入后台会停止相机，返回后恢复。

## SDK 外部帧输入

SDK 接收调用方自己的 `AVCaptureSession` 所提供的 **`CVPixelBuffer`**，支持 **BGRA、NV12 全范围、NV12 视频范围**，最大 **3840 × 2160**，也支持对应竖屏尺寸。SDK 本身不会打开相机或请求相机权限。

```swift
let sdk = LicensePlateRecognizer()
let frame = try await sdk.recognizeFrame(pixelBuffer, orientation: .up, mode: .fast)
for plate in frame.plates {
    print(plate.text, plate.confidence, plate.boundingBox)
}
```

`frame.detections` 是所有检测框；`frame.plates` 是通过德国格式校验的 OCR 结果。同一帧支持多个车牌。坐标均相对于按 `orientation` 处理后的画面，使用左上角原点和 0～1 归一化坐标。检测框可通过 `onDetections` 回调在 OCR 完成前显示。

如果相机输出已经旋转为正向画面，应传 `.up`，避免再次旋转。调用结束前不要修改或复用传入缓冲区的内存；实时调用应串行等待并丢弃忙时的新帧。

完整接口约定、接入示例、模型来源和验证步骤见 [EXTERNAL_FRAMES.md](EXTERNAL_FRAMES.md)。

## 项目文件

- `LicensePlateOCR/LicensePlateSDK/Resources/yolo.nb`：实际车牌检测模型。
- `NativePlateDetector/`：Paddle Lite 检测桥接、预处理、坐标还原和 NMS。
- `ThirdParty/PaddleLite/`：原分支的 Paddle Lite 运行库和头文件。
- `ThirdParty/model-provenance.json`：源提交、模型与运行库 SHA-256、预处理参数。
- `LicensePlateOCR/LicensePlateSDK/`：外部帧接口、Apple OCR 和德国车牌后处理。
- `LicensePlateOCR/Camera/`：由 Demo 持有的相机、实时框选和结果展示。
- `Tests/`：解码和帧处理回归测试。
- `Package.swift`：包含原生检测桥接的本地 Swift Package 定义。

## 平台与验证限制

复制的 Paddle Lite 二进制面向 iOS 真机，模拟器中的 Demo 会明确显示不支持的错误。当前 Swift Package 的 iOS 链接配置也要求真机目标；模拟器接入需要另行提供兼容的运行库。macOS 可使用注入的检测器运行测试，但不能执行这份 iOS 检测运行库。

模型、运行库和全部七个头文件已经与源分支进行字节一致性检查。当前开发环境为 Windows，尚未执行 Xcode 编译、Apple OCR 测试或 iPhone 真机验证；识别准确率、画面框位置、4K 输入和实时延迟需要按接入文档在真机上确认。

项目保留了原有 Core ML 检测器及 COCO 模型作为可选资源，默认程序已经使用 `yolo.nb`。替换成其他 Core ML 模型时可显式调用 `LicensePlateRecognizer(modelName:config:)`。
