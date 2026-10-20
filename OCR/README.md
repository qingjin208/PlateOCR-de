# 德国车牌识别：车牌检测 + Apple Vision OCR

当前 `feature/coreml-plate-detection` 分支使用由原 **`yolo.nb` 计算图和权重转换的 `license_plate_yolo.mlmodel`**，通过 Apple Core ML 执行车牌检测。129 个 FP32 参数张量与原模型逐字节一致，没有替换检测模型。检测后逐个裁剪车牌区域，继续交给 Apple Vision `RecognizeTextRequest` 识别字符。

> **新增：德国车牌关键点（Pose）检测 + 可切换 OCR**。已接入训练好的 `De-plate-pose-500.onnx`（4 角点关键点模型），通过 ONNX Runtime 推理，用 4 角点做透视矫正后再 OCR；App 可在 **Apple Vision / PaddleOCR** 间选择引擎并传入 SDK。详见 [POSE_INTEGRATION.md](POSE_INTEGRATION.md)。默认构建仍走原 Core ML 检测器，链接 `onnxruntime` 后自动改用 pose 模型。

`main` 保留原 Paddle Lite / `.nb` 方案，基准提交为 `9ec6f24`。本分支不链接 Paddle Lite，也不向 App 打包 `.nb`；原 `.nb` 仅保留在 `Models/source/` 作为离线转换输入。

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

- `LicensePlateOCR/LicensePlateSDK/Resources/license_plate_yolo.mlmodel`：由原 `.nb` 转换的 Core ML 车牌检测模型。
- `Models/source/yolo.nb`：原模型，仅供离线转换和一致性验证。
- `Models/nb-coreml-provenance.json`：模型 SHA-256、原始参数映射和转换元数据。
- `Models/nb-coreml-validation.json`：CPU 图计算一致性验证结果。
- `Scripts/convert_nb_to_coreml.py`：保留原权重的转换脚本。
- `Scripts/validate_nb_coreml.py`：原图与导出图的独立 CPU 验证脚本。
- `LicensePlateOCR/LicensePlateSDK/`：外部帧接口、Apple OCR 和德国车牌后处理。
- `LicensePlateOCR/Camera/`：由 Demo 持有的相机、实时框选和结果展示。
- `Tests/`：解码和帧处理回归测试。
- `Package.swift`：仅依赖 Apple 框架的 Swift Package 定义。

## 平台与验证限制

模型加载配置为 `MLComputeUnits.all`，允许 Core ML 选择符合条件的 CPU、GPU 和 Apple Neural Engine。是否实际使用 Neural Engine，以及速度和功耗是否改善，需要在目标 iPhone 上测量；不能仅凭模型格式作保证。Swift Package 已移除真机专用 Paddle 二进制限制，实际平台编译仍需 Apple 工具链验证。

已验证全部 129 个原始参数张量逐字节一致，且原 NB 计算图与序列化 Core ML 图在黑图、114 灰图和固定随机图上的独立 CPU 执行结果通过误差检查。这里的验证不使用实际 Paddle Lite 或 Apple Core ML 运行时，不能代替真机识别与性能测试。

当前环境为 Windows，尚未执行 Xcode 编译和 Swift 测试，也尚未验证 Core ML 真机推理。已加入实际模型加载测试及原图输出对照数据。转换方法、数值验证边界和真机对比步骤见 [Models/README.md](Models/README.md)。
