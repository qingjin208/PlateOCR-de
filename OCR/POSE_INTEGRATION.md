# 德国车牌关键点（Pose）检测 + 可切换 OCR 集成说明

本说明覆盖把训练好的德国车牌关键点检测模型接入 SDK 的改动，以及让 App 用户在
**Apple Vision** 与 **PaddleOCR** 之间选择 OCR 引擎并传入 SDK 的方式。

## 1. 处理管线（每帧）

```
输入帧 (CVPixelBuffer / CGImage)
   │
   ├─ 检测：ONNXPoseLicensePlateDetector
   │     De-plate-pose-500.onnx  →  [1, 17, 8400]
   │     解码为：目标检测框 + 4 个角点关键点
   │     角点顺序（固定，来自 data.yaml）：
   │       0 left_top  1 right_top  2 right_bottom  3 left_bottom
   │     NMS + 重映射到原图归一化坐标（左上原点，0…1）
   │
   ├─ 矫正（每个车牌）：PerspectivePlateCorrector
   │     用 4 角点经 CIPerspectiveCorrection 透视矫正为正向矩形图
   │     → 这就是喂给 OCR 的「矫正后图片」
   │     角点不可靠（缺失 / 可见性<1.5）时回退到检测框裁剪
   │
   ├─ OCR（用户选定的引擎）：
   │     .appleVision  → PlateTextRecognizer (Apple Vision RecognizeTextRequest)
   │     .paddleOCR    → PaddleOCRTextRecognizer（集成缝，需链接 PaddleOCR-iOS）
   │
   ├─ 后处理 + 德国车牌校验（沿用 PlatePostProcessor / GermanPlateValidator）
   │
   └─ 返回：
         detections : 全部检测框（OCR 前即可经 onDetections 显示）
         plates     : 每项含 boundingBox(原检测框) + correctedPlateImage(矫正图) + text + confidence + candidates
```

关键点模型输出 `[1, 17, 8400]`：`5(框cx,cy,w,h + conf) + 4×3(每点 x,y,可见性)`。
解码 / NMS / 坐标重映射复用 `YOLOOutputDecoder`（现已支持关键点）。

## 2. 依赖接入

### 2.1 ONNX Runtime（pose 检测，必选）

模型已在仓库：`LicensePlateOCR/LicensePlateSDK/Resources/De-plate-pose-500.onnx`
（若需更新，替换该文件；资源会被 Swift Package 的 `.process("Resources")` 打包）。

在本机（macOS + Xcode）添加 `onnxruntime-mobile-objc`：

- CocoaPods：`pod 'onnxruntime-mobile-objc'`，或
- 手动：下载 Microsoft 官方 `onnxruntime` 的 iOS xcframework，加到 App target 的
  Frameworks 并 **Embed & Sign**；若通过 Swift Package 使用，请加一个 `binaryTarget`。

链接后 `ONNXPoseLicensePlateDetector.swift` 与 `LicensePlateRecognizer(poseModelName:)` 即被
`#if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc)` 启用，`ContentView` 会自动改用 pose 检测器。
（CocoaPods 给 `onnxruntime-mobile-objc` 生成的模块名通常是 `onnxruntime_mobile_objc`，代码已同时兼容两种名称。）
未链接时 App 回退到旧的 Core ML 检测器（直到接入 ONNX Runtime）。

> ORT Objective-C API 的方法名在不同版本间可能有微小差异（如 `ORTSession.init`、
> `run(withInputs:outputNames:runOptions:)`、`tensorData()`）。若编译报错，请按你使用的
> ORT 版本调整 `ONNXPoseLicensePlateDetector.swift` 中的调用；文档与示例已尽量对齐 1.17+/1.18+。

### 2.2 PaddleOCR-iOS（可选 OCR 引擎）

`PaddleOCRTextRecognizer.swift` 已遵循 `PlateTextRecognizing`，但其中的
`runPaddleOCR(on:)` 是**集成缝**：在链接 PaddleOCR-iOS 后，把里面的示例替换为你的
PaddleOCR-iOS 实际调用（det + rec + cls 模型，通常从 App bundle 读取）。

链接 PaddleOCR-iOS 后，`OCREngine.paddleOCR.makeRecognizer()` 会返回它；未链接时
`.paddleOCR` 自动回退为 Apple Vision（保证管线可运行）。建议：未接入前在 App UI 中
禁用 / 提示 PaddleOCR 选项。

## 3. 预处理注意事项（真机首次验证必读）

`ONNXPoseLicensePlateDetector` 的预处理：letterbox 到 640×640（114 灰边）、**不翻转**、
RGB 且 /255（NCHW）。Ultralytics 导出的 ONNX 默认期望这种 OpenCV 风格（左上原点、行0为图顶）。

- 若真机上检测整体错乱/镜像：把 `ONNXPoseLicensePlateDetector` 的 `useRGB` 改为 `false`
  （即 BGR）即可，是一行改动。
- `letterboxedUpright` 与旧 Core ML 检测器用的 `letterboxed`（带 y 翻转）**不同**，
  不可混用——pose 模型需要正向（非翻转）缓冲。

## 4. 用 Python 参考验证（无需 Xcode）

仓库已提供 `Scripts/pose_pipeline_reference.py`，它用 numpy 复刻了整套算法并做了合成校验
（解码顺序、NMS、坐标重映射、4 点单应矫正），可离线跑通：

```bash
python OCR/Scripts/pose_pipeline_reference.py            # 合成校验（无需模型）
python OCR/Scripts/pose_pipeline_reference.py path/to/german_plate.jpg   # 需 onnxruntime
```

如要跑真实推理，请先 `pip install onnxruntime numpy pillow`。

## 5. 真机验证清单

- [ ] 在 iPhone（iOS 18+）上用 Xcode 编译，确认 `De-plate-pose-500.onnx` 已打包。
- [ ] 图片 / 视频 / 相机三种来源都能返回检测框与车牌文本。
- [ ] 结果卡片显示「矫正图」（透视矫正后的车牌）；角点不可靠时应回退为裁剪图（无矫正图）。
- [ ] 切换 OCR 引擎 Picker（Apple Vision / PaddleOCR），确认 PaddleOCR 在未接入时回退、接入后生效。
- [ ] 一帧多车牌、倾斜 / 远处小牌、部分遮挡场景下的框与文本质量。
- [ ] 若检测异常，先尝试 `useRGB=false`（BGR）。

## 6. 关键文件

- `LicensePlateSDK/ONNXPoseLicensePlateDetector.swift` — ONNX Runtime pose 检测器
- `LicensePlateSDK/PerspectivePlateCorrector.swift` — 4 角点透视矫正
- `LicensePlateSDK/OCREngine.swift` — OCR 引擎枚举与工厂
- `LicensePlateSDK/PaddleOCRTextRecognizer.swift` — PaddleOCR 集成缝
- `LicensePlateSDK/LicensePlateRecognizer.swift` — 管线编排 + `correctedPlateImage` 返回
- `LicensePlateSDK/LicensePlateDetector.swift` / `YOLOOutputDecoder.swift` — 检测结构与关键点解码
- `LicensePlateOCR/ContentView.swift` — Demo 的 OCR 引擎选择器与矫正图展示
- `Scripts/pose_pipeline_reference.py` — 算法参考与数值校验
