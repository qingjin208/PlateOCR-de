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
RGB 且 /255（NCHW）。Ultralytics 导出并训练时用的就是 RGB（`Image.open(...).convert("RGB")`），
`Scripts/pose_pipeline_reference.py` 也是按 RGB 复刻的，因此**默认 `useRGB = true`（RGB）**。

- 只有当某个导出确实是在 BGR 流水线上训练/导出时，才把 `useRGB` 切到 `false`（App 内有
  「输入通道顺序」开关，可运行时切换，无需重编译）。**不要**因为「真机效果差」就盲切 BGR——
  这类症状更常见的原因是角点/几何或阈值问题。
- `letterboxedUpright` 与旧 Core ML 检测器用的 `letterboxed` 现在**都是正向（top-row-first）
  缓冲**（2026-10-27 修正）：`CGContext(data:...)` 的坐标原点在**左下角**，不翻转会把图像
  **上下颠倒**写进缓冲，于是模型看到的是垂直镜像的画面——这正是「PC 一下全对、真机不行」的根因。
  `flipY` 默认 `true`（= 翻转成正向，与 PC `canvas[pad_y:pad_y+nh] = resized` 一致）。
  App 调试面板有「Letterbox 缓冲朝向」开关，可真机 A/B 验证（框上下镜像 → 说明应为反向）。

## 4. 用 Python 参考验证（无需 Xcode）

仓库已提供 `Scripts/pose_pipeline_reference.py`，它用 numpy 复刻了整套算法并做了合成校验
（解码顺序、NMS、坐标重映射、4 点单应矫正），可离线跑通：

```bash
python OCR/Scripts/pose_pipeline_reference.py            # 合成校验（无需模型）
python OCR/Scripts/pose_pipeline_reference.py path/to/german_plate.jpg   # 需 onnxruntime
```

如要跑真实推理，请先 `pip install onnxruntime numpy pillow`。

## 5. 强化模式（Enhanced Mode）

SDK 的默认行为不变（`RecognitionEnhancements.off`）。下列三项改进被打包成一个可选开关，
由 App 决定是否传给 SDK；子开关用于在真机上单独定位回归。

```swift
// 建实例时打开
let recognizer = LicensePlateRecognizer(poseModelName: "De-plate-pose-500", enhancements: .on)
// 运行时切换（下一次识别即生效，不需要重建模型 / OCR）
recognizer.enhancements.isEnabled = false
recognizer.enhancements.foldAwareScoring = false   // 只关其中一项
```

| 子开关 | 作用 | 关闭（legacy） | 打开 |
|---|---|---|---|
| `foldAwareScoring` | 校验器按「折叠次数」扣分，避免把系列字母折进数字 | `B AB 123E` → `B-A 8123E`（错读） | `B-AB 123E` |
| `strictNumericPart` | 数字段拒绝前导 `0`（FZV §10 号码为 1…9999） | `B-AB 0123` 被接受 | 拒绝该解读 |
| `keypointQuads` | 占满整块裁剪的车牌继承 pose 角点；矫正图 OCR 后不再线性反算框 | 图片上只画轴对齐框 | 叠加 pose 四边形 |

折叠感知排序修的是确定性错读：系列第 2 字母 ∈ {B, S, Z, G, T, I, O, D, L} 且数字只有
1~3 位并带 `E`/`H` 时，legacy 打分下「系列 1 字母 + 折叠出的 4 位数字」与正确解读同分，
而短系列先生成 → 必错（`B AB 1234E` 只因位数刚好凑满 4 位才侥幸正确）。

Demo App 在控制面板提供「强化识别（SDK Enhanced Mode）」总开关 + 三个子开关；
下方诊断行 `强化模式（SDK 侧）` 打印 SDK 实例真正持有的值，用来确认开关确实到达了 SDK。

## 6. 真机验证清单

- [ ] 在 iPhone（iOS 18+）上用 Xcode 编译，确认 `De-plate-pose-500.onnx` 已打包。
- [ ] 图片 / 视频 / 相机三种来源都能返回检测框与车牌文本。
- [ ] 结果卡片显示「矫正图」（透视矫正后的车牌）；角点不可靠时应回退为裁剪图（无矫正图）。
- [ ] 切换 OCR 引擎 Picker（Apple Vision / PaddleOCR），确认 PaddleOCR 在未接入时回退、接入后生效。
- [ ] 一帧多车牌、倾斜 / 远处小牌、部分遮挡场景下的框与文本质量。
- [ ] 打开 / 关闭「强化识别」，确认 `B AB 123E` 类读数与图片上的四边形叠层随之变化。
- [ ] 若检测异常，先看 UI 上的「检测器」诊断行确认真的走到了 ONNX pose（而非 Core ML 回退）；
      再看结果里是否有矫正图（没有则角点/几何校验被拒，回退成了框裁剪）。

## 7. 关键文件

- `LicensePlateSDK/ONNXPoseLicensePlateDetector.swift` — ONNX Runtime pose 检测器
- `LicensePlateSDK/PerspectivePlateCorrector.swift` — 4 角点透视矫正
- `LicensePlateSDK/OCREngine.swift` — OCR 引擎枚举与工厂
- `LicensePlateSDK/PaddleOCRTextRecognizer.swift` — PaddleOCR 集成缝
- `LicensePlateSDK/LicensePlateRecognizer.swift` — 管线编排 + `correctedPlateImage` 返回
- `LicensePlateSDK/RecognitionEnhancements.swift` — 强化模式开关（总开关 + 3 个子开关）
- `LicensePlateSDK/GermanPlateValidator.swift` — 德国车牌校验 + 折叠感知排序
- `LicensePlateSDK/LicensePlateDetector.swift` / `YOLOOutputDecoder.swift` — 检测结构与关键点解码
- `LicensePlateOCR/ContentView.swift` — Demo 的 OCR 引擎选择器与矫正图展示
- `Scripts/pose_pipeline_reference.py` — 算法参考与数值校验
