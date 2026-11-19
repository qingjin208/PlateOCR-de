# 德国车牌识别：车牌检测 + PaddleOCR 字符识别

当前分支使用 **LibreYOLO（YOLO-NAS-pose）导出的 `De-plate-pose.onnx`**（4 角点关键点模型），通过 ONNX Runtime 执行车牌检测——这是 SDK 唯一的检测器。检测后用 4 角点做透视矫正，再交给 **PaddleOCR PP-OCRv5 mobile 识别模型**（同样经 ONNX Runtime 推理）读取字符；Apple Vision `RecognizeTextRequest` 保留为可切换的 A/B 对照引擎。旧 Core ML 检测器（`license_plate_yolo.mlmodel`）已随本分支退役并从包中移除。

> **本分支变更：OCR 引擎由 Apple Vision 替换为 PaddleOCR**。字符识别不再调用 Vision 的 `.fast/.accurate` 两档，而是把矫正后的整行车牌送入 `PP-OCRv5_rec_mobile_infer.onnx` + `ppocrv5_dict.txt`，用受德国车牌字符集约束的 CTC 束搜索生成主结果与候选。关键点检测、透视矫正、图像清洁与德国车牌校验链路完全不变。详见 [POSE_INTEGRATION.md](POSE_INTEGRATION.md) 与 `Scripts/paddleocr_rec_reference.py`。链接 `onnxruntime`（`pod install`）后 pose 检测与 PaddleOCR 同时启用；未链接时 App 仍可启动，但检测器为不可用桩并在界面上显示原因。

`main` 保留原 Paddle Lite / `.nb` 方案，基准提交为 `9ec6f24`。本分支不链接 Paddle Lite，也不向 App 打包 `.nb`；原 `.nb` 仅保留在 `Models/source/` 作为离线转换输入。

## 使用

用 Xcode 16 或更新版本打开 `LicensePlateOCR.xcodeproj`，确认签名配置，选择运行 iOS 18 或更新版本的 **arm64 iPhone 真机**运行。

- **图片**：选择图片后自动识别，按图片方向处理并显示车牌框和结果。
- **视频**：选择视频后通过 `AVAssetReader` 解码，默认最多采样 60 帧；结果按文本多数票去重 —— 至少两帧一致才列出，只出现一帧的读数需要置信度 ≥ 0.85 才会显示，单帧误读不会单独成行。
- **相机**：点「开始实时识别」后持续检测；先显示车牌检测框，再显示 OCR 文字和置信度。历史列表按同一车牌跨帧投票合并成一行（`票数/总帧数`），不会把同一辆车的每次识别都记一条。
- **模式**：`fast` / `accurate` 在 PaddleOCR 下映射为「单次推理」与「多裁剪 + 更宽 CTC 候选搜索」（切到 Apple Vision 时仍是 `recognitionLevel` 两档），适用于图片、视频和实时相机。

实时相机默认使用 1080p NV12，最多每秒处理 5 帧。同一时间只处理一帧，忙时丢弃新帧，避免积压。实际速度取决于设备和每帧车牌数量；界面显示每次完整处理的耗时。离开相机页面或进入后台会停止相机，返回后恢复。

## SDK 输入：统一 CVPixelBuffer（默认 NV12）

App → SDK 的图像输入**只有 `CVPixelBuffer` 一种格式**，支持 **NV12 全范围（默认）**、**NV12 视频范围**、**BGRA**，最大 **3840 × 2160**（含对应竖屏尺寸）。`CGImage` / `Data` / `URL` 都不作为图像载荷跨过这条边界。SDK 本身不会打开相机或请求相机权限。

相机路径天然就是 NV12，直接把裸缓冲区交给 `recognizeFrame`；相册静态图和视频抽帧会得到 `CGImage`，由 SDK 的工厂统一包成 NV12 缓冲区后走同一入口（可用 `inputPixelFormat` 切到 BGRA）。

```swift
let sdk = LicensePlateRecognizer()   // inputPixelFormat 默认 .nv12

// 相机：原生 NV12 缓冲区，零转换
let live = try await sdk.recognizeFrame(pixelBuffer, orientation: .right,
                                        mode: .fast, tiled: false, onDetections: nil)

// 相册：把摆正的 CGImage 包成 SDK 输入缓冲区（默认 NV12）
let buffer = try sdk.makeInputBuffer(from: uprightCGImage)
let still = try await sdk.recognize(in: buffer, orientation: .up, mode: .fast, tiled: true)

// 需要 BGRA 的生产方：切换全局输入格式，或用工厂显式指定
sdk.inputPixelFormat = .bgra
let bgra = try PlatePixelBufferFactory.makeBuffer(from: uprightCGImage, format: .bgra)
```

`frame.detections` 是所有检测框；`frame.plates` 是通过德国格式校验的 OCR 结果。同一帧支持多个车牌。坐标均相对于按 `orientation` 处理后的画面，使用左上角原点和 0～1 归一化坐标。检测框可通过 `onDetections` 回调在 OCR 完成前显示。

如果相机输出已经旋转为正向画面，应传 `.up`，避免再次旋转。调用结束前不要修改或复用传入缓冲区的内存；实时调用应串行等待并丢弃忙时的新帧。视频方法的 `URL` 仅作为**资源定位符**——抽帧后照样转成 CVPixelBuffer 再进入识别入口。

完整接口约定、接入示例、模型来源和验证步骤见 [EXTERNAL_FRAMES.md](EXTERNAL_FRAMES.md)。

## PaddleOCR 精度评测与诊断

`Scripts/paddleocr_rec_reference.py` 与 SDK 的预处理 / CTC 解码同构，可在 PC 上复核矫正图并批量打分：

```bash
python Scripts/paddleocr_rec_reference.py --self-test
python Scripts/paddleocr_rec_reference.py rectified.png --mode fast
python Scripts/paddleocr_rec_reference.py --eval samples/ --labels samples/labels.tsv
```

真值来自 TSV（`文件名<TAB>期望车牌`）或图片文件名，输出准确率与字符错误率（CER）。**真机 A/B**：用同一批矫正图在设置页分别切换 PaddleOCR / Apple Vision 各记录一次读数，即可量化替换收益。目前仓库只有 4 张 Arial 合成样本的自检结果，尚无真实 FE-Schrift 车牌的精度证据。

两点容易混淆：

- `ppocrv5_dict.txt`（18,383 行）是模型 18,385 维输出头的 **index → 字符映射表**，不是候选清单，禁止裁剪成德国车牌字符集（`B` 在字典中是 class 16180，裁完会索引越界）。Swift 侧 `PlateOCRAlphabet` 只是 **CTC 解码约束**（41 字符 + blank）；物理裁剪输出头需改 ONNX 权重，且实测只提速、不提升判别精度。
- 打开设置页「诊断」后，检测信息行会额外显示**全字典 greedy 原始读数**及异常字符。该读数故意不进入候选池：当它含 `@`、汉字等非车牌字符时，说明 EU 蓝条 / 年检章 / 边框没遮干净；受限束搜索会吞掉这些字符并给出一个看似合法却错误的读数，所以这个信号必须单独保留。

`accurate` 在 PC 实测约为 `fast` 的 4.7 倍耗时（最多 5 次推理），实时相机建议固定使用 `fast`。

## 项目文件

- `LicensePlateOCR/LicensePlateSDK/Resources/De-plate-pose.onnx`：LibreYOLO YOLO-NAS-pose 车牌 4 角点关键点模型（SDK 唯一检测器）。
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

本分支的检测器与 PaddleOCR 识别器都跑在 **ONNX Runtime（CPU）** 上，不使用 Core ML，
因此也用不上 Apple Neural Engine 的自动调度（`MLComputeUnits.all` 是已退役 Core ML 检测器的旧配置，
上面的旧描述已清理）。ORT 用 `setIntraOpNumThreads(2)` 限制线程数；Apple 文档说明 Core ML 会自动在
CPU / GPU / ANE 间择优，所以长时间相机推理的功耗与发热需在真机上与 Core ML 方案对比 —— 这是当前架构的权衡。

工程最低部署目标 iOS 18.0、`SWIFT_VERSION = 5.0`，用到的 `OSAllocatedUnfairLock`(16+)、
`Locale.Language`(16+)、`@Observable`(17+)、`RecognizeTextRequest`(18+) 均在覆盖范围内。
Apple 平台规范对齐情况（启动时间、协作线程池、后台执行、隐私清单、命名 sRGB / 共享 CIContext）
见 [MAC_SETUP.md](MAC_SETUP.md) 的「Apple 平台规范对齐」一节。

当前环境为 Windows，**Swift 从未经过 Xcode 编译**，也没有真机精度 / 帧率数据。
已完成的验证只有：PC 参考脚本 `--self-test` / `--eval`、Swift 静态结构检查、模型与字典规格核对。
`Models/` 下的 NB→Core ML 转换与逐字节一致性记录属于旧检测器，仅保留作离线溯源，不参与本分支运行时；
详见 [Models/README.md](Models/README.md)。
