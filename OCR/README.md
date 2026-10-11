# 德国车牌识别 OCR（YOLO 检测 + Apple Vision 识别）

一个 iOS 项目：用 **YOLO 目标检测模型**定位德国车牌，再用 **Apple Vision 的
`RecognizeTextRequest`**（即官方示例 “Locating and displaying recognized text” 同款 OCR 引擎）
识别车牌字符，最后做**后处理**清洗干扰、校验德国车牌格式。SDK 形态设计，后续可独立封装。

## 功能

- **车牌检测**：调用 Core ML YOLO 模型（`CoreMLLicensePlateDetector`），模型无关，兼容
  Ultralytics YOLOv8 的原始输出布局（通道优先 `[1, 84, 8400]` 与 `[1, N, 4+numClasses]`
  两种都会自动识别），含 letterbox 预处理、坐标还原、NMS。检测类别通过 `YOLOConfig`
  的 `targetClassIndices` 过滤——单类车牌模型用 `[0]`，官方 COCO 占位模型用 `[2]`（car）。
- **车牌识别**：参考 Apple 官方文本检测/识别，使用 `RecognizeTextRequest`，支持
  `fast` / `accurate` 两种模式（**默认 `fast`**），识别语言默认 `de-DE`。
- **后处理（抗干扰）**：
  - 检测框裁剪后做轻度图像增强（对比度 + 锐化），抑制螺丝、边框、反光等干扰；
  - 按德国车牌格式 `^[A-ZÄÖÜ]{1,3}-[A-ZÄÖÜ]{1,2} ?[0-9]{1,4}[A-Z]?$` 做正则校验与规范化
    （自动补回连字符、剔除 EU 蓝条 “D” 伪影）；
  - 在多个候选中优先选择“格式合法 + 置信度最高”的结果；视频跨帧去重。
- **相机实时检测**：Demo 额外提供「相机」模式，参考 Apple 官方 `Camera` 示例的 `AVCapture` 写法——
  实时抽帧（节流 0.3s）跑完整流水线并在预览上叠加检测框，也可点快门拍摄静态图识别；
  支持「实时检测」开关与 `fast`/`accurate` 切换。当前占位模型为官方 `yolov8n`（检测 car → OCR 车辆区域）。
- **媒体输入**：Demo 支持三种来源——**图片**、**视频**上传（`PhotosPicker`），以及**相机实时拍摄**
  （`AVCaptureSession` 实时抽帧识别 + 快门拍摄识别），可在顶部分段切换。

## 目录结构

```
OCR/
├── LicensePlateOCR.xcodeproj      # Xcode 工程（衿青 Personal Team 签名占位）
├── Package.swift                   # 后续把 LicensePlateSDK 抽成独立 SPM 包用
├── Scripts/
│   └── build_yolov8n_coreml.py     # 一键生成官方 yolov8n 占位模型（本机运行）
├── Configuration/SampleCode.xcconfig
├── LicensePlateOCR/
│   ├── LicensePlateOCRApp.swift    # App 入口
│   ├── ContentView.swift           # Demo UI（图片/视频选择、模式切换、结果展示）
│   ├── Assets.xcassets
│   ├── PreviewContent/
│   └── LicensePlateSDK/            # ★ 可复用的识别引擎（SDK）
│       ├── LicensePlateRecognizer.swift   # 公开入口：image / video 识别
│       ├── LicensePlateDetector.swift      # YOLO 检测（Core ML）
│       ├── PlateTextRecognizer.swift       # Vision 文本识别（fast/accurate）
│       ├── GermanPlateValidator.swift      # 德国车牌正则校验与规范化
│       ├── PlatePostProcessor.swift        # 后处理 + 视频去重 + 图像增强
│       ├── LicensePlateError.swift
│       └── Resources/
│           ├── yolov8n.mlpackage            # 官方 COCO 占位模型（检测 car，由 download_model.sh 生成）
│           └── PUT_YOUR_YOLO_MODEL_HERE.txt  # 换成你自己的车牌模型说明
└── README.md
```

## 快速开始

### 1. 模型（官方 yolov8n 占位检测模型）

Demo 默认使用官方 Ultralytics `yolov8n`（COCO 80 类，输入 640×640）作为**占位检测模型**。
它**没有车牌类别**，所以 Demo 里用它检测 `car`（class 2），再对车辆区域做 OCR、用德国车牌
正则筛出车牌——这样就能直接跑通整条链路做演示。

> 模型文件不放进仓库（`.mlpackage` 较大），请用下面的一键脚本在本机（有网络）生成：

```bash
bash /Users/zhenzhipeng/Downloads/OCR/Scripts/download_model.sh
```

脚本会建虚拟环境、安装 `ultralytics`（会顺带下载 PyTorch，约几分钟）、下载官方
`yolov8n.pt` 并导出为 `yolov8n.mlpackage`，自动放进
`LicensePlateOCR/LicensePlateSDK/Resources/`。导出一次即可（Xcode 会把它编译成 `.mlmodelc`）。
（等价的可选方式：`python3 Scripts/build_yolov8n_coreml.py`，效果相同。）

> 想换成你自己的德国车牌模型时，二选一：
> 1. 把你的模型导出为 `license_plate_yolo.mlpackage` 放进 `Resources/`，并把
>    `ContentView` 里的 `LicensePlateRecognizer(modelName: "yolov8n", config: .cocoCar)`
>    改成 `LicensePlateRecognizer()`（默认即 `.germanPlate` 配置、`[0]` 车牌类）；或
> 2. 用其它文件名/类别时，显式传 `modelName` 与 `YOLOConfig(...)` 即可，`LicensePlateRecognizer`
>    与下游 OCR/后处理零改动。

导出你自己模型（`nms=False` 保留原始输出，便于 SDK 解码）：

```bash
yolo export model=best.pt format=coreml imgsz=640 nms=False
```

把生成的 `.mlpackage` 拖入 Xcode 的 `LicensePlateOCR/LicensePlateSDK/Resources/`，
勾选 *Copy items if needed* 并加入 `LicensePlateOCR` target。详见
`Resources/PUT_YOUR_YOLO_MODEL_HERE.txt`。

### 2. 选择开发团队（衿青 Personal Team）

打开 `LicensePlateOCR.xcodeproj` → *Signing & Capabilities* → *Team* 选择
**衿青 (Personal Team)**。Xcode 会自动写入 10 位 Team ID（工程里 `DEVELOPMENT_TEAM` 当前留空）。

### 3. 运行

选中一个 **iPhone** 运行目标（部署目标 iOS 18.0），⌘R 运行 Demo：
选择图片或视频 → 切换 `fast` / `accurate` → 查看识别结果与置信度。

## 作为 SDK 使用

`LicensePlateSDK/` 即为识别引擎，对外公开 `LicensePlateRecognizer`：

```swift
import LicensePlateSDK

let recognizer = LicensePlateRecognizer()          // 默认 fast 模式
let results = try await recognizer.recognize(in: cgImage, mode: .fast)
// results: [PlateResult]  ->  .text / .confidence / .boundingBox / .candidates

// 视频
let plates = try await recognizer.recognizeVideo(at: videoURL, mode: .accurate,
                                                 maxSamples: 30) { progress in ... }
```

后续想独立分发时，直接用仓库根目录的 `Package.swift`
（`swift build` 或作为本地 SPM 依赖引入其它工程）即可。

## 说明与可改进点

- 真实效果取决于 YOLO 模型的检测精度与训练数据；`YOLOConfig` 的
  `confidenceThreshold` / `iouThreshold` 可按需调参。
- 德国车牌使用 FE-Schrift 字体（刻意抗 OCR），Vision 通用 OCR 在脏污/逆光场景下可能出错，
  后处理已尽量补偿；若需更高精度，可换用专用车牌 OCR 模型替代 `PlateTextRecognizer`。
- 视频识别采用按时间抽帧（`AVAssetImageGenerator`，默认最多 30 帧），如需逐帧可改为
  `AVAssetReader` 管线。
