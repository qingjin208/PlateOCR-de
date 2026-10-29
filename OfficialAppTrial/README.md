# OfficialAppTrial —— 用你的 4 点 pose 模型跑官方 YOLO App

这个目录是把**你的德国车牌 pose 模型**接入**官方 Ultralytics YOLO iOS app** 的全套材料，
用于对比"官方 app 管线 + 你的模型"的实际效果。所有转换/构建都在 **Mac** 上完成
（Core ML 导出与 Xcode 构建无法在 Windows 完成）。

## 目录内容

| 文件 | 作用 |
|------|------|
| `De-plate-pose-500.pt` | 你的原始训练权重（已提交，5.6 MB） |
| `export_pose_coreml.py` | 把 `.pt` → 官方需要的 `.mlpackage`（对齐官方导出参数） |
| `Integration/PlateOCR.swift` | 自包含 OCR：四角透视矫正 + Vision 识别 + 德国车牌校验 |
| `Integration/INTEGRATION.md` | 在官方 `ViewController` 里的精确挂载步骤 |

---

## 一、结论先行：格式是否相同？

**不相同。** 官方 app 用的是 **Core ML `.mlpackage`**（int8、`nms=False`、input 640），
而你的 pose 模型是 **ONNX `.onnx`**。官方 app 内部**只走 Core ML**（`MLModel` / `Vision`），
**没有 ONNX 运行时**，不能直接丢 `.onnx` 进去。

**解法**：你手上有 `.pt` 原始权重 → 用 `ultralytics` 官方导出链路转成 `.mlpackage`。
这条路比从 ONNX 硬转干净得多，而且会**自动写入官方解码所需的元数据**。

### 好消息：官方解码对"4 点"天然兼容

- `PoseEstimator.PostProcessPose` 对传统布局 `[1, C, N]`（`shape[2] > shape[1]`）
  **动态**计算 `numKeypoints = (C - 5) / 3`。你的模型 `[1, 17, 8400]` → `C=17` → **4 个关键点**，正好。
- 官方 `Plot.drawSinglePersonKeypoints` **只在 17 点时才画骨架**，4 点只画 4 个圆点，**不会越界崩溃**。
- **任务类型由文件夹决定**：放进 `Models/Pose/` 就当 `.pose` 用，不依赖模型元数据 `task`。

---

## 二、Mac 上的操作步骤

### 1. 克隆官方 app（Mac 有网络）

```bash
git clone https://github.com/ultralytics/yolo-ios-app.git
cd yolo-ios-app
```

> 若想先看**官方原版**效果（用官方 `yolo26n-pose` 模型），先跑一次：
> ```bash
> bash scripts/download-models.sh      # 下载 7 个 nano 模型到 YOLOiOSApp/Models/
> ```
> 用 Xcode 打开 `YOLOiOSApp/YOLOiOSApp.xcodeproj`，真机 Run，任务选 Pose 即可。

### 2. 准备 Python 环境（官方要求 ultralytics ≥ 8.4.142）

```bash
# 在 yolo-ios-app 仓库根目录
uv venv --python 3.13 .venv
uv pip install "ultralytics[export-coreml]>=8.4.142"
source .venv/bin/activate
# 没有 uv 的话：python3 -m venv .venv && source .venv/bin/activate && pip install "ultralytics[export-coreml]>=8.4.142"
```

### 3. 转换你的权重（本目录的脚本）

```bash
# 在你的分支里，OfficialAppTrial 目录下
python export_pose_coreml.py \
    --weights De-plate-pose-500.pt \
    --copy-to-app /path/to/yolo-ios-app
```

脚本会：
- 以 `format="coreml", imgsz=640, quantize=8, nms=False` 导出（与官方 pose 一致）；
- 校验元数据 `task` 与输出形状（应看到 `输出 shape=[1,17,8400] -> 4 keypoints`）；
- 把 `De-plate-pose-500.mlpackage` 复制到
  `yolo-ios-app/YOLOiOSApp/Models/Pose/`。

> **精度提示**：官方 nano 模型用 `quantize=8`(INT8)。小模型 + 小关键点若在 INT8 下掉点，
> 改用 `--quantize 16`(FP16) 再导一次：`--quantize 16`。体积更大但精度更好。

### 4. 加入 OCR 并挂载

1. 把 `Integration/PlateOCR.swift` 拖进 **`YOLOiOSApp` app target**。
2. 按 `Integration/INTEGRATION.md` 的 2.1–2.4 修改 `YOLOiOSApp/YOLOiOSApp/ViewController.swift`：
   加 `plateOCR` + `plateLabel`，在 `yoloView(_:didReceiveResult:)` 里调用 `recognizePlate(...)`。

### 5. 构建运行

```bash
open YOLOiOSApp/YOLOiOSApp.xcodeproj
```

- 选中真机（iOS 16+），Run。
- 顶部模型选择器 → **Pose** → 选 `De-plate-pose-500`。
- 对准车牌：应看到 4 个彩色圆点贴在车牌四角；识别成功时顶部出现 `城市-编号` 文本。

---

## 三、排查清单

| 现象 | 原因 / 处理 |
|------|-------------|
| 模型列表里没有 `De-plate-pose-500` | `.mlpackage` 没进 bundle：确认它落在 `YOLOiOSApp/Models/Pose/` 且 `Models` 是 folder reference；Clean Build Folder 后重编 |
| 有 4 个点但位置整体偏移 | `result.originalImage` 方向与 `xyn` 不一致，按 INTEGRATION.md 第 3 节第 2 条先转 `.up` |
| 关键点数量不是 4 | 导出产物不是 `[1,17,8400]`：检查是否误导出成 end2end 头，用 `nms=False` 重导 |
| 只有点没文本 | OCR 需要 4 点顺序正确；确认 `orderCorners` 生效；调低 `minimumTextHeight` 或把 `quantize` 改 16 提升关键点精度 |
| 文本乱码 | `GermanPlateValidator` 会过滤非法字符；必要时加 `ocr.confidence >= 0.5` 门槛 |

---

## 四、局限（如实说明）

- 本目录的 OCR 集成代码**未在真机编译验证**（开发环境为 Windows，无 Xcode）。
  逻辑与官方 SDK 接口（`Keypoints.xyn`、`YOLOResult.originalImage`、`YOLOView.capturePhoto`）
  严格对齐，但坐标/方向的最后 1 帧对齐可能需要你在真机上微调。
- `PlateOCR.swift` 用的是 **Apple Vision** 文字识别（非你的 PaddleOCR）；
  若要复用你现有 SDK 的 PaddleOCR 引擎，把 `runVisionOCR` 换成 SDK 的 OCR 调用即可，
  透视矫正与车牌校验可直接复用。
