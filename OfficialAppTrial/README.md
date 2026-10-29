# OfficialAppTrial —— 用你的 4 点 pose 模型 + OCR 跑官方 YOLO App

这个目录把**你的德国车牌 pose 模型**和**车牌 OCR 文本识别**接进**官方 Ultralytics YOLO iOS app**，
用于对比"官方 app 管线 + 你的模型"的实际效果。转换与构建都在 **Mac** 上完成
（Core ML 导出与 Xcode 构建无法在 Windows 完成）。

> 官方 app 本身**没有 OCR**。OCR 是本目录提供的代码，由 `setup_official_app.py`
> **自动注入**官方 `ViewController.swift`——你不需要手工改任何 Swift 代码。

## 目录内容

| 文件 | 作用 |
|------|------|
| `De-plate-pose-500.pt` | 你的原始训练权重（已提交，5.6 MB） |
| `setup_official_app.py` | **一键**：clone 官方 app → 转模型 → 装模型 → 修 entitlements → 注入 OCR |
| `export_pose_coreml.py` | 单独用：把 `.pt` → 官方需要的 `.mlpackage` |
| `Integration/PlateOCR.swift` | OCR 实现（供阅读/复用；脚本已注入等价代码到 app） |
| `Integration/INTEGRATION.md` | 注入内容原理与**手工挂载**说明（脚本失效时的备选） |
| `_selftest_patch.py` | 注入逻辑本地自检（可选运行，验证脚本本身） |

---

## 一、结论先行：格式是否相同？

**不相同。** 官方 app 用 **Core ML `.mlpackage`**（int8、`nms=False`、input 640），
你的 pose 模型是 **ONNX `.onnx`**。官方 app 内部**只走 Core ML**（`MLModel` / `Vision`），
**没有 ONNX 运行时**，不能直接丢 `.onnx` 进去。

**解法**：用你手上的 `.pt` 走 `ultralytics` 官方导出链路转成 `.mlpackage`
（比从 ONNX 硬转干净，且自带官方解码所需元数据）。

### 好消息：官方解码对"4 点"天然兼容

- `PoseEstimator.PostProcessPose` 对传统布局 `[1, C, N]`（`shape[2] > shape[1]`）
  **动态**计算 `numKeypoints = (C - 5) / 3`。你的模型 `[1, 17, 8400]` → `C=17` → **4 个关键点**。
- 官方 `Plot.drawSinglePersonKeypoints` **只在 17 点时才画骨架**，4 点只画 4 个圆点，**不会越界崩溃**。
- **任务由文件夹决定**：放进 `Models/Pose/` 就当 `.pose`，`getModelFiles` 自动扫描，
  **不用改 `RemoteModels.swift`**。

---

## 二、Mac 上的操作步骤

### 1. 准备 Python 环境（官方要求 ultralytics ≥ 8.4.142）

```bash
uv venv --python 3.13 .venv
uv pip install "ultralytics[export-coreml]>=8.4.142"
source .venv/bin/activate
# 没有 uv：python3 -m venv .venv && source .venv/bin/activate && pip install "ultralytics[export-coreml]>=8.4.142"
```

### 2. 一条命令完成接入

在 `OfficialAppTrial/` 目录下：

```bash
python setup_official_app.py
```

它会依次做五件事：

1. **clone** 官方 app 到 `./yolo-ios-app`（已存在则复用）
2. **导出** `De-plate-pose-500.pt` → `De-plate-pose-500.mlpackage`
   （参数：`format=coreml, imgsz=640, quantize=8, nms=False`，与官方 pose 一致）
3. **安装模型**到 `yolo-ios-app/YOLOiOSApp/Models/Pose/`
4. **清空 entitlements** 里的 `associated-domains`（公司描述文件通常没有这个能力，否则签名失败）
5. **注入代码**到 `yolo-ios-app/YOLOiOSApp/YOLOiOSApp/ViewController.swift`：
   - 顶部加 `import Vision` / `import CoreImage`
   - 在 `loadSavedModelOrFirstOfficial()` 开头插入 `if plateLoadPoseModel() { return }`（**启动即强制加载你的模型**）
   - 在 `yoloView(_:didReceiveResult:)` 开头插入一行 OCR 调用
   - 文件末尾追加 `plateLoadPoseModel` + `PlateOCRService` + `GermanPlateValidator` + 辅助函数

### 2.1 为什么"模型列表里选不到我的模型"（重要）

官方 `showModelPicker()` 对列表做了**硬过滤**：

```swift
let names = Set((remoteModelsInfo[task.name] ?? []).map { $0.modelName.lowercased() })
let entries = makeModelEntries(for: task.name).filter { names.contains($0.displayName.lowercased()) }
```

只有**官方远程模型名**（`yolo26n-pose` 等）会留在列表里。你的 `.mlpackage` 虽然被
`getModelFiles(in: "Models/Pose")` 扫到并追加进 `currentModels`，却**永远不会显示在选择器 UI 中**，
因此**无法手动选择**。所以脚本改为**启动直接强制加载**你的模型（`.pose` 任务），
绕开选择器。启动后顶部标签会显示 `De-plate-pose-500`，即表示加载成功。

> 模型**不需要手动编译**：官方 `BasePredictor.create` 检测到非 `.mlmodelc` 时会自己
> 调用 `MLModel.compileModel(at:)`，所以 bundle 里放原始 `.mlpackage` 即可。

> **不新增任何源文件**，因此**不会改动 `project.pbxproj`**，Xcode 工程不会被破坏。
> 脚本幂等：重复运行会检测已注入并跳过。
> 想先看会改什么：`python setup_official_app.py --dry-run`

常用变体：

```bash
# 已有 clone / 已导出的模型
python setup_official_app.py --app-dir /path/to/yolo-ios-app --model exports/De-plate-pose-500.mlpackage

# INT8 掉点时用 FP16 重导，再传给 setup
python export_pose_coreml.py --weights De-plate-pose-500.pt --quantize 16
python setup_official_app.py --model exports/De-plate-pose-500.mlpackage
```

### 3. 构建运行

```bash
open yolo-ios-app/YOLOiOSApp/YOLOiOSApp.xcodeproj
```

- 选中真机（iOS 16+），Run。
- **无需在模型选择器里选**——脚本已让 app 启动即加载 `De-plate-pose-500`（见 2.1）。
  确认顶部标签显示 `De-plate-pose-500` 即加载成功。
- 对准车牌：应看到 **4 个彩色圆点**贴在车牌四角；识别成功时屏幕上方显示 `城市-编号`。

想先看**官方原版**基线（官方 `yolo26n-pose` 模型）：

```bash
cd yolo-ios-app && bash scripts/download-models.sh
```

然后 Run，任务选 Pose 即可（此时无 OCR，只有官方的关键点可视化）。

---

## 三、排查清单

| 现象 | 原因 / 处理 |
|------|-------------|
| 模型列表里没有 `De-plate-pose-500` | **预期行为**：官方选择器只列官方模型名（见 2.1）。脚本已改为启动强制加载，看顶部标签是否为 `De-plate-pose-500` |
| 顶部标签仍是官方模型 / 控制台打印 "Plate model not found in bundle" | `.mlpackage` 没进 bundle：确认在 `YOLOiOSApp/Models/Pose/` 下，Clean Build Folder 后重编 |
| 签名报 "doesn't include Associated Domains" | 脚本已清空 entitlements；仍报错就在 Xcode 的 Signing & Capabilities 里手动删掉 Associated Domains 那块 |
| 有 4 个点但位置整体偏移 | `result.originalImage` 方向与 `xyn` 不一致；在 `recognizePlateIfNeeded` 里先把帧转成 `.up` 方向（见 INTEGRATION.md 第 3 节） |
| 关键点数量不是 4 | 导出产物不是 `[1,17,8400]`；用 `nms=False` 重导，确认输出形状 |
| 只有点、没有文本 | 确认四角顺序正确（脚本已做质心重排兜底）；可把 `quantize` 改 16 提升关键点精度 |
| 注入脚本报"找不到 didReceiveResult" | 官方源码结构变了；按 INTEGRATION.md 手工加那一行即可 |
| 文本偶发乱码 | OCR 用的是 Apple Vision；要换成你的 PaddleOCR，替换 `runVisionOCR` 即可 |

---

## 四、局限（如实说明）

- 本环境的 git 访问 GitHub 被墙（`502`），**无法在此 clone 官方 app**，所以分支里是
  **接入脚本 + 权重 + OCR 代码**，官方 app 由 `setup_official_app.py` 在 Mac 上 clone。
- OCR 注入逻辑已用 `_selftest_patch.py` 本地自检（导入插入 / 调用注入 / 幂等**全部通过**），
  但**未在真机编译验证**（无 Xcode）。若官方 `ViewController.swift` 结构变化，
  脚本会明确报错并提示手工插入位置。
- OCR 默认用 **Apple Vision**；透视矫正与德国车牌校验可直接复用于你的 PaddleOCR 引擎。
