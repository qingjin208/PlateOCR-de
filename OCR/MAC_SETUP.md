# 在 Mac 上完成"德国车牌关键点检测 + 可切换 OCR"的接入

## 最简路径（推荐）：一键脚本
在 Mac 上进入 `OCR` 目录执行：

```bash
bash setup_mac.sh
```

脚本会：安装 CocoaPods（如未装）→ 生成 Podfile → `pod install` 拉取 ONNX Runtime iOS 框架 → 打开 `LicensePlateOCR.xcworkspace`。

> ⚠️ 跑完脚本后，**用 Xcode 打开 `LicensePlateOCR.xcworkspace`，不是 `.xcodeproj`**（pod install 后只能用 workspace）。

---

## 为什么是 CocoaPods（而不是手动下载 xcframework）
- 新版 ONNX Runtime **已不在 GitHub Release 提供 iOS 包**（release 里只有 Linux/macOS/Windows 库）。
- iOS 版 `onnxruntime-mobile-objc` 是微软通过 **CocoaPods** 分发的，二进制在 `download.onnxruntime.ai`，并会自动依赖 `onnxruntime-mobile-c`（真正的推理框架）。
- 我们 Swift 代码用的 `import onnxruntime`（ORTEnv / ORTSession / ORTValue 等 ObjC 封装）正是由这个 pod 编译提供的，手动丢一个 xcframework 无法提供这个模块。**所以 CocoaPods 是正确且最简单的方式。**

---

## Mac 上手动对照步骤（脚本已自动完成，这里供理解）
1. 装 CocoaPods：`sudo gem install cocoapods`（或 `brew install cocoapods`）。
2. 在 `OCR` 目录下建 `Podfile`：
   ```ruby
   platform :ios, '18.0'
   target 'LicensePlateOCR' do
     pod 'onnxruntime-mobile-objc', '1.18.0'
   end
   ```
3. `pod install --repo-update`（从 download.onnxruntime.ai 下载框架，需联网）。
4. 打开 **`LicensePlateOCR.xcworkspace`** → 选设备 → ▶ 运行。

### 关于 `.onnx` 模型
`LicensePlateOCR/LicensePlateSDK/Resources/De-plate-pose.onnx` 已在工程「文件夹同步」目录内，Xcode 会自动打进 App 包，**无需手动添加到 Copy Bundle Resources**。

### 关于真机签名
工程预设 `DEVELOPMENT_TEAM = 224Z7T23JY`（Automatic）。若这台 Mac 没有该团队证书，构建会在签名阶段失败 → 在 Xcode 里把 Target → Signing & Capabilities 的 Team 改成你自己的（免费 Apple ID 即可）。

---

## PaddleOCR（已实现，复用 ONNX Runtime）
- OCR 固定为 PaddleOCR（Apple Vision 已彻底移除，无引擎选择器）。「极速 / 精准」两档仍可切换。
- 不再需要 `PaddleOCR-iOS` / Paddle Lite：识别直接复用 `pod 'onnxruntime-objc'` 加载
  `Resources/PP-OCRv5_rec_mobile_infer.onnx`（16.5 MB）与 `Resources/ppocrv5_dict.txt`。
- 只跑 **rec**（不做 DB 文本检测）：关键点矫正图本身就是一条规整文本行，跑 det 反而会把 EU 蓝条和年检章切成伪文本框。
- 模型/字典来自 Apache-2.0 的 `imgocr` 0.2.4（Paddle2ONNX 转换）。缺失时可在 PC 上执行
  `python Scripts/paddleocr_rec_reference.py --fetch-models` 重新拉取；`--self-test` 用合成车牌验证预处理与 CTC 解码。
- **真机第一步**：在 Xcode 的 Build Phases → Copy Bundle Resources 里确认
  `PP-OCRv5_rec_mobile_infer.onnx` 与 `ppocrv5_dict.txt` 确实进了包。工程用
  `fileSystemSynchronizedGroups`（`files` 为空），`PaddleOCRResources` 会先按 Apple 文档把
  `subdirectory` 当作顶层 bundle 目录做多候选探测，仅在全部落空时才**限定在 `resourceURL` 内、
  跳过 Frameworks/PlugIns/*.lproj、最多 2000 项**地按文件名查找；两个资源仍找不到时 triage 行会显示原因，
  识别会**显式报错**而非假装成功。
- 精度对照：`python Scripts/paddleocr_rec_reference.py --eval samples/ --labels samples/labels.tsv`
  输出准确率与 CER；用同一批矫正图在真机分别以 `fast` / `accurate` 各跑一次即可量化模式间收益。设置页打开「诊断」
  后，检测信息行会显示全字典 greedy 原始读数与异常字符（清洁是否彻底的信号）。
- 若真机上 PaddleOCR 结果异常，先用该脚本对同一张矫正图跑一遍，区分「张量/解码问题」与「图片质量问题」。

---

## Apple 平台规范对齐（2026-11-16，按官方文档核查）
- **启动时间**（《Reducing your app's launch time》）：首帧前不得在主线程做重初始化。
  PaddleOCR 的 16.5 MB 图 / 字典 / ORT session / warmup 已从 `ContentView.init` 移出，
  改为首帧后 `.task` 后台预热 + 首次识别按需 `await` 同一份一次性加载。
  ⚠️ **遗留项**：检测器 `De-plate-pose.onnx` 仍在 `LicensePlateRecognizer` 的
  convenience init 里同步加载，冷启动耗时需在真机用 Instruments / XCTest 量测后再决定是否同样延迟化。
- **Swift 并发**（WWDC21《Swift concurrency: Behind the scenes》）：协作线程池只有 CPU 核数个线程，
  不能在其中阻塞。`ORTSession.run` 已搬到专用串行队列，只有纯 Swift `DecoderOutput` 跨回调用方；
  三处 `NSLock` 全部换成 Apple 推荐的 `OSAllocatedUnfairLock`，且 recognizer 构建移出临界区。
- **渲染**（`CIContext` / `CGColorSpace` 文档）：`CIContext` 改为 SDK 级共享单例（原本每个调用点各建一个）；
  9 处 `CGColorSpaceCreateDeviceRGB()`（device-dependent）统一为命名 sRGB，保证与 PC 参考的字节级一致。
- **后台执行**（《Extending your app's background execution time》）：图片 / 视频识别 pass 现在成对申请
  `beginBackgroundTask(withName:expirationHandler:)` / `endBackgroundTask`，用户锁屏时不会在 5 秒后被挂起中断。
- **字符约束**：PaddleOCR 的字符表由 ONNX 图与字典固定（18,385 类），运行时不可改；
  727 条城市码只作为择优提示交给 `GermanPlateValidator`，不硬过滤（词表缺项不误拒真牌）。
  EU 蓝条的 D 与主字符等高，靠 `PlateImageCleaner` 遮白，不靠字符过滤。
- **隐私 / 上架**：`PhotosPicker` 进程外取图，**不需要** `NSPhotoLibraryUsageDescription`；
  相机描述 `NSCameraUsageDescription` 已声明。App 自身未使用 `UserDefaults`、文件时间戳等
  required reason API，**目前不需要** `PrivacyInfo.xcprivacy`；上架前请检查
  `onnxruntime-objc` / `onnxruntime-mobile-objc` pod 是否自带已签名的隐私清单与签名。
- **iPad（既有配置，非本分支改动）**：`UIRequiresFullScreen = YES` + `TARGETED_DEVICE_FAMILY = 1,2`
  会禁用 Split View / Slide Over；若要多任务需在 Target 设置里调整。

---

## 离线方案（仅当 Mac 无法访问 download.onnxruntime.ai 时）
若 `pod install` 因网络失败，告诉我——我给你"在 Windows 下载两个 pod 归档包 → scp 到 Mac → 手动集成"的离线步骤（ heavier，因为要带入编译好的 Pods 产物）。

## 本次新增 / 改动文件
- 新增：`setup_mac.sh`（CocoaPods 一键接入）、`POSE_INTEGRATION.md`
- 改动：`LicensePlateRecognizer.swift`、`YOLOOutputDecoder.swift`、`LicensePlateDetector.swift`、`ContentView.swift`、新增 `ONNXPoseLicensePlateDetector.swift` / `PerspectivePlateCorrector.swift` / `PaddleOCRSupport.swift`（原 `OCREngine.swift`） / `OCRTypes.swift` / `PaddleOCRTextRecognizer.swift`
- 资源：`Resources/De-plate-pose.onnx`、`Resources/PP-OCRv5_rec_mobile_infer.onnx`、
  `Resources/ppocrv5_dict.txt`
- PC 校验脚本：`Scripts/paddleocr_rec_reference.py`（`--self-test` / `--eval` / `--fetch-models`）
