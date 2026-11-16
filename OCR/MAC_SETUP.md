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
- App 内「Apple Vision / PaddleOCR」选择器可用，**默认 PaddleOCR**；Apple Vision 仅供 A/B 对照。
- 不再需要 `PaddleOCR-iOS` / Paddle Lite：识别直接复用 `pod 'onnxruntime-objc'` 加载
  `Resources/PP-OCRv5_rec_mobile_infer.onnx`（16.5 MB）与 `Resources/ppocrv5_dict.txt`。
- 只跑 **rec**（不做 DB 文本检测）：关键点矫正图本身就是一条规整文本行，跑 det 反而会把 EU 蓝条和年检章切成伪文本框。
- 模型/字典来自 Apache-2.0 的 `imgocr` 0.2.4（Paddle2ONNX 转换）。缺失时可在 PC 上执行
  `python Scripts/paddleocr_rec_reference.py --fetch-models` 重新拉取；`--self-test` 用合成车牌验证预处理与 CTC 解码。
- 若真机上 PaddleOCR 结果异常，先用该脚本对同一张矫正图跑一遍，区分「张量/解码问题」与「图片质量问题」。

---

## 离线方案（仅当 Mac 无法访问 download.onnxruntime.ai 时）
若 `pod install` 因网络失败，告诉我——我给你"在 Windows 下载两个 pod 归档包 → scp 到 Mac → 手动集成"的离线步骤（ heavier，因为要带入编译好的 Pods 产物）。

## 本次新增 / 改动文件
- 新增：`setup_mac.sh`（CocoaPods 一键接入）、`POSE_INTEGRATION.md`
- 改动：`LicensePlateRecognizer.swift`、`YOLOOutputDecoder.swift`、`LicensePlateDetector.swift`、`ContentView.swift`、新增 `ONNXPoseLicensePlateDetector.swift` / `PerspectivePlateCorrector.swift` / `OCREngine.swift` / `PaddleOCRTextRecognizer.swift`
- 资源：`Resources/De-plate-pose.onnx`
