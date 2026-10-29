# 把车牌 OCR 嵌入官方 YOLO App —— 集成说明

目标：让官方 app 用**你的 4 点 pose 模型**检测车牌四角，然后在检测到车牌时，用四角做透视矫正 + Apple Vision OCR，把识别到的德国车牌文本显示在画面上。

---

## 0. 前置：模型放哪里、怎么被 App 发现

官方 app 通过**文件夹**决定任务，不依赖模型元数据：

- `YOLOiOSApp/Models/Pose/` → 任务 = `.pose`
- 官方工程把 `YOLOiOSApp/Models` 作为 folder reference 打进 bundle；`ViewController` 启动时用
  `getModelFiles(in: "Models/Pose")` 扫描该目录，**自动列出**里面的 `.mlpackage`。
  所以只要把转换好的 `De-plate-pose-500.mlpackage` 放进这个目录并重新 build 即可，
  **不需要改 `RemoteModels.swift`**（那个文件只登记"可下载"的官方模型 URL）。

> 官方 pose 模型（`yolo26n-pose.mlpackage`）默认不在仓库里，是构建时由
> `scripts/download-models.sh` 下载的。要"拷贝官方模型"就在 Mac 上跑一次该脚本；
> 要用你自己的模型，就用 `export_pose_coreml.py` 直接产出并复制进去。

---

## 1. 加入 OCR 源文件

把 `PlateOCR.swift` 拖进 **`YOLOiOSApp` 这个 app target**（不是 `UltralyticsYOLO` package）。
它会提供 `PlateOCRService`、`PlateOCRResult`、`GermanPlateValidator`。

---

## 2. 在 ViewController 里挂载（最小改动）

### 2.1 加属性

```swift
private let plateOCR = PlateOCRService(throttle: 0.3)
private let plateLabel = UILabel()
```

### 2.2 在 `viewDidLoad` 末尾加一个显示结果的 label

```swift
plateLabel.translatesAutoresizingMaskIntoConstraints = false
plateLabel.textColor = .white
plateLabel.backgroundColor = UIColor.black.withAlphaComponent(0.55)
plateLabel.font = .monospacedSystemFont(ofSize: 26, weight: .semibold)
plateLabel.textAlignment = .center
plateLabel.numberOfLines = 1
plateLabel.layer.cornerRadius = 8
plateLabel.layer.masksToBounds = true
plateLabel.isHidden = true
view.addSubview(plateLabel)
NSLayoutConstraint.activate([
    plateLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
    plateLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 96),
    plateLabel.widthAnchor.constraint(lessThanOrEqualTo: view.widthAnchor, constant: -40),
    plateLabel.heightAnchor.constraint(equalToConstant: 46),
])
```

### 2.3 在已有的 `YOLOViewDelegate` 回调里加 OCR

找到文件末尾 `extension ViewController` 里的：

```swift
func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult)
```

在**保留原有性能显示逻辑之后**，追加：

```swift
// —— 只在 Pose 任务、且本帧有 ≥4 个关键点时做车牌 OCR ——
guard currentTask == "Pose",
      let keypoints = result.keypointsList.first,
      keypoints.xyn.count >= 4
else { return }

recognizePlate(from: keypoints, result: result, in: view)
```

### 2.4 新增 OCR 辅助方法

把下面整段加到 `ViewController` 的类体内（不要放进 extension 之外也可以，放在同一个 extension 里最简单）：

```swift
private func recognizePlate(from keypoints: Keypoints, result: YOLOResult, in view: YOLOView) {
    // 取当前帧图像：优先用 result.originalImage；没有就现抓一帧。
    let handle: (UIImage) -> Void = { [weak self] frame in
        guard let self, let cg = frame.cgImage else { return }

        // keypoints.xyn 是归一化坐标(0..1)。映射到 frame 的像素空间。
        let w = CGFloat(cg.width), h = CGFloat(cg.height)
        let pixelCorners = keypoints.xyn.prefix(4).map {
            CGPoint(x: CGFloat($0.x) * w, y: CGFloat($0.y) * h)
        }
        // 重排为 LT/RT/RB/LB，避免模型输出顺序不同导致"蝴蝶结"透视畸变。
        let ordered = self.orderCorners(Array(pixelCorners))

        self.plateOCR.recognize(frame: frame, corners: ordered) { [weak self] ocr in
            guard let self else { return }
            DispatchQueue.main.async {
                if let ocr {
                    self.plateLabel.text = "  \(ocr.text)  "
                    self.plateLabel.isHidden = false
                } else {
                    // 没识别到就保留上一次结果，避免闪烁；如想清空改成 isHidden = true
                }
            }
        }
    }

    if let frame = result.originalImage {
        handle(frame)
    } else {
        view.capturePhoto { image in
            if let image { handle(image) }
        }
    }
}

/// 按质心四象限把 4 个点重排为 [leftTop, rightTop, rightBottom, leftBottom]。
private func orderCorners(_ pts: [CGPoint]) -> [CGPoint] {
    guard pts.count == 4 else { return pts }
    let cx = pts.map(\.x).reduce(0, +) / 4
    let cy = pts.map(\.y).reduce(0, +) / 4
    var lt = CGPoint.zero, rt = CGPoint.zero, rb = CGPoint.zero, lb = CGPoint.zero
    for p in pts {
        switch (p.x < cx, p.y < cy) {
        case (true, true):   lt = p
        case (false, true):  rt = p
        case (false, false): rb = p
        case (true, false):  lb = p
        }
    }
    return [lt, rt, rb, lb]
}
```

---

## 3. 坐标/方向注意事项（真机排查顺序）

1. **`result.originalImage` 是否可用**
   `PoseEstimator` 会把当前帧塞进 `result.originalImage`；若为 `nil`，代码已自动回退到
   `view.capturePhoto {}` 现抓一帧（与关键点可能差 1 帧，实时场景可接受）。

2. **归一化坐标 → 像素**
   这里用 `keypoints.xyn * cgImage 像素尺寸`，与 `orig_shape` 无关，避免 letterbox 反算误差。
   若发现裁剪框整体偏移/方向反了，说明 `originalImage` 的 `imageOrientation` 与 `xyn`
   的朝向不一致——可在 `handle` 开头先把 `frame` 用 `UIGraphicsImageRenderer` 画成 `.up` 方向，
   再用 `cgImage`，坐标即对齐（本项目 SDK 里遇到过同类问题）。

3. **角点顺序**
   官方 `PostProcessPose` **不重排**关键点，按模型原始输出顺序。本项目模型固定为
   `left_top, right_top, right_bottom, left_bottom`；上面的 `orderCorners` 已按质心再兜底一次，
   两种情况都能得到正确四边形。

4. **4 个关键点不会崩**
   官方 `Plot.drawSinglePersonKeypoints` **只在 `numKeypoints == 17` 时画骨架**，4 点只画 4 个圆点，
   并有 `startIdx/endIdx < points.count` 的守卫。所以 4 点模型在官方 app 里是安全的。

5. **性能**
   `PlateOCRService` 内部已做 0.3s 节流 + 后台串行队列；`didReceiveResult` 每帧都回调，不必再手动降频。

6. **调试**
   先确认 `currentTask == "Pose"` 且 `keypoints.xyn.count == 4`（可在回调里 `print` 一次）。
   若 `count` 不是 4，说明模型没被识别为传统布局，检查导出产物输出形状是否为 `[1, 17, 8400]`。

---

## 4. 与官方"骨架可视化"的关系

- 你的 4 点是**车牌四角**，不是人体 17 关节。官方会自动画 4 个彩色圆点（无连线），
  这正好可以当作"关键点是否贴合车牌"的直观调试视图。
- 想把四点连成框，可参考官方 `Plot.drawKeypoints`，用 `drawLine` 在 4 点间连线；
  官方已提供 `drawLine` / `drawCircle`，改动很小。

---

## 5. 可选：只在检测到车牌时显示，避免误报

`GermanPlateValidator.normalize` 已做结构校验（`城市1-3字母 + 1-2字母 + 1-4数字 + 可选E/H`，
且城市+识别号 ≤8 字符）。若仍有误报，可再加一条：`ocr.confidence >= 0.5` 才显示。
