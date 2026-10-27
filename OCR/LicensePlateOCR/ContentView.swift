/*
 License Plate OCR SDK — Demo App

 Abstract:
 Redesigned UI (see Mockups/plate-ocr-demo-redesign.html): a clean header with a
 settings gear, a sliding source segmented control (拍照 / 相册 / 视频), an inline
 极速 / 精准 mode toggle, a dark rounded preview with the SDK's aligned detection
 overlay, and German-plate result cards. Advanced controls (OCR engine, input channel,
 enhancement) moved into a bottom sheet.

 IMPORTANT — detection/overlay alignment:
 The image overlay (`PlateQuadOverlay`) still maps normalized detection boxes onto the
 displayed picture with a `GeometryReader` scaled-to-fit transform, so boxes stay glued
 to the plate. The camera/video overlays live in `CameraPreview`/`VideoPlatePreview`
 and keep their own aspect-fill / `videoRect` mapping. This redesign only changes chrome
 and cards; it does NOT touch those geometry mappings.
 */

import SwiftUI
import Combine
import PhotosUI
import CoreTransferable
import CoreGraphics
import Vision
import CoreImage
import UniformTypeIdentifiers
import UIKit
import AVFoundation

/// Accent colour matching the HTML mockup (#2f6bff). Internal so CameraView can reuse it.
let appAccent = Color(red: 0.184, green: 0.420, blue: 1.0)

enum MediaSource: String, CaseIterable, Hashable {
    case camera = "拍照"
    case image = "相册"
    case video = "视频"
}

struct ContentView: View {
    // Share the bundled plate detector and Apple OCR across all media sources.
    @State private var recognizer: LicensePlateRecognizer
    @State private var source: MediaSource = .camera
    @State private var cameraController: CameraController
    /// Opt-in SDK strengthening. One master switch in the sheet plus three isolated
    /// sub-switches, so a regression can be narrowed down on device.
    @State private var enhancements: RecognitionEnhancements

    /// One recognizer is shared by the image/video pipeline and the live camera
    /// so the YOLO model is loaded only once.
    init() {
        // The Demo starts with the enhancements on; the toggle below can turn them off.
        let enhancements = RecognitionEnhancements.on
        let rec: LicensePlateRecognizer
        #if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc)
        // ONNX Runtime linked: use the trained German-plate YOLOv8-pose model directly.
        rec = LicensePlateRecognizer(poseModelName: "De-plate-pose-500",
                                     engine: .appleVision,
                                     enhancements: enhancements)
        #else
        // ONNX Runtime not linked yet: fall back to the legacy Core ML detector until the
        // pose model + onnxruntime framework are added (see POSE_INTEGRATION.md).
        rec = LicensePlateRecognizer()
        rec.enhancements = enhancements
        #endif
        _recognizer = State(initialValue: rec)
        _enhancements = State(initialValue: enhancements)
        _cameraController = State(initialValue: CameraController(recognizer: rec))
        // Start with live detection OFF so the camera tab shows the idle guide frame
        // (matching the mockup: tap 开始实时识别 to begin).
        cameraController.liveEnabled = false
    }

    @State private var imageItem: PhotosPickerItem?
    @State private var videoItem: PhotosPickerItem?
    @State private var imageData: Data?
    @State private var displayImage: UIImage?
    @State private var videoURL: URL?

    @State private var mode: RecognitionMode = .fast   // default = fast
    @State private var ocrEngine: OCREngine = .appleVision   // user-selected OCR engine
    /// Input channel order for the ONNX detector. RGB matches Ultralytics training and the
    /// PC reference pipeline; switch to BGR only if an export needs it.
    @State private var useRGB: Bool = true
    @State private var results: [PlateResult] = []
    /// Number of raw detections last pass (before OCR/validation). Shown in the
    /// "nothing recognized" state to distinguish "model saw nothing" from
    /// "model saw plates but OCR/validation rejected them".
    @State private var lastDetectionCount: Int = 0
    @State private var isBusy = false
    @State private var videoProgress = 0.0
    @State private var errorMessage: String?
    /// Plays the selected video so the upload has a visible reaction (the SDK samples
    /// frames for recognition; the player just previews the source clip).
    @State private var player: AVPlayer?
    /// Per-sampled-frame results, used to paint the plate tint while the video plays.
    @State private var videoSamples: [PlateVideoSample] = []
    /// Playback state for the custom video preview (AVPlayerLayer + overlay has no
    /// built-in transport controls).
    @State private var isPlaying = false

    /// Bottom-sheet for advanced controls (OCR engine / input channel / enhancement).
    @State private var showSettings = false

    /// Drives the camera session lifecycle from the app's foreground/background state so
    /// it is no longer tied to the `CameraView`'s mount/unmount (which caused the black
    /// flash when switching media sources).
    @Environment(\.scenePhase) private var scenePhase

    // The body is split into small, explicitly typed pieces: the full modifier chain used
    // to blow past the compiler's type-check budget ("unable to type-check this expression
    // in reasonable time"). Each concern now lives in its own property / method below.
    var body: some View {
        scrollContent
            .overlay { settingsOverlay }
            .animation(.easeInOut(duration: 0.3), value: showSettings)
            .onChange(of: imageItem) { _, newItem in handleImageItemChange(newItem) }
            .onChange(of: videoItem) { _, newItem in handleVideoItemChange(newItem) }
            .onChange(of: mode) { _, newMode in handleModeChange(newMode) }
            .onChange(of: ocrEngine) { _, newEngine in handleOCREngineChange(newEngine) }
            .onChange(of: useRGB) { _, newValue in handleUseRGBChange(newValue) }
            .onChange(of: enhancements) { _, newValue in handleEnhancementsChange(newValue) }
            .onChange(of: videoURL) { _, newURL in handleVideoURLChange(newURL) }
            .onChange(of: source) { _, newSource in handleSourceChange(newSource) }
            .onChange(of: scenePhase) { _, phase in handleScenePhaseChange(phase) }
            .task { await initialSetup() }
    }

    private var scrollContent: some View {
        ScrollView {
            VStack(spacing: 16) {
                header
                mediaPreview
                resultsSection
                errorText
            }
            .padding(.vertical)
        }
    }

    @ViewBuilder
    private var errorText: some View {
        if let message = errorMessage {
            Text(message).font(.footnote).foregroundStyle(.red).padding(.horizontal)
        }
    }

    @ViewBuilder
    private var settingsOverlay: some View {
        if showSettings {
            ZStack(alignment: .bottom) {
                Color.black.opacity(0.4)
                    .ignoresSafeArea()
                    .onTapGesture { withAnimation { showSettings = false } }
                SettingsSheet(mode: $mode,
                              ocrEngine: $ocrEngine,
                              useRGB: $useRGB,
                              enhancements: $enhancements) { withAnimation { showSettings = false } }
                    .padding(.horizontal, 4)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
    }

    // MARK: - Change handlers

    private func handleImageItemChange(_ newItem: PhotosPickerItem?) {
        guard let newItem else { return }
        Task { await runImage(newItem) }
    }

    private func handleVideoItemChange(_ newItem: PhotosPickerItem?) {
        guard let newItem else { return }
        Task { await runVideo(newItem) }
    }

    private func handleModeChange(_ newMode: RecognitionMode) {
        cameraController.mode = newMode
    }

    private func handleOCREngineChange(_ newEngine: OCREngine) {
        recognizer.ocrEngine = newEngine
    }

    private func handleUseRGBChange(_ newValue: Bool) {
        recognizer.useRGB = newValue
    }

    /// Push the enhancement switches into the SDK: the live camera shares this same
    /// recognizer instance, so one assignment covers image, video and camera.
    private func handleEnhancementsChange(_ newValue: RecognitionEnhancements) {
        recognizer.enhancements = newValue
    }

    private func handleVideoURLChange(_ newURL: URL?) {
        player?.pause()
        isPlaying = false
        if let newURL {
            player = AVPlayer(url: newURL)
        } else {
            player = nil
        }
    }

    private func handleSourceChange(_ newSource: MediaSource) {
        if newSource != .video {
            player?.pause()
            isPlaying = false
        }
        // Camera session follows the selected source: (re)start on the camera tab,
        // suspend when leaving it. Keeps the preview layer mounted (no black flash).
        if newSource == .camera {
            Task { await cameraController.setup() }
        } else {
            cameraController.suspendSession()
        }
    }

    private func handleScenePhaseChange(_ phase: ScenePhase) {
        if phase == .background {
            cameraController.suspendSession()
        } else if phase == .active {
            if source == .camera { Task { await cameraController.setup() } }
        }
    }

    private func initialSetup() async {
        // Kick the camera session once at launch when we open on the camera tab.
        if source == .camera { await cameraController.setup() }
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .center, spacing: 10) {
            Text("车牌识别").font(.headline.bold())
            SourceSegment(selection: $source)
                .frame(width: 198)
            Spacer(minLength: 0)
            Button {
                withAnimation { showSettings = true }
            } label: {
                Image(systemName: "gearshape.fill")
                    .font(.system(size: 18))
                    .foregroundStyle(.secondary)
                    .frame(width: 38, height: 38)
                    .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
            }
        }
        .padding(.horizontal)
        .padding(.top, 8)
    }

    // MARK: - Media preview

    @ViewBuilder
    private var mediaPreview: some View {
        VStack(spacing: 0) {
            // The camera view stays mounted across media-source switches so its
            // AVCaptureVideoPreviewLayer (and the configured session) are never torn down
            // and rebuilt — that rebuild was the source of the black flash on return. When
            // another source is selected it collapses to zero height and is hidden, and the
            // session is suspended (so the green camera indicator goes away). Lifecycle is
            // driven by the `source` / `scenePhase` observers below, not by this view's
            // mount/unmount.
            CameraView(controller: cameraController)
                .frame(height: source == .camera ? 470 : 0)
                .opacity(source == .camera ? 1 : 0)
                .allowsHitTesting(source == .camera)
                .clipShape(RoundedRectangle(cornerRadius: 16))
                .padding(.horizontal)
            if source == .image { imagePreview }
            if source == .video { videoPreview }
        }
    }

    @ViewBuilder
    private var imagePreview: some View {
        if let uiImage = displayImage {
            Image(uiImage: uiImage)
                .resizable()
                .scaledToFit()
                .clipShape(RoundedRectangle(cornerRadius: 16))
                // Overlay maps normalized boxes onto the scaled-to-fit image, so the
                // tint tracks the plate exactly (no rotation/scale drift).
                .overlay { PlateQuadOverlay(results: results) }
                .padding(.horizontal)
        } else {
            PhotosPicker(selection: $imageItem, matching: .images) {
                uploadTile(title: "点击选择图片", subtitle: "从相册选取一张含车牌的照片")
            }
            .buttonStyle(.plain)
            .padding(.horizontal)
        }
    }

    @ViewBuilder
    private var videoPreview: some View {
        if let currentPlayer = player {
            VideoPlayerCard(player: currentPlayer,
                            samples: videoSamples,
                            isPlaying: $isPlaying)
                .padding(.horizontal)
        } else {
            PhotosPicker(selection: $videoItem, matching: .videos) {
                uploadTile(title: "点击选择视频", subtitle: "从相册选取一段含车牌的短片")
            }
            .buttonStyle(.plain)
            .padding(.horizontal)
        }
    }

    // MARK: - Video player card

    /// AVPlayerLayer preview + transport button, kept as a standalone view so the
    /// `player` binding is unwrapped exactly once (the previous inline `guard let player`
    /// / `if let player` re-unwrapped an already non-optional value).
    private struct VideoPlayerCard: View {
        let player: AVPlayer
        let samples: [PlateVideoSample]
        @Binding var isPlaying: Bool

        var body: some View {
            ZStack(alignment: .topTrailing) {
                VideoPlatePreview(player: player, samples: samples)
                    .frame(height: 340)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                    .overlay(alignment: .bottom) {
                        Text("视频预览（彩色区域为采样帧识别到的车牌，结果见下方列表）")
                            .font(.caption2).foregroundStyle(.white)
                            .padding(6)
                            .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 8))
                            .padding(8)
                    }
                Button(action: togglePlayback) {
                    Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 30))
                        .symbolRenderingMode(.palette)
                        .foregroundStyle(.white, .black.opacity(0.45))
                }
                .padding(10)
            }
            .onReceive(NotificationCenter.default.publisher(for: .AVPlayerItemDidPlayToEndTime)) { note in
                // Keep the transport icon in sync when the clip finishes on its own.
                if (note.object as? AVPlayerItem) == player.currentItem {
                    isPlaying = false
                }
            }
        }

        private func togglePlayback() {
            if player.rate == 0 {
                // Parked at the end (or not yet started): rewind, then play.
                if let item = player.currentItem,
                   item.duration.flags.contains(.valid),
                   CMTimeCompare(item.currentTime(), item.duration) >= 0 {
                    player.seek(to: .zero)
                }
                player.play()
                isPlaying = true
            } else {
                player.pause()
                isPlaying = false
            }
        }
    }

    private func uploadTile(title: String, subtitle: String) -> some View {
        VStack(spacing: 12) {
            RoundedRectangle(cornerRadius: 16)
                .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [8]))
                .foregroundStyle(.gray)
                .frame(height: 200)
                .overlay {
                    VStack(spacing: 10) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 30)).foregroundStyle(appAccent)
                        Text(title).font(.subheadline.bold()).foregroundStyle(.secondary)
                        Text(subtitle).font(.caption).foregroundStyle(.tertiary)
                    }
                }
        }
    }

    // MARK: - Results

    @ViewBuilder
    private var resultsSection: some View {
        if source == .camera {
            CameraResultsView(controller: cameraController)
        } else if !results.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Text("共识别到 \(results.count) 块车牌").font(.headline)
                    Spacer(minLength: 0)
                    Button("重新选择") { clearResults() }
                        .font(.subheadline).foregroundStyle(appAccent)
                }
                .padding(.horizontal)
                ForEach(Array(results.enumerated()), id: \.element.id) { pair in
                    PlateCardView(result: pair.element, index: pair.offset + 1)
                        .padding(.horizontal)
                }
            }
        } else if isBusy {
            ProgressView(source == .video ? "视频识别中…" : "识别中…")
                .padding(.horizontal)
        } else if (source == .image && displayImage != nil) || (source == .video && videoURL != nil) {
            // Media loaded but nothing recognized → give explicit feedback so it does not
            // look like "nothing happened". Show the detection count: >0 means the model
            // found candidates but OCR/validation rejected them.
            Text(lastDetectionCount > 0
                 ? "未识别到车牌：模型检测到 \(lastDetectionCount) 个候选框，但 OCR 未确认（请尝试切换 OCR 引擎或识别模式）。"
                 : "未检测到车牌。可尝试更换素材、切换识别模式或 OCR 引擎后重试。")
                .font(.footnote).foregroundStyle(.secondary).padding(.horizontal)
        }
    }

    private func clearResults() {
        // Remove the temp copy the video import created so repeated selections don't
        // accumulate .mov files in the system temporary directory.
        if let url = videoURL { try? FileManager.default.removeItem(at: url) }
        results = []
        imageData = nil
        displayImage = nil
        videoURL = nil
        player?.pause()
        player = nil
        videoSamples = []
        lastDetectionCount = 0
    }

    // MARK: - Run

    private func runImage(_ item: PhotosPickerItem) async {
        // Clear the selection once we've consumed it, so re-selecting the same asset
        // (a common re-test) produces a fresh change and re-triggers recognition.
        defer { if imageItem == item { imageItem = nil } }
        do {
            guard let data = try await item.loadTransferable(type: Data.self) else {
                errorMessage = "无法读取图片"
                return
            }
            // Honor EXIF orientation up front and bake it into a CGImage so the image we
            // display and the image we feed to the detector share one coordinate space.
            guard let uiImage = UIImage(data: data),
                  let oriented = uiImage.cgImageWithOrientationApplied() else {
                errorMessage = "无法读取图片"
                return
            }
            imageData = data
            displayImage = UIImage(cgImage: oriented)
            videoURL = nil
            videoSamples = []
            results = []
            lastDetectionCount = 0
            errorMessage = nil
            isBusy = true
            let pixelBuffer = try recognizer.prepareFrame(from: oriented)
            // Tiled detection: still photos (especially multi-plate collages) keep their
            // plates large enough for the fixed-640 pose model to see them. The tiled
            // pass falls back to a single detect when the image is small.
            let frame = try await recognizer.recognizeFrame(pixelBuffer, mode: mode, tiled: true)
            results = frame.plates
            // Surface detection vs. OCR outcome so a silent failure is diagnosable:
            lastDetectionCount = frame.detections.count
            isBusy = false
        } catch {
            isBusy = false
            errorMessage = error.localizedDescription
        }
    }

    private func runVideo(_ item: PhotosPickerItem) async {
        defer { if videoItem == item { videoItem = nil } }
        do {
            guard let movie = try await item.loadTransferable(type: ImportedMovie.self) else {
                errorMessage = "无法读取视频"
                return
            }
            let dst = movie.url
            if let old = videoURL, old != dst { try? FileManager.default.removeItem(at: old) }

            videoURL = dst
            imageData = nil
            displayImage = nil
            results = []
            videoSamples = []
            lastDetectionCount = 0
            errorMessage = nil
            isBusy = true
            videoProgress = 0
            let samples = try await recognizer.recognizeVideoSamples(at: dst, mode: mode, maxSamples: 30) { progress in
                Task { @MainActor in videoProgress = progress }
            }
            videoSamples = samples
            results = PlatePostProcessor.deduplicate(samples.flatMap(\.plates))
            lastDetectionCount = samples.reduce(0) { $0 + $1.detections.count }
            isBusy = false
        } catch {
            isBusy = false
            errorMessage = error.localizedDescription
        }
    }
}

// MARK: - Source segmented control (拍照 / 相册 / 视频)

private struct SourceSegment: View {
    @Binding var selection: MediaSource

    private let order: [MediaSource] = [.camera, .image, .video]
    private let labels: [MediaSource: String] = [
        .camera: "拍照", .image: "相册", .video: "视频"
    ]
    private let icons: [MediaSource: String] = [
        .camera: "camera", .image: "photo.on.rectangle", .video: "video.fill"
    ]

    private var index: Int { order.firstIndex(of: selection) ?? 0 }

    var body: some View {
        GeometryReader { geo in
            let pad: CGFloat = 4
            let colW = (geo.size.width - pad * 2) / 3
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 11)
                    .fill(Color(.secondarySystemBackground))
                    .frame(width: colW, height: geo.size.height - pad * 2)
                    .offset(x: pad + CGFloat(index) * colW)
                    .animation(.spring(response: 0.3, dampingFraction: 0.7), value: index)
                HStack(spacing: 0) {
                    ForEach(order, id: \.self) { src in
                        Button {
                            selection = src
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: icons[src] ?? "")
                                Text(labels[src] ?? "")
                            }
                            .font(.system(size: 13.5, weight: .semibold))
                            .foregroundStyle(selection == src ? appAccent : .secondary)
                            .frame(maxWidth: .infinity)
                        }
                    }
                }
                .padding(pad)
            }
            .background(Color(.tertiarySystemBackground), in: RoundedRectangle(cornerRadius: 14))
        }
        .frame(height: 44)
    }
}

// MARK: - Bottom settings sheet

private struct SettingsSheet: View {
    @Binding var mode: RecognitionMode
    @Binding var ocrEngine: OCREngine
    @Binding var useRGB: Bool
    @Binding var enhancements: RecognitionEnhancements
    var onDone: () -> Void

    /// Debug-only options (RGB/BGR channel order) are collapsed by default so the sheet's
    /// first screen stays focused on the controls a normal user needs.
    @State private var showDebug = false

    var body: some View {
        VStack(spacing: 12) {
            Capsule().fill(Color(.systemGray4))
                .frame(width: 38, height: 5)
                .padding(.top, 8)
            Text("高级设置").font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            Text("常用参数已前置，其余为进阶选项（默认即可）")
                .font(.caption).foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)

            settingRow(title: "识别模式", subtitle: "极速响应更快；精准更稳（默认极速）") {
                Picker("识别模式", selection: $mode) {
                    Text("极速").tag(RecognitionMode.fast)
                    Text("精准").tag(RecognitionMode.accurate)
                }
                .pickerStyle(.segmented)
                .frame(width: 180)
                .tint(appAccent)
            }

            settingRow(title: "OCR 引擎", subtitle: "Apple Vision 系统级、零依赖；PaddleOCR 精度更高") {
                Picker("OCR 引擎", selection: $ocrEngine) {
                    ForEach(OCREngine.allCases, id: \.self) { Text($0.displayName) }
                }
                .pickerStyle(.segmented)
                .frame(width: 190)
                .tint(appAccent)
            }

            DisclosureGroup("调试选项", isExpanded: $showDebug) {
                settingRow(title: "输入通道顺序", subtitle: "默认 RGB（与 PC 参考一致）；仅当检测明显错乱时切 BGR") {
                    Picker("输入通道", selection: $useRGB) {
                        Text("RGB").tag(true)
                        Text("BGR").tag(false)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 130)
                    .tint(appAccent)
                }
            }
            .tint(.secondary)
            .font(.subheadline)
            .padding(14)
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 14))
            VStack(alignment: .leading, spacing: 8) {
                Toggle(isOn: $enhancements.isEnabled) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("强化识别（SDK Enhanced Mode）")
                        Text("折叠感知排序 · 数字段严格校验 · 关键点四边形叠层")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
                if enhancements.isEnabled {
                    VStack(alignment: .leading, spacing: 6) {
                        Toggle("折叠感知排序（修正 3 位 / E·H 误读）",
                               isOn: $enhancements.foldAwareScoring)
                        Toggle("数字段严格校验（拒绝前导 0）",
                               isOn: $enhancements.strictNumericPart)
                        Toggle("关键点四边形叠层（用 pose 角点）",
                               isOn: $enhancements.keypointQuads)
                    }
                    .font(.footnote)
                    .padding(.top, 8)
                }
            }
            .padding(14)
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 14))

            Button("完成") { onDone() }
                .font(.headline.bold())
                .frame(maxWidth: .infinity)
                .padding(14)
                .background(appAccent, in: RoundedRectangle(cornerRadius: 14))
                .foregroundStyle(.white)
        }
        .padding(20)
        .background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: 24, style: .continuous))
        .shadow(color: .black.opacity(0.2), radius: 20, y: -8)
    }

    private func settingRow<Content: View>(title: String, subtitle: String,
                                           @ViewBuilder content: () -> Content) -> some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.subheadline)
                Text(subtitle).font(.caption2).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            content()
        }
        .padding(14)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 14))
    }
}

// MARK: - German-plate result card

/// Renders a German plate the way the HTML mockup does: blue EU strip (stars + D),
/// white field with bold plate text, and an optional年检圆章 (E / H suffix).
private struct GermanPlateView: View {
    let text: String

    private var hasSeal: Bool { text.hasSuffix("E") || text.hasSuffix("H") }

    var body: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                Text("★★★★★").font(.system(size: 4)).foregroundStyle(Color(red: 1, green: 0.83, blue: 0))
                Text("★★★★★").font(.system(size: 4)).foregroundStyle(Color(red: 1, green: 0.83, blue: 0))
                Text("D").font(.system(size: 9, weight: .bold)).foregroundStyle(.white)
            }
            .frame(width: 18)
            .background(Color(red: 0.04, green: 0.23, blue: 0.57))

            Text(text)
                .font(.system(size: 17, weight: .bold, design: .default))
                .foregroundStyle(.black)
                .padding(.horizontal, 8)
                .frame(height: 30)
                .background(Color.white)

            if hasSeal {
                Circle()
                    .fill(Color.clear)
                    .frame(width: 16, height: 16)
                    .overlay(
                        Circle()
                            .fill(AngularGradient(colors: [.red, .orange, .blue, .green, .red],
                                                 center: .center))
                            .overlay(Circle().stroke(.white, lineWidth: 2))
                    )
                    .padding(.trailing, 5)
            }
        }
        .frame(height: 30)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.gray.opacity(0.4), lineWidth: 1))
    }
}

private struct ConfidenceBar: View {
    let value: Float

    var body: some View {
        GeometryReader { geo in
            Capsule().fill(Color(.systemGray5))
                .overlay(alignment: .leading) {
                    Capsule().fill(Color.green)
                        .frame(width: max(0, min(geo.size.width, geo.size.width * CGFloat(value))))
                }
        }
        .frame(height: 5)
    }
}

/// One recognition result rendered as a card: stable plate colour swatch (matches the
/// on-screen overlay), German-plate visual, index, confidence and candidates.
private struct PlateCardView: View {
    let result: PlateResult
    let index: Int

    private var paletteColor: PlateOverlayColor {
        PlateOverlayPalette.color(forText: result.text, box: result.boundingBox)
    }
    private var color: Color { Color(paletteColor.cgColor) }

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            RoundedRectangle(cornerRadius: 4)
                .fill(color)
                .frame(width: 8, height: 38)
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 10) {
                    GermanPlateView(text: result.text)
                    Spacer(minLength: 0)
                    Text("#\(index)").font(.caption).foregroundStyle(.secondary)
                    Text("\(Int(result.confidence * 100))%").font(.caption).foregroundStyle(.secondary)
                }
                ConfidenceBar(value: result.confidence)
                if result.candidates.count > 1 {
                    Text("候选: " + result.candidates.prefix(3).map(\.text).joined(separator: " / "))
                        .font(.caption2).foregroundStyle(.tertiary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(12)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
    }
}

/// Live camera results, shown as the same German-plate cards below the preview.
struct CameraResultsView: View {
    let controller: CameraController

    var body: some View {
        if controller.liveResults.isEmpty {
            EmptyView()
        } else {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Text("实时识别 \(controller.liveResults.count) 块车牌").font(.headline)
                    Spacer(minLength: 0)
                }
                .padding(.horizontal)
                ForEach(Array(controller.liveResults.enumerated()), id: \.element.id) { pair in
                    PlateCardView(result: pair.element, index: pair.offset + 1)
                        .padding(.horizontal)
                }
            }
        }
    }
}

private struct ImportedMovie: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .movie) { received in
            let destination = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString).appendingPathExtension("mov")
            try FileManager.default.copyItem(at: received.file, to: destination)
            return ImportedMovie(url: destination)
        }
    }
}

/// Plate overlay for an uploaded image: keeps the original detection box and adds a
/// semi-transparent solid tint plus a matching solid outline over the plate area.
struct PlateQuadOverlay: View {
    let results: [PlateResult]

    var body: some View {
        GeometryReader { geometry in
            ForEach(Array(results.enumerated()), id: \.element.id) { pair in
                let index = pair.offset
                let result = pair.element
                let tint = Color(cgColor: Self.color(for: result).cgColor)

                PlateQuadShape(quad: result.plateQuad)
                    .fill(tint.opacity(PlateOverlayPalette.fillAlpha))
                    .overlay {
                        PlateQuadShape(quad: result.plateQuad)
                            .stroke(tint, lineWidth: PlateOverlayPalette.strokeWidth)
                    }
                    .allowsHitTesting(false)

                Text("#\(index + 1) \(result.text)")
                    .font(.caption)
                    .padding(2)
                    .background(tint.opacity(0.95))
                    .foregroundStyle(.white)
                    .fixedSize()
                    .position(x: Self.centerX(result, in: geometry.size),
                              y: max(10, Self.topY(result, in: geometry.size) - 10))
            }
        }
    }

    static func color(for result: PlateResult) -> PlateOverlayColor {
        PlateOverlayPalette.color(for: PlateOverlayPalette.key(text: result.text, box: result.boundingBox))
    }

    /// Label anchors derived from the plate quad (not the box) so the chip sits above the
    /// tilted plate outline rather than above a taller bounding rectangle.
    private static func topY(_ result: PlateResult, in size: CGSize) -> CGFloat {
        (result.plateQuad.map(\.y).min() ?? result.boundingBox.minY) * size.height
    }

    private static func centerX(_ result: PlateResult, in size: CGSize) -> CGFloat {
        let xs = result.plateQuad.map(\.x)
        let mid = ((xs.min() ?? result.boundingBox.minX) + (xs.max() ?? result.boundingBox.maxX)) / 2
        return min(max(mid * size.width, 60), max(60, size.width - 60))
    }
}

/// Draws the plate outline (the 4 pose corners, or the box corners when there are no
/// keypoints) as a filled/stroked quadrilateral in the view's coordinate space.
struct PlateQuadShape: Shape {
    /// Normalized (top-left origin, 0...1) corner points.
    let quad: [CGPoint]

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let points = quad.map { CGPoint(x: $0.x * rect.width, y: $0.y * rect.height) }
        guard points.count >= 3 else { return path }
        path.addLines(points)
        path.closeSubpath()
        return path
    }
}

extension UIImage {
    /// Returns a CGImage whose pixels are physically rotated/flopped to match
    /// `imageOrientation`, so downstream coordinate math lines up with what SwiftUI
    /// actually draws. `cgImage` alone ignores the orientation tag.
    func cgImageWithOrientationApplied() -> CGImage? {
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        let rendered = renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }
        return rendered.cgImage
    }
}
