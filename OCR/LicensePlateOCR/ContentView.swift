/*
 License Plate OCR SDK — Demo App
 
 Abstract:
 Lets the user upload an image or a video, pick `fast` / `accurate` recognition, and
 shows the detected German plates with bounding boxes (image) or a list (video).
 */

import SwiftUI
import PhotosUI
import CoreTransferable
import CoreGraphics
import Vision
import CoreImage
import UniformTypeIdentifiers
import UIKit
import AVFoundation

struct ContentView: View {
    // Share the bundled plate detector and Apple OCR across all media sources.
    // Held in `@State` (not a plain `let`) so re-evaluating the View cannot reload the
    // ONNX model or leave the SDK talking to a recognizer the UI no longer owns.
    @State private var recognizer: LicensePlateRecognizer
    @State private var source: MediaSource = .image
    @State private var cameraController: CameraController
    /// Opt-in SDK strengthening. One master switch in the UI plus three isolated
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
    }

    @State private var imageItem: PhotosPickerItem?
    @State private var videoItem: PhotosPickerItem?
    @State private var imageData: Data?
    /// Image actually shown in the overlay. Baked to a physically oriented CGImage so
    /// its coordinate space matches what the detector/render pipeline uses.
    @State private var displayImage: UIImage?
    @State private var videoURL: URL?

    @State private var mode: RecognitionMode = .fast   // default = fast
    @State private var ocrEngine: OCREngine = .appleVision   // user-selected OCR engine
    /// Input channel order for the ONNX detector. RGB matches Ultralytics training and the
    /// PC reference pipeline; switch to BGR only if an export needs it.
    @State private var useRGB: Bool = true
    @State private var results: [PlateResult] = []
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

    enum MediaSource: String, CaseIterable {
        case image = "图片"
        case video = "视频"
        case camera = "相机"
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    // Source picker: image or video
                    Picker("来源", selection: $source) {
                        ForEach(MediaSource.allCases, id: \.self) { Text($0.rawValue) }
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal)

                    // Upload control (image / video) or live camera
                    if source == .image {
                        PhotosPicker("选择图片", selection: $imageItem, matching: .images)
                            .buttonStyle(.borderedProminent)
                    } else if source == .video {
                        PhotosPicker("选择视频", selection: $videoItem, matching: .videos)
                            .buttonStyle(.borderedProminent)
                    } else {
                        CameraView(controller: cameraController)
                            .frame(height: 540)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                            .padding(.horizontal)
                    }

                    // Mode picker (fast / accurate)
                    Picker("识别模式", selection: $mode) {
                        ForEach(RecognitionMode.allCases, id: \.self) { Text($0.rawValue.capitalized) }
                    }
                    .pickerStyle(.segmented)
                    .overlay(Capsule().stroke(.blue, lineWidth: 1))
                    .padding(.horizontal)

                    // OCR engine picker (Apple Vision / PaddleOCR) — user choice is passed to the SDK.
                    Picker("OCR 引擎", selection: $ocrEngine) {
                        ForEach(OCREngine.allCases, id: \.self) { Text($0.displayName) }
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal)

                    // Input channel order for the ONNX detector. RGB matches Ultralytics
                    // training and the PC reference; BGR is kept as an escape hatch.
                    Picker("输入通道顺序", selection: $useRGB) {
                        Text("RGB").tag(true)
                        Text("BGR").tag(false)
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal)
                    .overlay(alignment: .bottom) {
                        Text("默认 RGB（与 PC 参考一致）；仅当检测明显错乱时切到 BGR")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .padding(.top, 2)
                    }

                    // SDK enhancement master switch. The whole struct is handed to the SDK
                    // on every change; recognition picks it up on the next pass (image,
                    // video and live camera all share this recognizer).
                    Toggle(isOn: $enhancements.isEnabled) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("强化识别（SDK Enhanced Mode）")
                            Text("折叠感知排序 · 数字段严格校验 · 关键点四边形叠层")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.horizontal)

                    if enhancements.isEnabled {
                        VStack(alignment: .leading, spacing: 4) {
                            Toggle("折叠感知排序（修正 3 位数字 + E/H 误读）",
                                   isOn: $enhancements.foldAwareScoring)
                            Toggle("数字段严格校验（拒绝前导 0）",
                                   isOn: $enhancements.strictNumericPart)
                            Toggle("关键点四边形叠层（用 pose 角点）",
                                   isOn: $enhancements.keypointQuads)
                        }
                        .font(.footnote)
                        .padding(.horizontal)
                    }

                    if isBusy {
                        if source == .video {
                            ProgressView("视频识别中…", value: videoProgress).padding(.horizontal)
                        } else {
                            ProgressView("识别中…").padding(.horizontal)
                        }
                    }

                    // On-device diagnostic: confirms the ONNX pose detector actually loaded
                    // (vs. silently fell back to the legacy Core ML / failing stub), and
                    // which enhancement flags the SDK itself currently holds.
                    Text("检测器：\(recognizer.detectorInfo)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal)
                    Text("强化模式（SDK 侧）：\(recognizer.enhancements.summary)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal)

                    if let errorMessage {
                        Text(errorMessage)
                            .foregroundStyle(.red)
                            .font(.footnote)
                            .padding(.horizontal)
                    }

                    // Preview + overlay (image) or video player with per-frame plate tint
                    if source == .image, let uiImage = displayImage {
                        Image(uiImage: uiImage)
                            .resizable()
                            .scaledToFit()
                            .overlay { PlateQuadOverlay(results: results) }
                            .padding(.horizontal)
                    } else if source == .video, let player {
                        ZStack(alignment: .topTrailing) {
                            VideoPlatePreview(player: player, samples: videoSamples)
                                .frame(height: 340)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                                .overlay(alignment: .bottom) {
                                    Text("视频预览（彩色区域为采样帧识别到的车牌，结果见下方列表）")
                                        .font(.caption2)
                                        .foregroundStyle(.white)
                                        .padding(6)
                                        .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 8))
                                        .padding(8)
                                }
                            Button {
                                if player.rate == 0 {
                                    player.play()
                                    isPlaying = true
                                } else {
                                    player.pause()
                                    isPlaying = false
                                }
                            } label: {
                                Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                                    .font(.system(size: 30))
                                    .symbolRenderingMode(.palette)
                                    .foregroundStyle(.white, .black.opacity(0.45))
                            }
                            .padding(10)
                        }
                    }

                    // Results
                    if !results.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("识别结果（\(results.count)）").font(.headline)
                            ForEach(Array(results.enumerated()), id: \.element.id) { pair in
                                let index = pair.offset
                                let result = pair.element
                                VStack(alignment: .leading, spacing: 2) {
                                    HStack(spacing: 6) {
                                        // Same colour as the plate tint in the overlay.
                                        RoundedRectangle(cornerRadius: 3)
                                            .fill(Color(cgColor: plateColor(for: result).cgColor))
                                            .frame(width: 12, height: 12)
                                        Text("#\(index + 1)")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                        Text(result.text)
                                            .font(.title3.bold())
                                    }
                                    Text(String(format: "置信度 %.2f", result.confidence))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                    if result.candidates.count > 1 {
                                        Text("候选: " + result.candidates.prefix(3).map(\.text).joined(separator: " / "))
                                            .font(.caption2)
                                            .foregroundStyle(.tertiary)
                                    }
                                    if let corrected = result.correctedPlateImage {
                                        Image(uiImage: UIImage(cgImage: corrected))
                                            .resizable()
                                            .scaledToFit()
                                            .frame(maxHeight: 64)
                                            .background(Color.black)
                                            .clipShape(RoundedRectangle(cornerRadius: 6))
                                            .padding(.top, 4)
                                    }
                                }
                                .padding(8)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color(.secondarySystemBackground))
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                            }
                        }
                        .padding(.horizontal)
                    }

                    // Media loaded but nothing recognized → give explicit feedback so it
                    // does not look like "nothing happened".
                    if results.isEmpty, !isBusy,
                       (source == .image && displayImage != nil) || (source == .video && videoURL != nil) {
                        Text("未识别到车牌。可尝试更换素材、切换识别模式或 OCR 引擎后重试。")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .padding(.horizontal)
                    }
                }
                .padding(.vertical)
            }
            .navigationTitle("德国车牌识别")
        }
        .onChange(of: imageItem) { _, newItem in
            if let newItem { Task { await runImage(newItem) } }
        }
        .onChange(of: videoItem) { _, newItem in
            if let newItem { Task { await runVideo(newItem) } }
        }
        .onChange(of: mode) { _, newMode in cameraController.mode = newMode }
        .onChange(of: ocrEngine) { _, newEngine in recognizer.ocrEngine = newEngine }
        .onChange(of: useRGB) { _, newValue in recognizer.useRGB = newValue }
        // Push the enhancement switches into the SDK: the live camera shares this same
        // recognizer instance, so one assignment covers image, video and camera.
        .onChange(of: enhancements) { _, newValue in recognizer.enhancements = newValue }
        .onChange(of: videoURL) { _, newURL in
            if let newURL {
                player?.pause()
                player = AVPlayer(url: newURL)
                isPlaying = false
            } else {
                player?.pause()
                player = nil
                isPlaying = false
            }
        }
        .onChange(of: source) { _, newSource in
            if newSource != .video {
                player?.pause()
                isPlaying = false
            }
        }
    }

    // MARK: - Run

    /// Stable colour for a plate: keyed on its text so the overlay tint and the list
    /// swatch always agree, and so a plate keeps its colour between runs.
    private func plateColor(for result: PlateResult) -> PlateOverlayColor {
        PlateQuadOverlay.color(for: result)
    }

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
            // (Previously the display used UIImage's auto-orientation while detection ran on
            // the raw, un-rotated pixels, so boxes landed in the wrong place.)
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
            errorMessage = nil
            isBusy = true
            let pixelBuffer = try recognizer.prepareFrame(from: oriented)
            results = try await recognizer.recognizeFrame(pixelBuffer, mode: mode).plates
            isBusy = false
        } catch {
            isBusy = false
            errorMessage = error.localizedDescription
        }
    }

    private func runVideo(_ item: PhotosPickerItem) async {
        // Clear the selection once we've consumed it, so re-selecting the same asset
        // (a common re-test) produces a fresh change and re-triggers recognition.
        defer { if videoItem == item { videoItem = nil } }
        do {
            // Copy the file during the transfer callback, before Photos releases it.
            guard let movie = try await item.loadTransferable(type: ImportedMovie.self) else {
                errorMessage = "无法读取视频"
                return
            }
            let dst = movie.url
            // Keep the new clip for previewing; delete the *previous* temp video instead of
            // the current one. (Previously `defer { removeItem(dst) }` deleted the file the
            // instant recognition finished, so the VideoPlayer had nothing left to play.)
            if let old = videoURL, old != dst { try? FileManager.default.removeItem(at: old) }

            videoURL = dst
            imageData = nil
            displayImage = nil
            results = []
            videoSamples = []
            errorMessage = nil
            isBusy = true
            videoProgress = 0
            // Keep every sampled frame so the player can tint the plates that were found
            // at the current playback position; the list below shows the deduplicated set.
            let samples = try await recognizer.recognizeVideoSamples(at: dst, mode: mode, maxSamples: 30) { progress in
                Task { @MainActor in videoProgress = progress }
            }
            videoSamples = samples
            results = PlatePostProcessor.deduplicate(samples.flatMap(\.plates))
            isBusy = false
        } catch {
            isBusy = false
            errorMessage = error.localizedDescription
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
                let box = result.boundingBox
                let tint = Color(cgColor: Self.color(for: result).cgColor)

                Rectangle()
                    .stroke(.red, lineWidth: 2)
                    .frame(width: box.width * geometry.size.width,
                           height: box.height * geometry.size.height)
                    .position(x: box.midX * geometry.size.width,
                              y: box.midY * geometry.size.height)

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
        // `default` is a class function here (imported from `+defaultFormat`), not a property.
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        let rendered = renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }
        return rendered.cgImage
    }
}
