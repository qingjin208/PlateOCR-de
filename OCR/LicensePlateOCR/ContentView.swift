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

struct ContentView: View {
    // Share the bundled plate detector and Apple OCR across all media sources.
    private let recognizer: LicensePlateRecognizer
    @State private var source: MediaSource = .image
    @State private var cameraController: CameraController

    /// One recognizer is shared by the image/video pipeline and the live camera
    /// so the YOLO model is loaded only once.
    init() {
        #if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc)
        // ONNX Runtime linked: use the trained German-plate YOLOv8-pose model directly.
        let rec = LicensePlateRecognizer(poseModelName: "De-plate-pose-500", engine: .appleVision)
        #else
        // ONNX Runtime not linked yet: fall back to the legacy Core ML detector until the
        // pose model + onnxruntime framework are added (see POSE_INTEGRATION.md).
        let rec = LicensePlateRecognizer()
        #endif
        self.recognizer = rec
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
    @State private var results: [PlateResult] = []
    @State private var isBusy = false
    @State private var videoProgress = 0.0
    @State private var errorMessage: String?

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

                    if isBusy {
                        if source == .video {
                            ProgressView("视频识别中…", value: videoProgress).padding(.horizontal)
                        } else {
                            ProgressView("识别中…").padding(.horizontal)
                        }
                    }

                    if let errorMessage {
                        Text(errorMessage)
                            .foregroundStyle(.red)
                            .font(.footnote)
                            .padding(.horizontal)
                    }

                    // Preview + overlay (image) or result note (video)
                    if source == .image, let uiImage = displayImage {
                        Image(uiImage: uiImage)
                            .resizable()
                            .scaledToFit()
                            .overlay {
                                GeometryReader { geometry in
                                    ForEach(results) { result in
                                        let box = result.boundingBox
                                        Rectangle().stroke(.red, lineWidth: 2)
                                            .frame(width: box.width * geometry.size.width,
                                                   height: box.height * geometry.size.height)
                                            .position(x: box.midX * geometry.size.width,
                                                      y: box.midY * geometry.size.height)
                                            Text(result.text)
                                                .font(.caption)
                                                .padding(2)
                                                .background(.red.opacity(0.8))
                                                .foregroundStyle(.white)
                                                .fixedSize()
                                                .position(x: box.midX * geometry.size.width,
                                                          y: max(10, box.minY * geometry.size.height - 10))
                                    }
                                }
                            }
                            .padding(.horizontal)
                    } else if source == .video, videoURL != nil {
                        Image(systemName: "film")
                            .resizable()
                            .scaledToFit()
                            .frame(width: 64, height: 64)
                            .foregroundStyle(.gray)
                            .overlay(alignment: .bottom) {
                                Text("视频已选择，点击上方重新选择可再次识别")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                    }

                    // Results
                    if !results.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("识别结果（\(results.count)）").font(.headline)
                            ForEach(results) { result in
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(result.text)
                                        .font(.title3.bold())
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
                }
                .padding(.vertical)
            }
            .navigationTitle("德国车牌识别")
        }
        .task(id: imageItem) {
            guard let item = imageItem else { return }
            await runImage(item)
        }
        .task(id: videoItem) {
            guard let item = videoItem else { return }
            await runVideo(item)
        }
        .onChange(of: mode) { _, newMode in cameraController.mode = newMode }
        .onChange(of: ocrEngine) { _, newEngine in recognizer.ocrEngine = newEngine }
    }

    // MARK: - Run

    private func runImage(_ item: PhotosPickerItem) async {
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
        do {
            // Copy the file during the transfer callback, before Photos releases it.
            guard let movie = try await item.loadTransferable(type: ImportedMovie.self) else {
                errorMessage = "无法读取视频"
                return
            }
            let dst = movie.url
            defer { try? FileManager.default.removeItem(at: dst) }

            videoURL = dst
            imageData = nil
            displayImage = nil
            results = []
            errorMessage = nil
            isBusy = true
            videoProgress = 0
            results = try await recognizer.recognizeVideo(at: dst, mode: mode, maxSamples: 30) { progress in
                Task { @MainActor in videoProgress = progress }
            }
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

/// Draws a bounding box from a normalized (top-left origin) rect.
struct ResultBox: Shape {
    let rect: CGRect
    func path(in rect: CGRect) -> Path {
        let normalized = NormalizedRect(x: rect.origin.x, y: rect.origin.y,
                                        width: rect.width, height: rect.height)
        let converted = normalized.toImageCoordinates(rect.size, origin: .upperLeft)
        return Path(converted)
    }
}

extension UIImage {
    /// Returns a CGImage whose pixels are physically rotated/flopped to match
    /// `imageOrientation`, so downstream coordinate math lines up with what SwiftUI
    /// actually draws. `cgImage` alone ignores the orientation tag.
    func cgImageWithOrientationApplied() -> CGImage? {
        let format = UIGraphicsImageRendererFormat.default
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        let rendered = renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }
        return rendered.cgImage
    }
}
