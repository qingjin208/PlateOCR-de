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

struct ContentView: View {
    // Placeholder pipeline: official Ultralytics `yolov8n` (COCO) detects *cars*
    // (class 2); the SDK then OCRs each car region and keeps only German-plate text.
    // Swap to your own model later with e.g.
    //   LicensePlateRecognizer(modelName: "license_plate_yolo", config: .germanPlate)
    private let recognizer: LicensePlateRecognizing
    @State private var source: MediaSource = .image
    @State private var cameraController: CameraController

    /// One recognizer is shared by the image/video pipeline and the live camera
    /// so the YOLO model is loaded only once.
    init() {
        let rec = LicensePlateRecognizer(modelName: "yolov8n", config: .cocoCar)
        self.recognizer = rec
        _cameraController = State(initialValue: CameraController(recognizer: rec))
    }

    @State private var imageItem: PhotosPickerItem?
    @State private var videoItem: PhotosPickerItem?
    @State private var imageData: Data?
    @State private var videoURL: URL?

    @State private var mode: RecognitionMode = .fast   // default = fast
    @State private var results: [PlateResult] = []
    @State private var isBusy = false
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

                    if isBusy {
                        ProgressView(source == .video ? "视频识别中…" : "识别中…")
                            .padding(.horizontal)
                    }

                    if let errorMessage {
                        Text(errorMessage)
                            .foregroundStyle(.red)
                            .font(.footnote)
                            .padding(.horizontal)
                    }

                    // Preview + overlay (image) or result note (video)
                    if source == .image, let data = imageData, let uiImage = UIImage(data: data) {
                        Image(uiImage: uiImage)
                            .resizable()
                            .scaledToFit()
                            .overlay {
                                ForEach(results) { result in
                                    ResultBox(rect: result.boundingBox)
                                        .stroke(.red, lineWidth: 2)
                                        .overlay(alignment: .topLeading) {
                                            Text(result.text)
                                                .font(.caption)
                                                .padding(2)
                                                .background(.red.opacity(0.8))
                                                .foregroundStyle(.white)
                                                .offset(y: -18)
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
    }

    // MARK: - Run

    private func runImage(_ item: PhotosPickerItem) async {
        do {
            guard let data = try await item.loadTransferable(type: Data.self),
                  let uiImage = UIImage(data: data),
                  let cg = uiImage.cgImage else {
                errorMessage = "无法读取图片"
                return
            }
            imageData = data
            videoURL = nil
            results = []
            errorMessage = nil
            isBusy = true
            results = try await recognizer.recognize(in: cg, mode: mode)
            isBusy = false
        } catch {
            isBusy = false
            errorMessage = error.localizedDescription
        }
    }

    private func runVideo(_ item: PhotosPickerItem) async {
        do {
            // Load the picked video as a file URL. `Movie` is not reliably
            // exposed by PhotosUI on every SDK/iOS slice, so we use `URL.self`
            // (PhotosUI provides a URL representation for video items) and copy
            // it to a stable temporary location so the file outlives the transfer.
            guard let sourceURL = try await item.loadTransferable(type: URL.self) else {
                errorMessage = "无法读取视频"
                return
            }
            let dst = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension(sourceURL.pathExtension.isEmpty ? "mov" : sourceURL.pathExtension)
            try? FileManager.default.removeItem(at: dst)
            try FileManager.default.copyItem(at: sourceURL, to: dst)

            videoURL = dst
            imageData = nil
            results = []
            errorMessage = nil
            isBusy = true
            results = try await recognizer.recognizeVideo(at: dst, mode: mode, maxSamples: 30, progress: nil)
            isBusy = false
        } catch {
            isBusy = false
            errorMessage = error.localizedDescription
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
