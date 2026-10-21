import SwiftUI
import UIKit

struct CameraView: View {
    @Bindable var controller: CameraController
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        VStack(spacing: 0) {
            if controller.isAuthorized && !controller.setupFailed {
                CameraPreview(controller: controller)
                    .overlay(alignment: .topLeading) {
                        HStack(spacing: 6) {
                            Circle()
                                .fill(controller.isAnalyzing ? .yellow : .green)
                                .frame(width: 8, height: 8)
                            Text(controller.isAnalyzing ? "正在识别" : "实时检测")
                            Spacer()
                            Text("框选 \(controller.liveDetections.count) · 识别 \(controller.liveResults.count)")
                        }
                        .font(.caption.monospaced())
                        .foregroundStyle(.white)
                        .padding(10)
                        .background(.black.opacity(0.65), in: RoundedRectangle(cornerRadius: 8))
                        .padding(10)
                    }
                    .overlay(alignment: .bottomLeading) {
                        Text("耗时 \(controller.elapsedMilliseconds) ms · 帧 \(Int(controller.frameSize.width))×\(Int(controller.frameSize.height))")
                            .font(.caption.monospaced())
                            .foregroundStyle(.white)
                            .padding(8)
                            .background(.black.opacity(0.65), in: Capsule())
                            .padding(10)
                    }
                    .frame(height: 330)

                HStack {
                    Label("识别终端", systemImage: "terminal")
                        .font(.subheadline.bold())
                    Text("\(controller.recognitionHistory.count)")
                        .font(.caption.monospaced())
                        .foregroundStyle(.green)
                    Spacer()
                    Button(role: .destructive) {
                        controller.clearRecognitionHistory()
                    } label: {
                        Label("清空记录", systemImage: "trash")
                            .font(.caption)
                    }
                    .disabled(controller.recognitionHistory.isEmpty)
                    Toggle("实时", isOn: $controller.liveEnabled)
                        .labelsHidden()
                        .fixedSize()
                }
                .foregroundStyle(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 9)
                .background(Color(white: 0.16))

                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 5) {
                        if controller.recognitionHistory.isEmpty {
                            Text("等待识别车牌…")
                                .foregroundStyle(.gray)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.vertical, 8)
                        } else {
                            ForEach(controller.recognitionHistory) { entry in
                                HStack(alignment: .firstTextBaseline, spacing: 8) {
                                    Text(entry.date, format: .dateTime.hour().minute().second())
                                        .foregroundStyle(.gray)
                                    Text(entry.detectionNumber > 0 ? "#\(entry.detectionNumber)" : "#—")
                                        .foregroundStyle(.cyan)
                                    Text(entry.result.text)
                                        .fontWeight(.semibold)
                                        .foregroundStyle(.green)
                                    Text(String(format: "%.0f%%", entry.result.confidence * 100))
                                        .foregroundStyle(.white.opacity(0.7))
                                    Spacer(minLength: 0)
                                }
                                .font(.system(size: 12, design: .monospaced))
                                .padding(.vertical, 2)
                                .accessibilityElement(children: .combine)
                                .accessibilityLabel("\(entry.result.text), 检测框 \(entry.detectionNumber), 置信度 \(Int(entry.result.confidence * 100))%")
                            }
                        }
                    }
                    .padding(10)
                }
                .defaultScrollAnchor(.bottom)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.black)
                .foregroundStyle(.white)

                HStack(spacing: 12) {
                    Text("黄框：检测框")
                        .foregroundStyle(.yellow)
                    Text("彩色区域：已识别车牌")
                        .foregroundStyle(.green)
                    Spacer()
                    // Diagnostic echo: shows the analyzed buffer as-is with the same
                    // boxes, so a preview-vs-buffer mismatch is visible on device.
                    Text("回显")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Toggle("", isOn: $controller.debugEchoEnabled)
                        .labelsHidden()
                        .fixedSize()
                    Text("\(controller.elapsedMilliseconds) ms")
                        .foregroundStyle(.secondary)
                }
                .font(.caption2.monospaced())
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(Color(white: 0.12))
            } else {
                ContentUnavailableView("相机不可用", systemImage: "camera",
                    description: Text(controller.setupFailed ? "相机初始化失败，请检查设备。" : "请允许相机访问权限。"))
            }
            if let message = controller.errorMessage {
                Text(message).font(.footnote).foregroundStyle(.red).padding(8)
            }
        }
        .background(Color.black)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .task { await controller.setup() }
        .onDisappear { controller.stop() }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active { Task { await controller.setup() } }
            else { controller.stop() }
        }
    }
}
