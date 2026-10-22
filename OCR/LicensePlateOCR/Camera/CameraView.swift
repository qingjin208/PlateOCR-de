import SwiftUI
import UIKit

struct CameraView: View {
    @Bindable var controller: CameraController
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        VStack(spacing: 0) {
            if controller.isAuthorized && !controller.setupFailed {
                // The live preview fills the available space; the status, timing and a
                // small legend float on top of it.
                CameraPreview(controller: controller)
                    .overlay(alignment: .topLeading) { statusPill }
                    .overlay(alignment: .topTrailing) { latencyPill }
                    .overlay(alignment: .bottom) { legend }
                    .frame(maxHeight: .infinity)

                // Compact toolbar: live toggle, clear, history count, diagnostic echo.
                toolbar
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color(white: 0.12))

                // Recognition history log, scrolled to the latest entry.
                historyList
                    .frame(maxHeight: 200)
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

    private var statusPill: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(controller.isAnalyzing ? .yellow : .green)
                .frame(width: 8, height: 8)
            Text(controller.isAnalyzing ? "识别中" : "实时检测")
            Text("框 \(controller.liveDetections.count) · 牌 \(controller.liveResults.count)")
                .foregroundStyle(.secondary)
        }
        .font(.caption.monospaced())
        .foregroundStyle(.white)
        .padding(8)
        .background(.black.opacity(0.6), in: Capsule())
        .padding(10)
    }

    private var latencyPill: some View {
        Text("\(controller.elapsedMilliseconds) ms · \(Int(controller.frameSize.width))×\(Int(controller.frameSize.height))")
            .font(.caption.monospaced())
            .foregroundStyle(.white)
            .padding(8)
            .background(.black.opacity(0.6), in: Capsule())
            .padding(10)
    }

    private var legend: some View {
        HStack(spacing: 12) {
            Label {
                Text("检测框").foregroundStyle(.white)
            } icon: {
                RoundedRectangle(cornerRadius: 2).fill(.yellow).frame(width: 10, height: 10)
            }
            Label {
                Text("已识别车牌").foregroundStyle(.white)
            } icon: {
                RoundedRectangle(cornerRadius: 2).fill(.green).frame(width: 10, height: 10)
            }
        }
        .font(.caption2)
        .padding(6)
        .background(.black.opacity(0.55), in: Capsule())
        .padding(10)
    }

    private var toolbar: some View {
        HStack(spacing: 12) {
            Toggle("实时", isOn: $controller.liveEnabled)
                .labelsHidden()
                .fixedSize()
            Button(role: .destructive) {
                controller.clearRecognitionHistory()
            } label: {
                Label("清空", systemImage: "trash")
            }
            .disabled(controller.recognitionHistory.isEmpty)
            Spacer()
            Text("\(controller.recognitionHistory.count) 条")
                .font(.caption.monospaced())
                .foregroundStyle(.green)
            HStack(spacing: 4) {
                Text("回显").font(.caption2).foregroundStyle(.secondary)
                Toggle("", isOn: $controller.debugEchoEnabled)
                    .labelsHidden()
                    .fixedSize()
            }
        }
        .foregroundStyle(.white)
    }

    private var historyList: some View {
        VStack(alignment: .leading, spacing: 0) {
            if controller.recognitionHistory.isEmpty {
                Text("等待识别车牌…")
                    .foregroundStyle(.gray)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 4) {
                        ForEach(controller.recognitionHistory) { entry in
                            historyRow(entry)
                        }
                    }
                    .padding(10)
                }
                .defaultScrollAnchor(.bottom)
            }
        }
        .frame(maxWidth: .infinity)
        .background(Color.black)
    }

    private func historyRow(_ entry: CameraRecognitionLogEntry) -> some View {
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
    }
}
