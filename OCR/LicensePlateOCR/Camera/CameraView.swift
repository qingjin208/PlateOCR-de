import SwiftUI
import UIKit

struct CameraView: View {
    @Bindable var controller: CameraController
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        VStack(spacing: 12) {
            if controller.isAuthorized && !controller.setupFailed {
                CameraPreview(controller: controller)
                    .overlay(alignment: .topLeading) {
                        Text(controller.isAnalyzing ? "识别中…" : "实时检测")
                            .padding(8)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                            .padding()
                    }
                Toggle("实时检测", isOn: $controller.liveEnabled).padding(.horizontal)
                Text("检测到 \(controller.liveDetections.count) 个车牌 · 已识别 \(controller.liveResults.count) 个 · \(controller.elapsedMilliseconds) ms")
                    .font(.caption).foregroundStyle(.secondary)
                if controller.liveResults.isEmpty {
                    Text("将车牌放入画面即可自动识别，无需拍摄")
                        .font(.footnote).foregroundStyle(.secondary)
                } else {
                    ForEach(controller.liveResults) { result in
                        LabeledContent(result.text, value: String(format: "%.0f%%", result.confidence * 100))
                            .padding(.horizontal)
                    }
                }
            } else {
                ContentUnavailableView("相机不可用", systemImage: "camera",
                    description: Text(controller.setupFailed ? "相机初始化失败，请检查设备。" : "请允许相机访问权限。"))
            }
            if let message = controller.errorMessage {
                Text(message).font(.footnote).foregroundStyle(.red).padding(.horizontal)
            }
        }
        .task { await controller.setup() }
        .onDisappear { controller.stop() }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active { Task { await controller.setup() } }
            else { controller.stop() }
        }
    }
}
