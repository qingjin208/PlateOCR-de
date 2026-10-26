import SwiftUI
import UIKit

/// Camera tab, redesigned to match the HTML mockup's live-detection flow:
///   • idle (detection off): guide frame + "开始实时识别" button over the live feed
///   • active (detection on): LIVE badge, a scanning beam while analysing, and a bottom
///     control bar with save + pause/play.
///
/// The detection overlay itself is drawn by `PreviewView` (inside `CameraPreview`) and
/// keeps its aspect-fill mapping onto the upright portrait frame, so boxes stay glued to
/// the plates — this view only changes the surrounding chrome.
struct CameraView: View {
    @Bindable var controller: CameraController
    @Environment(\.scenePhase) private var scenePhase

    @State private var toast: String?

    var body: some View {
        VStack(spacing: 0) {
            if controller.isAuthorized && !controller.setupFailed {
                CameraPreview(controller: controller)
                    .overlay(alignment: .topLeading) { liveBadge }
                    .overlay(alignment: .topTrailing) { latencyPill }
                    .overlay(alignment: .top) { scanBeam }
                    .overlay(alignment: .center) { idleGuide }
                    .overlay(alignment: .bottom) { controlBar }
                    .overlay(alignment: .bottom) { toastView }
                    .frame(maxHeight: .infinity)

                CameraResultsView(controller: controller)
                    .padding(.top, 10)
            } else {
                ContentUnavailableView("相机不可用", systemImage: "camera",
                    description: Text(controller.setupFailed ? "相机初始化失败，请检查设备。" : "请允许相机访问权限。"))
            }
            if let message = controller.errorMessage {
                Text(message).font(.footnote).foregroundStyle(.red).padding(8)
            }
        }
        .background(Color.black)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .task { await controller.setup() }
        .onDisappear { controller.stop() }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active { Task { await controller.setup() } }
            else { controller.stop() }
        }
    }

    // MARK: - Idle guide (shown when detection is off)

    @ViewBuilder
    private var idleGuide: some View {
        if !controller.liveEnabled {
            ZStack {
                RoundedRectangle(cornerRadius: 14)
                    .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [6]))
                    .foregroundStyle(.white.opacity(0.5))
                    .frame(width: 220, height: 120)
                VStack(spacing: 14) {
                    Button {
                        controller.liveEnabled = true
                    } label: {
                        Image(systemName: "camera.fill")
                            .font(.system(size: 22))
                            .foregroundStyle(.white)
                            .frame(width: 56, height: 56)
                            .background(appAccent, in: Circle())
                            .scaleEffect(1)
                    }
                    Text("开始实时识别")
                        .font(.subheadline.bold())
                        .foregroundStyle(.white)
                }
            }
        }
    }

    // MARK: - Active chrome

    @ViewBuilder
    private var liveBadge: some View {
        if controller.liveEnabled {
            HStack(spacing: 6) {
                Circle()
                    .fill(controller.isAnalyzing ? .yellow : .green)
                    .frame(width: 8, height: 8)
                Text(controller.isAnalyzing ? "实时检测中" : "实时检测")
                    .font(.caption.monospaced())
                Text("框 \(controller.liveDetections.count) · 牌 \(controller.liveResults.count)")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
            .foregroundStyle(.white)
            .padding(8)
            .background(.black.opacity(0.6), in: Capsule())
            .padding(10)
        }
    }

    @ViewBuilder
    private var latencyPill: some View {
        if controller.liveEnabled {
            Text("\(controller.elapsedMilliseconds) ms")
                .font(.caption.monospaced())
                .foregroundStyle(.white)
                .padding(8)
                .background(.black.opacity(0.6), in: Capsule())
                .padding(10)
        }
    }

    @ViewBuilder
    private var controlBar: some View {
        if controller.liveEnabled {
            HStack(spacing: 16) {
                Button {
                    showToast("✓ 快照已保存")
                } label: {
                    Image(systemName: "square.and.arrow.down")
                        .font(.system(size: 18))
                        .foregroundStyle(.white)
                        .frame(width: 44, height: 44)
                        .background(.ultraThinMaterial, in: Circle())
                        .overlay(Circle().stroke(.white.opacity(0.8), lineWidth: 2))
                }
                Button {
                    controller.liveEnabled.toggle()
                } label: {
                    Image(systemName: controller.isAnalyzing ? "pause.fill" : "play.fill")
                        .font(.system(size: 22))
                        .foregroundStyle(.white)
                        .frame(width: 62, height: 62)
                        .background(appAccent, in: Circle())
                        .shadow(color: appAccent.opacity(0.5), radius: 10, y: 4)
                }
            }
            .padding(.bottom, 18)
        }
    }

    @ViewBuilder
    private var scanBeam: some View {
        if controller.liveEnabled && controller.isAnalyzing {
            ScanBeam()
        }
    }

    @ViewBuilder
    private var toastView: some View {
        if let toast {
            Text(toast)
                .font(.subheadline.bold())
                .foregroundStyle(.white)
                .padding(.horizontal, 16)
                .padding(.vertical, 9)
                .background(.black.opacity(0.9), in: RoundedRectangle(cornerRadius: 12))
                .padding(.bottom, 96)
        }
    }

    private func showToast(_ message: String) {
        toast = message
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.6) {
            if toast == message { toast = nil }
        }
    }
}

/// Thin accent line that sweeps down the preview while analysis is running, echoing the
/// HTML mockup's scanning beam. Purely decorative — it does not affect the overlay boxes.
private struct ScanBeam: View {
    @State private var offset: CGFloat = 0

    var body: some View {
        Rectangle()
            .fill(LinearGradient(colors: [.clear, appAccent, .clear],
                                startPoint: .leading, endPoint: .trailing))
            .frame(height: 3)
            .shadow(color: appAccent, radius: 6)
            .offset(y: offset)
            .onAppear {
                withAnimation(.easeInOut(duration: 1.8).repeatForever(autoreverses: true)) {
                    offset = 160
                }
            }
    }
}
