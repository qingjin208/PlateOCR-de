import SwiftUI
import UIKit

/// Camera tab, redesigned to match the HTML mockup's live-detection flow:
///   • idle (detection off): a clean live feed with a dashed guide frame, and the
///     "开始实时识别" button placed BELOW the preview (not overlaid on the video).
///   • active (detection on): the feed carries only lightweight telemetry pills in its two
///     corners — status + plate count top-left, latency top-right — plus the detection
///     boxes/labels (the actual recognition output) and a decorative scanning beam.
///     Every *control* (pause/stop) stays BELOW the preview so no button covers the
///     picture. Detection runs continuously and renders all plates seen in a frame; it
///     only ends when the user taps 停止.
///
/// The detection overlay itself is drawn by `PreviewView` (inside `CameraPreview`) and
/// keeps its aspect-fill mapping onto the upright portrait frame, so boxes stay glued to
/// the plates — this view only changes the surrounding chrome.
struct CameraView: View {
    @Bindable var controller: CameraController

    var body: some View {
        VStack(spacing: 0) {
            if controller.isAuthorized && !controller.setupFailed {
                CameraPreview(controller: controller)
                    // Scanning beam is detection visualisation, not a control — keep on feed.
                    .overlay(alignment: .top) { scanBeam }
                    // Dashed guide frame is a placement hint, not a button — keep on feed.
                    .overlay(alignment: .center) { guideFrame }
                    // Live telemetry: tiny pills pinned to the two top corners, so the
                    // numbers stay visible without stealing a whole row of height.
                    .overlay(alignment: .topLeading) { statusPill }
                    .overlay(alignment: .topTrailing) { latencyPill }
                    .frame(maxHeight: .infinity)
            } else {
                ContentUnavailableView("相机不可用", systemImage: "camera",
                    description: Text(controller.setupFailed ? "相机初始化失败，请检查设备。" : "请允许相机访问权限。"))
            }

            // Controls (开始 / 暂停 / 停止 / 清空) live in `CameraControls`, rendered by the
            // parent *below* this preview on the app's normal background — so the video area
            // itself shows no black bar beneath it.
            if let message = controller.errorMessage {
                Text(message).font(.footnote).foregroundStyle(.red)
                    .frame(maxWidth: .infinity)
                    .padding(8)
                    .background(Color.black)
            }
        }
        .background(Color.black)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Idle guide frame (dashed hint on the feed, no button)

    @ViewBuilder
    private var guideFrame: some View {
        if !controller.liveEnabled {
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [6]))
                .foregroundStyle(.white.opacity(0.5))
                .frame(width: 220, height: 120)
        }
    }

    // MARK: - Live telemetry pills (pinned to the two top corners of the feed)

    @ViewBuilder
    private var statusPill: some View {
        if controller.liveEnabled {
            HStack(spacing: 5) {
                Circle()
                    .fill(controller.isAnalyzing ? .yellow : .green)
                    .frame(width: 6, height: 6)
                Text(controller.paused ? "已暂停" : (controller.isAnalyzing ? "实时检测中" : "实时检测"))
                    .fontWeight(.semibold)
                Text("框 \(controller.liveDetections.count) · 牌 \(controller.liveResults.count)")
                    .foregroundStyle(.white.opacity(0.72))
            }
            .font(.system(size: 11, design: .monospaced))
            .foregroundStyle(.white)
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(.black.opacity(0.45), in: Capsule())
            .padding(.leading, 10)
            .padding(.top, 10)
        }
    }

    @ViewBuilder
    private var latencyPill: some View {
        if controller.liveEnabled {
            Text("\(controller.elapsedMilliseconds) ms")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.white.opacity(0.88))
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(.black.opacity(0.45), in: Capsule())
                .padding(.trailing, 10)
                .padding(.top, 10)
        }
    }

    // MARK: - Scanning beam (detection visualisation, kept on the feed)

    @ViewBuilder
    private var scanBeam: some View {
        if controller.liveEnabled && !controller.paused && controller.isAnalyzing {
            ScanBeam()
        }
    }
}

/// Thin accent line that sweeps down the preview while analysis is running, echoing the
/// HTML mockup's scanning beam. Purely decorative — it does not affect the overlay boxes.
/// The sweep distance is derived from the actual preview height so the beam stays centred
/// even if the preview container is resized (the previous hard-coded 160 drifted off-centre).
private struct ScanBeam: View {
    @State private var offset: CGFloat = 0
    @State private var maxOffset: CGFloat = 150

    var body: some View {
        GeometryReader { geo in
            Rectangle()
                .fill(LinearGradient(colors: [.clear, appAccent, .clear],
                                    startPoint: .leading, endPoint: .trailing))
                .frame(height: 3)
                .shadow(color: appAccent, radius: 6)
                .offset(y: offset)
                .onAppear {
                    maxOffset = max(40, geo.size.height * 0.42)
                    withAnimation(.easeInOut(duration: 1.8).repeatForever(autoreverses: true)) {
                        offset = maxOffset
                    }
                }
        }
    }
}

/// Camera transport + history controls.
///
/// Rendered by the parent *below* the preview, on the app's normal background (never on the
/// black preview surface), so the video area carries no black bar and the "清空检测历史"
/// button can share the same row as 暂停 / 停止.
struct CameraControls: View {
    let controller: CameraController

    var body: some View {
        HStack(spacing: 10) {
            if controller.liveEnabled {
                pauseButton
                stopButton
                Spacer(minLength: 0)
                if !controller.recognitionHistory.isEmpty { clearButton }
            } else if controller.recognitionHistory.isEmpty {
                startButtonFullWidth
            } else {
                startButton
                Spacer(minLength: 0)
                clearButton
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 10)
        .padding(.bottom, 6)
    }

    private var startButton: some View {
        Button {
            controller.liveEnabled = true
            controller.paused = false
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "camera.fill")
                Text("开始实时识别").font(.headline)
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 22)
            .frame(height: 50)
            .background(appAccent, in: RoundedRectangle(cornerRadius: 14))
        }
    }

    private var startButtonFullWidth: some View {
        Button {
            controller.liveEnabled = true
            controller.paused = false
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "camera.fill")
                Text("开始实时识别").font(.headline)
            }
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity)
            .frame(height: 50)
            .background(appAccent, in: RoundedRectangle(cornerRadius: 14))
        }
    }

    private var pauseButton: some View {
        Button {
            controller.paused.toggle()
            if controller.paused { controller.isAnalyzing = false }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: controller.paused ? "play.fill" : "pause.fill")
                    .font(.system(size: 11, weight: .bold))
                Text(controller.paused ? "继续" : "暂停")
                    .font(.system(size: 13, weight: .semibold))
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 18)
            .frame(height: 34)
            .background(Color.white.opacity(0.16), in: Capsule())
        }
    }

    private var stopButton: some View {
        Button {
            controller.liveEnabled = false
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "stop.fill")
                    .font(.system(size: 10, weight: .bold))
                Text("停止")
                    .font(.system(size: 13, weight: .semibold))
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 18)
            .frame(height: 34)
            .background(appAccent, in: Capsule())
        }
    }

    private var clearButton: some View {
        Button {
            controller.clearRecognitionHistory()
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "trash")
                    .font(.system(size: 13, weight: .semibold))
                Text("清空检测历史")
                    .font(.system(size: 13, weight: .semibold))
            }
            .foregroundStyle(.secondary)
            .padding(.horizontal, 14)
            .frame(height: 34)
            .background(Color(.tertiarySystemBackground), in: Capsule())
        }
    }
}
