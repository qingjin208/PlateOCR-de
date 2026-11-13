import SwiftUI
import UIKit

/// Camera tab, redesigned to match the HTML mockup's live-detection flow:
///   • idle (detection off): a clean live feed with a "开始实时识别" button placed BELOW
///     the preview (not overlaid on the video).
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

    // MARK: - Live telemetry pills (pinned to the two top corners of the feed)

    @ViewBuilder
    private var statusPill: some View {
        if controller.liveEnabled {
            HStack(spacing: 5) {
                // The dot reflects the *session* state, not the per-cycle `isAnalyzing`
                // flag: analysis runs at ~5 fps, so a yellow/green flip there made the
                // pill strobe continuously.
                Circle()
                    .fill(controller.paused ? Color.gray : Color.green)
                    .frame(width: 6, height: 6)
                Text(controller.paused ? "已暂停" : "实时检测中")
                    .fontWeight(.semibold)
                // "峰值" = the model's best raw anchor score before the confidence
                // threshold. When 框 stays 0 it answers "why" at a glance: score ≈ 0.0-0.1
                // → the model never saw a plate-like input; ≥0.25 → it does see the plate
                // and the decode/validation side is dropping it.
                Text(controller.lastTopScore.map {
                        "框 \(controller.liveDetections.count) · 牌 \(controller.liveResults.count) · 峰值 \(String(format: "%.2f", $0))"
                     }
                     ?? "框 \(controller.liveDetections.count) · 牌 \(controller.liveResults.count)")
                    .foregroundStyle(.white.opacity(0.72))
                // Frames the gate accepted. 0 after tapping 开始 = the capture pipeline is
                // not feeding frames; rising while latency stays 0 ms = every analysis
                // throws before finishing (see `frameCount`).
                Text("帧 \(controller.frameCount)")
                    .foregroundStyle(.white.opacity(0.55))
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
            VStack(alignment: .trailing, spacing: 3) {
                HStack(spacing: 6) {
                    Text("\(controller.elapsedMilliseconds) ms")
                    // The SDK has exactly one detector (the LibreYOLO ONNX pose model).
                    // Anything else running means the ONNX path did not come up — say so
                    // in the pill; it is the #1 cause of "no keypoints, no results".
                    // (The stub's name, e.g. LicensePlateDetectorUnavailable(原因), does
                    // not contain "ONNX", so the warning fires exactly when needed.)
                    if !controller.detectorName.hasPrefix("ONNXPoseLicensePlateDetector") {
                        Text("检测器未走 ONNX")
                            .foregroundStyle(.yellow)
                    }
                }
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.white.opacity(0.88))
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(.black.opacity(0.45), in: Capsule())
                // The actual failure cause, surfaced from the recognizer (model missing
                // from the bundle vs. ORT session failure — different fixes). Kept small
                // and capped so a long error message cannot cover the feed.
                if !controller.detectorName.hasPrefix("ONNXPoseLicensePlateDetector") {
                    Text(controller.detectorName)
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.yellow.opacity(0.85))
                        .lineLimit(5)
                        .truncationMode(.tail)
                        .frame(maxWidth: 300, alignment: .trailing)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 8))
                }
            }
            .padding(.trailing, 10)
            .padding(.top, 10)
        }
    }

    // MARK: - Camera transport + history controls.
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
            // `isAnalyzing` is reset by `CameraController.paused.didSet`; writing it from
            // the view (the old behaviour) broke the moment pause was triggered from
            // anywhere else.
            controller.paused.toggle()
        } label: {
            HStack(spacing: 5) {
                Image(systemName: controller.paused ? "play.fill" : "pause.fill")
                    .font(.system(size: 11, weight: .bold))
                Text(controller.paused ? "继续" : "暂停")
                    .font(.system(size: 13, weight: .semibold))
            }
            // Controls sit on the app background (light in light mode), so the previous
            // `.white` text on `Color.white.opacity(0.16)` was invisible. Match the
            // 清空检测历史 button instead, with the accent colour marking the paused state.
            .foregroundStyle(controller.paused ? appAccent : Color.secondary)
            .padding(.horizontal, 18)
            .frame(height: 34)
            .background(Color(.tertiarySystemBackground), in: Capsule())
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
