/*
 License Plate OCR SDK — Demo App: Camera View

 Abstract:
 SwiftUI surface for the live camera. Shows the preview with real-time detection
 boxes, a "live detection" toggle, a fast/accurate picker, and a shutter button
 that captures a still and shows the recognized plates.
 */

import SwiftUI
import UIKit
import Combine

struct CameraView: View {
    @Bindable var controller: CameraController

    var body: some View {
        VStack(spacing: 12) {
            if controller.isAuthorized {
                ZStack(alignment: .topLeading) {
                    CameraPreview(controller: controller)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)

                    if controller.isAnalyzing {
                        ProgressView("识别中…")
                            .padding(8)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                            .padding()
                    }

                    if controller.setupFailed {
                        Text("相机初始化失败，请检查设备后重试。")
                            .padding(8)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                            .padding()
                    }
                }

                // Controls
                VStack(spacing: 12) {
                    HStack(spacing: 16) {
                        Toggle("实时检测", isOn: $controller.liveEnabled)
                            .fixedSize()
                        Picker("模式", selection: $controller.mode) {
                            ForEach(RecognitionMode.allCases, id: \.self) { m in
                                Text(m == .fast ? "快速" : "精准").tag(m)
                            }
                        }
                        .pickerStyle(.segmented)
                    }
                    .padding(.horizontal)

                    if let message = controller.errorMessage {
                        Text(message)
                            .font(.footnote)
                            .foregroundStyle(.red)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }

                    if controller.isFrozen {
                        Button("重拍") { controller.resumeLive() }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.large)
                    } else {
                        Button(action: { controller.capturePhoto() }) {
                            Image(systemName: "camera.circle.fill")
                                .resizable()
                                .scaledToFit()
                                .frame(width: 60, height: 60)
                                .foregroundStyle(.white)
                                .background(Circle().fill(.blue))
                        }
                    }
                }
                .padding(.bottom, 8)

                // Captured-still results (shown after the shutter freezes the preview)
                if controller.isFrozen {
                    if controller.capturedResults.isEmpty {
                        Text("未识别到德国车牌")
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal)
                    } else {
                        List(controller.capturedResults) { r in
                            LabeledContent(r.text, value: String(format: "%.0f%%", r.confidence * 100))
                        }
                        .frame(height: 170)
                    }
                }
            } else {
                ContentUnavailableView {
                    Label("需要相机权限", systemImage: "camera")
                } description: {
                    Text(controller.setupFailed
                          ? "相机初始化失败，请检查设备后重试。"
                          : "请在系统设置中允许本 App 使用相机。")
                }
            }
        }
        .task {
            _ = await controller.setup()
        }
        // Re-check permission when returning from the system Settings screen
        // (granted there, then back to the app) so the view updates immediately.
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)) { _ in
            Task { await controller.setup() }
        }
    }
}
