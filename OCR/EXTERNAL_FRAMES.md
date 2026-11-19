# External frame recognition

> **NOTE (2026-11):** The Core ML detector and Apple Vision OCR described below have both been retired. This branch now ships exactly one detector — the LibreYOLO YOLO-NAS-pose ONNX export `De-plate-pose.onnx` (four plate-corner keypoints) — and one OCR engine, PaddleOCR PP-OCRv5, both run through ONNX Runtime. The text below is kept as the historical provenance record of the old `.nb` → Core ML conversion; the current buffer-only contract is in the next section.

The default pipeline uses VINRecognizeSDK's original `yolo.nb` graph and weights converted to `license_plate_yolo.mlmodel`, followed by Apple's Vision `RecognizeTextRequest`. Detection runs through Core ML with eligible compute units set to `.all`. This branch has no Paddle runtime dependency. The SDK does not create a camera or request camera permission. Only the demo's `CameraController` owns an `AVCaptureSession`.

## The App → SDK boundary is CVPixelBuffer-only (2026-11)

Every public recognition entry takes a `CVPixelBuffer`. `CGImage`, `Data` and `URL` never cross the boundary as an image payload. Two buffer layouts are supported: **NV12 (default)** and **BGRA**. The camera already delivers NV12 full-range frames natively; still photos and video frames are wrapped by the SDK's factory into NV12 buffers before entering the same funnel.

```swift
let sdk = LicensePlateRecognizer()          // inputPixelFormat defaults to .nv12

// Camera: raw NV12 buffer, no conversion.
let live = try await sdk.recognizeFrame(pixelBuffer, orientation: .right,
                                        mode: .fast, tiled: false, onDetections: nil)

// Still photo: wrap an upright CGImage in the SDK's input buffer (NV12 by default).
let buffer = try sdk.makeInputBuffer(from: uprightCGImage)
let still = try await sdk.recognize(in: buffer, orientation: .up, mode: .fast, tiled: true)

// Opt into BGRA if your producer cannot emit NV12:
sdk.inputPixelFormat = .bgra
let bgra = try sdk.makeInputBuffer(from: uprightCGImage)

// Or use the factory directly for an explicit layout:
let explicit = try PlatePixelBufferFactory.makeBuffer(from: uprightCGImage, format: .nv12)
```

`recognizeFrame` accepts BGRA, NV12 full range, or NV12 video range. Maximum dimensions are 3840 × 2160 (including the portrait equivalent). Pass the actual image orientation; use `.up` if the pixels are already upright. `recognize(in:orientation:mode:tiled:)` is the still-image convenience that delegates to the same funnel. Video takes a `URL` only as a **resource locator** — sampled frames are decoded, transformed, wrapped as CVPixelBuffers (NV12 by default) and passed through `recognizeFrame` internally.

## Demo behavior

The camera preview automatically detects and recognizes plates. It first shows yellow detection boxes, then adds recognized text and confidence. The result list shows the current frame, and console messages include text, confidence, bounds and processing latency. Repeated readings are logged at most once every two seconds per plate. The demo captures 1080p NV12 frames in portrait, processes at most five frames per second, and keeps only one frame in flight. Actual throughput depends on the device and plate count; 4K input support is not a 4K real-time performance claim.

Switching away from the camera or backgrounding the app stops capture and invalidates pending UI updates. The recognition mode picker applies to all three media sources. Camera input is rotated once by AVFoundation; the SDK does not rotate these upright buffers again. See Apple's [capture orientation guidance](https://developer.apple.com/library/archive/qa/qa1744/index.html).

## Model and build

The original NB model came from `feature/german-plate-external-frame-demo`, commit `b75dc3e8955e5e8f7be41ac655156752f5d9b93c`. `main` at `9ec6f24` retains that NB runtime baseline. This branch converts the same graph and all 129 FP32 parameter tensors to Core ML built-in neural-network layers. See [model provenance](Models/nb-coreml-provenance.json) and [conversion details](Models/README.md).

The Core ML model incorporates the source's BGR mean/std normalization. The SDK preserves 640 × 640 letterboxing, 0.45 confidence threshold and 0.45 NMS IoU threshold. It decodes `[1, 5, 8400]` with center/size coordinates in input pixels, restores original image coordinates and clips boxes at image boundaries. Core Graphics supplies letterboxing. No custom Paddle layers or replacement model are used.

Open `LicensePlateOCR.xcodeproj` with Xcode 16 or newer and run on an iPhone with iOS 18 or newer. Xcode compiles the bundled `.mlmodel` to `.mlmodelc`. The Swift Package uses its own resource bundle and no local native-library search paths. Simulator and macOS builds no longer need a Paddle runtime slice; compilation and prediction still need verification on Apple platforms.

## Validation

Windows checks verify original weight bytes and compare the recovered NB graph evaluated with PyTorch against the serialized Core ML graph evaluated independently through ONNX Runtime. Three synthetic inputs pass; detailed evidence is in [nb-coreml-validation.json](Models/nb-coreml-validation.json). This is not actual Core ML or Paddle Lite runtime parity.

On macOS 15 or newer, `swift test` in `OCR/` covers both raw output layouts, float16/float32/float64 storage, clipping/NMS, model compilation/loading, and Core ML CPU predictions against two original-graph fixtures. Existing tests also cover multiple plate crops, NV12 ranges, orientation, oversized-frame rejection and fragmented OCR text. The Swift tests were not run in the Windows workspace.

Before accepting device behavior, build the demo and check BGRA and both NV12 ranges at 4K, several plates in one frame, portrait alignment, permission denial, mode changes, camera tab re-entry and background/foreground transitions. Compare recognized text with known plates and inspect the reported latency. Windows cannot run Xcode, Apple Vision or the iOS runtime; device accuracy and end-to-end latency remain unverified until this check is performed.
