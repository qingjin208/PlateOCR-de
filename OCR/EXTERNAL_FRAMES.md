# External frame recognition

The default pipeline uses VINRecognizeSDK's `yolo.nb` plate detector followed by Apple's Vision `RecognizeTextRequest`. Paddle OCR models are not used. The SDK does not create a camera or request camera permission. Only the demo's `CameraController` owns an `AVCaptureSession`.

## Input and results

`LicensePlateRecognizer.recognizeFrame` accepts a caller-owned `CVPixelBuffer` in BGRA, NV12 full range, or NV12 video range. The maximum dimensions are 3840 × 2160, including the portrait equivalent. Pass the actual image orientation; use `.up` if the capture output has already rotated the pixels.

```swift
let sdk = LicensePlateRecognizer()
let result = try await sdk.recognizeFrame(
    pixelBuffer,
    orientation: .up,
    mode: .fast,
    onDetections: { boxes, size in
        // Dispatch UI updates to MainActor. These boxes arrive before OCR finishes.
    }
)
for plate in result.plates {
    print(plate.text, plate.confidence, plate.boundingBox)
}
```

`detections` contains all localized plates; `plates` contains only OCR readings accepted by the German format validator. Both use normalized coordinates with a top-left origin in the oriented image. `imageSize` gives that image's pixel dimensions. One frame may return several plates, including several detections without readable text. A detector or OCR failure throws instead of being silently hidden.

Do not mutate or reuse a submitted buffer's memory until its call completes. A retained buffer reference keeps the allocation alive but cannot prevent the caller from modifying its contents. Await recognition serially in your frame processing loop. Drop incoming frames while recognition is busy; do not enqueue every camera frame.

Still images are normalized for orientation by the demo, rendered into BGRA buffers, and processed by the same frame API. Oversized still images are scaled down while preserving aspect ratio. Imported videos are decoded using `AVAssetReader`, transformed using the track's preferred transform, and sampled through the same API. The default file-video limit is 30 sampled frames; it is not a guarantee that every transient plate appears in the returned results.

## Demo behavior

The camera preview automatically detects and recognizes plates. It first shows yellow detection boxes, then adds recognized text and confidence. The result list shows the current frame, and console messages include text, confidence, bounds and processing latency. Repeated readings are logged at most once every two seconds per plate. The demo captures 1080p NV12 frames in portrait, processes at most five frames per second, and keeps only one frame in flight. Actual throughput depends on the device and plate count; 4K input support is not a 4K real-time performance claim.

Switching away from the camera or backgrounding the app stops capture and invalidates pending UI updates. The recognition mode picker applies to all three media sources. Camera input is rotated once by AVFoundation; the SDK does not rotate these upright buffers again. See Apple's [capture orientation guidance](https://developer.apple.com/library/archive/qa/qa1744/index.html).

## Model and build

The source branch is `feature/german-plate-external-frame-demo`, commit `b75dc3e8955e5e8f7be41ac655156752f5d9b93c`. The model, Paddle Lite static runtime and seven runtime headers are copied byte-for-byte. Their provenance and model/runtime SHA-256 hashes are recorded in [model-provenance.json](ThirdParty/model-provenance.json). Inspection of the copied runtime found LLVM bitcode objects targeting `arm64-apple-ios9.0.0`; compatibility with the selected Xcode linker must be confirmed by a device build.

The native wrapper preserves the source's actual BGR channel order, mean/std normalization, 640 × 640 letterboxing, 0.45 confidence threshold and 0.45 NMS IoU threshold. It supports both source output layouts, restores original image coordinates and clips boxes at image boundaries. Core Graphics supplies preprocessing instead of adding an OpenCV dependency; interpolation may differ slightly from the source implementation.

Open `LicensePlateOCR.xcodeproj` with Xcode 16 or newer and run on a physical arm64 iPhone with iOS 18 or newer. The copied runtime is an iOS device binary. The demo's simulator path reports an explicit unsupported-runtime error; it does not substitute the old car detector. The Swift Package supports device integration through a native target with local include/library paths. Its current iOS linker settings require a physical-device destination; simulator package consumers need a compatible simulator runtime. The macOS package path supports injected-detector tests but does not execute the bundled iOS detector.

## Validation

Portable decoder tests cover multiple plates, overlapping proposals, both tensor layouts, normalized/pixel coordinates, letterbox reversal, edge clipping, padding-only detections, invalid shapes and nonfinite scores:

```sh
c++ -std=c++17 Tests/PlateDecoderTests.cpp -o /tmp/plate-decoder-tests
/tmp/plate-decoder-tests
```

On macOS 15 or newer, `swift test` in `OCR/` exercises injected detectors and OCR, multiple plate crops, NV12 ranges, orientation, oversized-frame rejection, and fragmented OCR text. These tests do not assess the real model's accuracy. Neither the C++ nor Swift tests were executed in the Windows workspace: the available MSVC installation lacks standard-library headers, and Apple frameworks are unavailable. Completed checks cover copied-file integrity, Xcode project syntax/source membership/link settings, and `git diff --check`.

Before accepting device behavior, build the demo and check BGRA and both NV12 ranges at 4K, several plates in one frame, portrait alignment, permission denial, mode changes, camera tab re-entry and background/foreground transitions. Compare recognized text with known plates and inspect the reported latency. Windows cannot run Xcode, Apple Vision or the iOS runtime; device accuracy and end-to-end latency remain unverified until this check is performed.
