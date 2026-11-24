/*
 US Plate Recognition Service

 Bridge between the demo app and the US-plate OCRSDK (Objective-C++). The US SDK's
 public contract is frame based: `init` + `processPixelBuffer:completion:`. It has no
 still-image, video-file, orientation, mode or tiled entry points, so this file adapts
 the app's three existing sources to that single frame funnel:

   - camera: native NV12 frame, rotated here when orientation != .up (the US SDK does
     not apply orientation itself), then sent as BGRA;
   - photo: upright CGImage is already a CVPixelBuffer, sent as BGRA;
   - video: frames are decoded with AVAssetReader, the preferredTransform is baked in,
     and each frame goes through the same funnel.

 The framework is optional at build time. Without a rebuilt OCRSDK.xcframework that
 exposes `processPixelBuffer:completion:`, `canImport(OCRSDK)` is false (or the stub
 bridge is selected) and recognition fails with an explicit error instead of breaking
 the German build.
 */

import Foundation
import AVFoundation
import CoreGraphics
import CoreImage
import CoreVideo
import ImageIO

/// Top-level switch rendered by the chevron menu next to the header title.
enum PlateRegion: String, CaseIterable, Identifiable {
    case germany = "Germany"
    case unitedStates = "USA"

    var id: String { rawValue }

    var menuTitle: String {
        switch self {
        case .germany: return "German plate (DE)"
        case .unitedStates: return "US plate (USA)"
        }
    }

    var shortTitle: String {
        switch self {
        case .germany: return "DE"
        case .unitedStates: return "US"
        }
    }
}

/// One plate read by the US SDK. `boundingBox` is in input-pixel coordinates with a
/// top-left origin, as returned by `OCRPlateResult.boundingBox`.
struct USPlateReading: Sendable {
    let text: String
    let confidence: Float
    let boundingBox: CGRect
}

/// Minimal contract the app needs from the US SDK. Kept separate from the SDK type so
/// the rest of the app compiles with or without the rebuilt framework.
protocol USPlateSDKBridging: Sendable {
    func recognize(_ buffer: CVPixelBuffer) async throws -> [USPlateReading]
}

#if canImport(OCRSDK)
import OCRSDK

/// Requires an OCRSDK.xcframework built from the US-plate branch with the
/// `processPixelBuffer:completion:` API and the `OCRPlateResult` class exposed.
final class USPlateSDKBridge: USPlateSDKBridging, @unchecked Sendable {
    private let sdk = OCRSDK()

    func recognize(_ buffer: CVPixelBuffer) async throws -> [USPlateReading] {
        try await withCheckedThrowingContinuation { continuation in
            // The SDK calls back on its own serial queue; continuation.resume is
            // thread-safe, and the SDK retains the buffer for the duration of the call.
            self.sdk.processPixelBuffer(buffer) { results, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                let readings = (results ?? []).map {
                    USPlateReading(text: $0.text ?? "",
                                   confidence: $0.confidence,
                                   boundingBox: $0.boundingBox)
                }
                continuation.resume(returning: readings)
            }
        }
    }
}
#else
/// Build-time fallback so adding the region switch never breaks the German build.
final class USPlateSDKUnavailableBridge: USPlateSDKBridging {
    func recognize(_ buffer: CVPixelBuffer) async throws -> [USPlateReading] {
        throw LicensePlateError.invalidInput(
            "US OCRSDK is not linked. Add an OCRSDK.xcframework that exposes "
            + "init + processPixelBuffer:completion: to enable US plate recognition.")
    }
}
#endif

enum USPlateSDKBridgeFactory {
    static func makeBridge() -> any USPlateSDKBridging {
        #if canImport(OCRSDK)
        return USPlateSDKBridge()
        #else
        return USPlateSDKUnavailableBridge()
        #endif
    }
}

/// Makes a buffer upright before handing it to the US SDK. The US SDK does not accept
/// an orientation parameter and assumes the incoming pixels are display-oriented.
private enum USFrameUprighter {
    static func upright(_ buffer: CVPixelBuffer,
                        orientation: CGImagePropertyOrientation,
                        context: CIContext) throws -> CVPixelBuffer {
        guard orientation != .up else { return buffer }
        let image = CIImage(cvPixelBuffer: buffer).oriented(orientation)
        guard let cgImage = context.createCGImage(image, from: image.extent) else {
            throw LicensePlateError.invalidInput("Could not rotate the input frame for the US SDK")
        }
        return try PlatePixelBufferFactory.makeBuffer(from: cgImage, format: .bgra)
    }
}

/// US equivalent of `LicensePlateRecognizer.recognizeVideoSamples`: decodes a finite
/// number of evenly spaced frames and bakes `preferredTransform` into the pixels.
enum USVideoFrameSampler {
    struct Frame {
        let buffer: CVPixelBuffer
        let size: CGSize
        let time: Double
    }

    static func sample(url: URL,
                       maxSamples: Int,
                       progress: (@Sendable (Double) -> Void)?) async throws -> [Frame] {
        guard maxSamples > 0 else { return [] }
        return try await Task.detached(priority: .utility) { () throws -> [Frame] in
            let asset = AVURLAsset(url: url)
            guard let track = asset.tracks(withMediaType: .video).first else {
                throw LicensePlateError.invalidInput("The selected video has no video track")
            }
            let transform = track.preferredTransform
            let natural = track.naturalSize
            let width = abs(transform.a) * natural.width + abs(transform.c) * natural.height
            let height = abs(transform.b) * natural.width + abs(transform.d) * natural.height
            guard width > 0, height > 0 else {
                throw LicensePlateError.invalidInput("The selected video has an invalid size")
            }

            let reader = try AVAssetReader(asset: asset)
            let output = AVAssetReaderTrackOutput(
                track: track,
                outputSettings: [String(kCVPixelBufferPixelFormatTypeKey): kCVPixelFormatType_32BGRA])
            output.alwaysCopiesSampleData = false
            guard reader.canAdd(output) else {
                throw LicensePlateError.invalidInput("Could not attach a frame reader to the video")
            }
            reader.add(output)
            guard reader.startReading() else {
                throw LicensePlateError.invalidInput("Could not start reading the video")
            }

            let duration = CMTimeGetSeconds(asset.duration)
            let estimatedFrames = max(1, Int(duration * 30))
            let step = max(1, estimatedFrames / maxSamples)
            let context = CIContext(options: nil)
            var frames: [Frame] = []
            var index = 0

            while frames.count < maxSamples, let sample = output.copyNextSampleBuffer() {
                defer { index += 1 }
                guard index % step == 0,
                      let pixel = CMSampleBufferGetImageBuffer(sample) else { continue }
                let ciImage = CIImage(cvPixelBuffer: pixel).transformed(by: transform)
                guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { continue }
                let buffer = try PlatePixelBufferFactory.makeBuffer(from: cgImage, format: .bgra)
                frames.append(Frame(buffer: buffer,
                                    size: CGSize(width: width, height: height),
                                    time: CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))))
                progress?(min(1.0, Double(frames.count) / Double(maxSamples)))
            }

            if reader.status == .failed, frames.isEmpty, let failure = reader.error {
                throw failure
            }
            return frames
        }.value
    }
}

/// Adapter that exposes the US SDK through the same protocol the camera controller and
/// the image/video passes already use. All US results are converted to `PlateResult`,
/// whose bounding box is normalized so the existing overlays keep working unchanged.
final class USPlateRecognitionService: LicensePlateRecognizing, @unchecked Sendable {
    private let bridge: any USPlateSDKBridging
    private let context = CIContext(options: nil)

    init(bridge: any USPlateSDKBridging = USPlateSDKBridgeFactory.makeBridge()) {
        self.bridge = bridge
    }

    func recognizeFrame(_ pixelBuffer: CVPixelBuffer,
                        orientation: CGImagePropertyOrientation,
                        mode: RecognitionMode,
                        tiled: Bool,
                        onDetections: (@Sendable ([PlateDetection], CGSize) -> Void)?) async throws -> PlateFrameResult {
        let buffer = try USFrameUprighter.upright(pixelBuffer, orientation: orientation, context: context)
        let readings = try await bridge.recognize(buffer)
        let width = CGFloat(CVPixelBufferGetWidth(buffer))
        let height = CGFloat(CVPixelBufferGetHeight(buffer))
        guard width > 0, height > 0 else {
            throw LicensePlateError.invalidInput("The US SDK received an empty frame")
        }
        let size = CGSize(width: width, height: height)
        let plates = readings.map { reading -> PlateResult in
            PlateResult(text: reading.text,
                        confidence: reading.confidence,
                        boundingBox: CGRect(x: reading.boundingBox.minX / width,
                                            y: reading.boundingBox.minY / height,
                                            width: reading.boundingBox.width / width,
                                            height: reading.boundingBox.height / height),
                        imageSize: size,
                        candidates: [])
        }
        // The US SDK does not expose raw detections; report none so the diagnostics
        // panel stays honest about where each result came from.
        onDetections?([], size)
        return PlateFrameResult(detections: [], plates: plates, imageSize: size)
    }

    func recognize(in buffer: CVPixelBuffer,
                   orientation: CGImagePropertyOrientation,
                   mode: RecognitionMode,
                   tiled: Bool) async throws -> PlateFrameResult {
        try await recognizeFrame(buffer, orientation: orientation, mode: mode,
                                 tiled: tiled, onDetections: nil)
    }

    func recognizeVideo(at url: URL,
                        mode: RecognitionMode,
                        maxSamples: Int,
                        progress: (@Sendable (Double) -> Void)?) async throws -> [PlateResult] {
        let frames = try await USVideoFrameSampler.sample(url: url,
                                                           maxSamples: maxSamples,
                                                           progress: progress)
        var plates: [PlateResult] = []
        var seen = Set<String>()
        for frame in frames {
            try Task.checkCancellation()
            let result = try await recognizeFrame(frame.buffer, orientation: .up,
                                                  mode: mode, tiled: false, onDetections: nil)
            for plate in result.plates where seen.insert(plate.text).inserted {
                plates.append(plate)
            }
        }
        return plates
    }
}
