/*
 License Plate OCR SDK — CVPixelBuffer input contract

 Abstract:
 The App → SDK boundary is CVPixelBuffer-only. This file defines the two supported
 buffer layouts (NV12 by default, BGRA optional) and the factory that turns an
 already-upright CGImage (still photo / decoded video frame) into one of them.

 The camera needs no conversion: AVCaptureVideoDataOutput already delivers NV12
 full-range buffers, which are handed to `recognizeFrame` unchanged. The factory is
 for media whose source is a CGImage — Photos picks, and frames pulled out of a
 video with AVAssetReader.
 */

import Foundation
import CoreGraphics
import CoreVideo
import CoreImage
import Accelerate

/// Pixel layout of the CVPixelBuffer that crosses the App → SDK boundary.
///
/// `.nv12` is the default: it is the native layout of the iOS camera and of most
/// hardware video decoders, so those paths need zero conversion. `.bgra` is kept
/// as an opt-in alternative for callers whose producer cannot emit NV12.
public enum PlateInputPixelFormat: Sendable {
    case nv12
    case bgra

    /// CoreVideo pixel-format fourcc matching this layout.
    public var cvPixelFormatType: OSType {
        switch self {
        case .nv12: return kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        case .bgra: return kCVPixelFormatType_32BGRA
        }
    }
}

/// Converts decoded stills / video frames into the SDK's CVPixelBuffer input.
public enum PlatePixelBufferFactory {

    /// Build a CVPixelBuffer from an upright CGImage in the requested layout.
    ///
    /// - Parameters:
    ///   - image: An upright image (EXIF orientation already baked in by ImageIO).
    ///   - format: Pixel layout of the produced buffer. Defaults to `.nv12`.
    ///   - maxPixelSize: Long-edge cap; 48 MP photos gain nothing for a 512-input
    ///     detector and only burn memory. The image is never upscaled.
    /// - Returns: An IOSurface-backed CVPixelBuffer readable by `recognizeFrame`.
    public static func makeBuffer(
        from image: CGImage,
        format: PlateInputPixelFormat = .nv12,
        maxPixelSize: Int = 3840
    ) throws -> CVPixelBuffer {
        // Match recognizeFrame's working cap on BOTH edges: a 3024x4032 portrait
        // would otherwise be scaled to 2880x3840 here and then again to 1620x2160
        // inside recognizeFrame — two resamples instead of one.
        let scale = min(1, CGFloat(maxPixelSize) / CGFloat(max(image.width, image.height)),
                           CGFloat(2160) / CGFloat(min(image.width, image.height)))
        // 4:2:0 chroma subsampling requires even dimensions; round down so the two
        // planes stay consistent and the aspect ratio is preserved.
        let width = max(2, Int((CGFloat(image.width) * scale).rounded(.down)) & ~1)
        let height = max(2, Int((CGFloat(image.height) * scale).rounded(.down)) & ~1)

        // Render once into BGRA first. That gives us an exact, known-size byte layout
        // (native endianness B,G,R,A) which is both a valid return value for `.bgra`
        // and an unambiguous source for the RGB→NV12 conversion below.
        let bgra = try renderBGRA(image, width: width, height: height)
        switch format {
        case .bgra:
            return bgra
        case .nv12:
            return try convertToNV12(bgra, width: width, height: height)
        }
    }

    // MARK: - BGRA

    private static func renderBGRA(_ image: CGImage, width: Int, height: Int) throws -> CVPixelBuffer {
        let fitted = CIImage(cgImage: image).transformed(by: CGAffineTransform(
            scaleX: CGFloat(width) / CGFloat(image.width),
            y: CGFloat(height) / CGFloat(image.height)))
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer)
        guard status == kCVReturnSuccess, let buffer else {
            throw LicensePlateError.invalidInput("Could not allocate a BGRA frame buffer.")
        }
        SDKGraphics.ciContext.render(fitted, to: buffer)
        return buffer
    }

    // MARK: - NV12

    /// Convert a BGRA CVPixelBuffer into an NV12 (4:2:0, full-range) CVPixelBuffer.
    ///
    /// Uses Accelerate's `vImageConvert_ARGB8888To420Yp8_CbCr8`, which produces
    /// exactly the two planes NV12 stores: a full-resolution Y plane and a
    /// half-height, interleaved CbCr plane. Full-range ITU-R 709 coefficients match
    /// how Core Image decodes full-range NV12 back into RGB, so the
    /// buffer → `CIImage(cvPixelBuffer:)` round-trip inside `recognizeFrame` is
    /// colour-consistent.
    private static func convertToNV12(_ source: CVPixelBuffer,
                                      width: Int, height: Int) throws -> CVPixelBuffer {
        guard CVPixelBufferLockBaseAddress(source, .readOnly) == kCVReturnSuccess,
              let srcBase = CVPixelBufferGetBaseAddress(source) else {
            throw LicensePlateError.invalidInput("Could not lock the source frame.")
        }
        defer { CVPixelBufferUnlockBaseAddress(source, .readOnly) }
        var src = vImage_Buffer(data: srcBase,
                                height: vImagePixelCount(height),
                                width: vImagePixelCount(width),
                                rowBytes: CVPixelBufferGetBytesPerRow(source))

        var destination: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height,
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &destination)
        guard status == kCVReturnSuccess, let destination else {
            throw LicensePlateError.invalidInput("Could not allocate an NV12 frame buffer.")
        }
        // Tag the buffer with the same 709/full-range colourimetry the vImage
        // conversion below encodes. Without these attachments, CIImage(cvPixelBuffer:)
        // may assume the 601 matrix and shift colours on the way back to RGB.
        CVBufferSetAttachment(destination, kCVImageBufferYCbCrMatrixKey,
                              kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(destination, kCVImageBufferColorPrimariesKey,
                              kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(destination, kCVImageBufferTransferFunctionKey,
                              kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
        guard CVPixelBufferLockBaseAddress(destination, []) == kCVReturnSuccess,
              let yBase = CVPixelBufferGetBaseAddressOfPlane(destination, 0),
              let cbcrBase = CVPixelBufferGetBaseAddressOfPlane(destination, 1) else {
            throw LicensePlateError.invalidInput("NV12 frame is missing a plane.")
        }
        defer { CVPixelBufferUnlockBaseAddress(destination, []) }
        var luma = vImage_Buffer(data: yBase,
                                 height: vImagePixelCount(height),
                                 width: vImagePixelCount(width),
                                 rowBytes: CVPixelBufferGetBytesPerRowOfPlane(destination, 0))
        // Interleaved CbCr: same width as Y, half the height, two bytes per pixel.
        var chroma = vImage_Buffer(data: cbcrBase,
                                   height: vImagePixelCount(height / 2),
                                   width: vImagePixelCount(width),
                                   rowBytes: CVPixelBufferGetBytesPerRowOfPlane(destination, 1))

        var pixelRange = vImage_YpCbCrPixelRange(
            Yp_bias: 0, CbCr_bias: 128,
            YpRangeMax: 255, CbCrRangeMax: 255,
            YpMax: 255, YpMin: 0, CbCrMax: 255, CbCrMin: 0)
        var info = vImage_ARGBToYpCbCr()
        let infoStatus = vImageConvert_ARGBToYpCbCr_GenerateConversion(
            kvImage_ARGBToYpCbCrMatrix_ITU_R_709_2, &pixelRange, &info,
            kvImageARGB8888, kvImage420Yp8_CbCr8, vImage_Flags(kvImageNoFlags))
        guard infoStatus == kvImageNoError else {
            throw LicensePlateError.invalidInput("Could not build the RGB→YUV conversion.")
        }

        // BGRA bytes in memory are B,G,R,A (little endian); vImage's ARGB slot order
        // is A,R,G,B, so byte indices 3,2,1,0 map onto those slots respectively.
        let permute: [UInt8] = [3, 2, 1, 0]
        let error = permute.withUnsafeBufferPointer { ptr in
            vImageConvert_ARGB8888To420Yp8_CbCr8(&src, &luma, &chroma, &info,
                                                 ptr.baseAddress,
                                                 vImage_Flags(kvImageNoFlags))
        }
        guard error == kvImageNoError else {
            throw LicensePlateError.invalidInput("RGB→NV12 conversion failed.")
        }
        return destination
    }
}
