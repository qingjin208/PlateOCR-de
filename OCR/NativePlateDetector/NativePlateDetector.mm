#import "include/NativePlateDetector.h"
#import <TargetConditionals.h>
#include "PlateDecoder.hpp"
#include <mutex>
#if TARGET_OS_IOS && !TARGET_OS_SIMULATOR
#include "paddle_api.h"
#include "paddle_use_ops.h"
#include "paddle_use_kernels.h"
#endif

@implementation NativePlateBox
@end

static NSError *DetectorError(NSString *message) {
    return [NSError errorWithDomain:@"LicensePlateDetector" code:1
                          userInfo:@{NSLocalizedDescriptionKey: message}];
}

@implementation NativePlateDetector {
    std::mutex _mutex;
#if TARGET_OS_IOS && !TARGET_OS_SIMULATOR
    std::shared_ptr<paddle::lite_api::PaddlePredictor> _predictor;
#endif
}
- (nullable instancetype)initWithModelPath:(NSString *)path error:(NSError **)error {
    self = [super init];
    if (!self) return nil;
#if TARGET_OS_IOS && !TARGET_OS_SIMULATOR
    try {
        paddle::lite_api::MobileConfig config;
        config.set_model_from_file(path.UTF8String);
        config.set_threads(2);
        config.set_power_mode(paddle::lite_api::LITE_POWER_HIGH);
        _predictor = paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(config);
        if (!_predictor) throw std::runtime_error("Could not load yolo.nb.");
        return self;
    } catch (const std::exception &e) {
        if (error) *error = DetectorError([NSString stringWithUTF8String:e.what()]);
        return nil;
    }
#else
    if (error) *error = DetectorError(@"The bundled Paddle Lite runtime requires a physical arm64 iOS device.");
    return nil;
#endif
}
- (nullable NSArray<NativePlateBox *> *)detectImage:(CGImageRef)image error:(NSError **)error {
#if TARGET_OS_IOS && !TARGET_OS_SIMULATOR
    std::lock_guard<std::mutex> lock(_mutex);
    try {
        int width = (int)CGImageGetWidth(image), height = (int)CGImageGetHeight(image);
        if (width <= 0 || height <= 0) throw std::runtime_error("Empty image.");
        float scale = std::min(640.f / width, 640.f / height);
        int resizedW = (int)std::round(width * scale), resizedH = (int)std::round(height * scale);
        int padX = (640 - resizedW) / 2, padY = (640 - resizedH) / 2;
        std::vector<uint8_t> rgba(640 * 640 * 4, 114);
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        CGContextRef context = CGBitmapContextCreate(rgba.data(), 640, 640, 8, 640 * 4,
            colorSpace, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
        CGColorSpaceRelease(colorSpace);
        if (!context) throw std::runtime_error("Could not allocate detector input.");
        CGContextSetRGBFillColor(context, 114 / 255., 114 / 255., 114 / 255., 1);
        CGContextFillRect(context, CGRectMake(0, 0, 640, 640));
        CGContextTranslateCTM(context, 0, 640);
        CGContextScaleCTM(context, 1, -1);
        CGContextSetInterpolationQuality(context, kCGInterpolationMedium);
        CGContextDrawImage(context, CGRectMake(padX, padY, resizedW, resizedH), image);
        CGContextRelease(context);
        auto input = _predictor->GetInput(0);
        input->Resize({1, 3, 640, 640});
        float *tensor = input->mutable_data<float>();
        const float mean[] = {.485f, .456f, .406f}, deviation[] = {.229f, .224f, .225f};
        // Preserve the source branch's actual BGR channel order and normalization.
        for (int c = 0; c < 3; ++c)
            for (int i = 0; i < 640 * 640; ++i)
                tensor[c * 640 * 640 + i] = (rgba[i * 4 + (2 - c)] / 255.f - mean[c]) / deviation[c];
        _predictor->Run();
        auto output = _predictor->GetOutput(0);
        auto shape = output->shape();
        if (shape.size() != 3 || shape[0] != 1)
            throw std::runtime_error("Unexpected yolo.nb output shape.");
        auto boxes = plate::decode(output->data<float>(), (int)shape[1], (int)shape[2],
                                   width, height, scale, padX, padY);
        NSMutableArray<NativePlateBox *> *results = [NSMutableArray array];
        for (const auto &box : boxes) {
            NativePlateBox *result = [NativePlateBox new];
            result.rect = CGRectMake(box.x, box.y, box.w, box.h);
            result.confidence = box.score;
            [results addObject:result];
        }
        return results;
    } catch (const std::exception &e) {
        if (error) *error = DetectorError([NSString stringWithUTF8String:e.what()]);
        return nil;
    }
#else
    if (error) *error = DetectorError(@"Plate detection requires a physical arm64 iOS device.");
    return nil;
#endif
}
@end
