#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <CoreVideo/CoreVideo.h>

NS_ASSUME_NONNULL_BEGIN

/// One plate read from an externally supplied frame.
@interface OCRPlateResult : NSObject
/// Recognized plate text. For the US pipeline this is "State Plate" when the
/// state classifier succeeds, otherwise the raw recognizer output.
@property(nonatomic, copy) NSString *text;
/// Axis-aligned bounding box in the input CVPixelBuffer's pixel coordinate
/// system (top-left origin). Convert to view coordinates in the caller.
@property(nonatomic, assign) CGRect boundingBox;
/// Recognition confidence reported by the text recognizer.
@property(nonatomic, assign) float confidence;
@end

@interface OCRSDK : NSObject

/// Creates a frame-input SDK instance. The SDK does not access the camera or
/// create any UI. Models are loaded once here; reuse a single instance.
- (instancetype)init;

/// Processes one externally supplied camera frame. Supported formats are
/// kCVPixelFormatType_32BGRA and NV12 (video/full range). The SDK does NOT
/// rotate frames: the caller must supply display-oriented pixels.
/// Frames arriving while inference is busy are dropped to bound latency.
/// The completion runs on the SDK's private serial queue, not the main queue.
- (void)processPixelBuffer:(CVPixelBufferRef)pixelBuffer
                completion:(void (^)(NSArray<OCRPlateResult *> *results,
                                     NSError * _Nullable error))completion;

/// Same as above, with explicit control over the legacy demo's center-plate
/// crop. Camera frames should pass YES; still images and extracted video frames
/// (already upright and free of the preview's frame) should pass NO.
- (void)processPixelBuffer:(CVPixelBufferRef)pixelBuffer
            useCenterCrop:(BOOL)useCenterCrop
                completion:(void (^)(NSArray<OCRPlateResult *> *results,
                                     NSError * _Nullable error))completion;

/// Legacy entry: SDK-owned camera UI (kept for the existing demo).
- (instancetype)initWithPreview:(UIViewController *)viewController;
- (void)startRecognize:(void (^)(NSMutableArray<NSString *> *resultArray, UIImage *processedImage))completion;

@end

NS_ASSUME_NONNULL_END
