#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/videoio/cap_ios.h>

#if defined(NO)
#  warning Detected Apple 'NO' macro definition, it can cause build conflicts. Please, include this header before any Apple headers.
#endif

#import "OCRSDK.h"
#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>

#include "yolo_process.h"
#include "pipeline.h"
#include "timer.h"
#include <algorithm>
#include <arm_neon.h>
#include <iostream>
#include <mutex>
#include <paddle_api.h>
#include <paddle_use_kernels.h>
#include <paddle_use_ops.h>
#include <string>
#import <sys/timeb.h>
#include <vector>

using namespace paddle::lite_api;
using namespace cv;

// CocoaPods may copy pod resources flat into the app bundle, or preserve the
// pod's `Resources/models` and `Resources/labels` subfolders. `-pathForResource:
// ofType:` only looks at the bundle root, so a matched set that lives in those
// subfolders silently resolves to nil and the NSAssert below fires with no clue
// about which file is actually missing. Probe every layout we have shipped.
static NSString *OCRSDKResourcePath(NSBundle *bundle, NSString *name, NSString *ext) {
    static NSArray<NSString *> *directories = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        directories = @[@"", @"models", @"labels", @"Resources/models", @"Resources/labels"];
    });
    for (NSString *directory in directories) {
        NSString *path = [bundle pathForResource:name ofType:ext inDirectory:directory];
        if (path.length > 0) return path;
    }
    return nil;
}

Pipeline *pipe_;
Timer tic;
YoloPredictor *yoloPredictor;

@interface OCRSDK () <CvVideoCameraDelegate>
@property(nonatomic, strong) CvVideoCamera *videoCamera;
@property(nonatomic, strong) dispatch_queue_t processingQueue;
@property(nonatomic, assign) BOOL processingFrame;
@property(nonatomic, assign) NSUInteger pendingFrames;
@property(nonatomic) std::string dict_path;
@property(nonatomic) std::string config_path;
@property(nonatomic) cv::Mat cvimg;
@property (nonatomic, strong) UIViewController *parentViewController;
@property (nonatomic, strong) UIViewController *currentViewController;
@property (nonatomic, weak) UIImageView *currentImageView;
@property (nonatomic, weak) UIImageView *previewImageView;
@property(nonatomic) NSMutableArray<NSString *> *myArray;
@property(nonatomic) UIImage *myImage;
@property (nonatomic, strong) NSDate *startTime;
- (void)initializePipeline;
@end

@implementation OCRPlateResult
@end

@implementation OCRSDK
@synthesize currentImageView;

- (instancetype)init {
    self = [super init];
    if (self) {
        self.processingQueue = dispatch_queue_create("com.itlogica.ocr.external-frame", DISPATCH_QUEUE_SERIAL);
        [self initializePipeline];
    }
    return self;
}

- (void)initializePipeline {
    NSBundle *bundle = [NSBundle bundleForClass:[self class]];
    // Exact model set from the iOS release line (origin/iOS). The US pipeline is
    // a matched set: YOLO plate box + 51-class state classifier + PaddleOCR
    // det/cls/rec. The rec model `rec-en.nb` must be paired with `rec-en.txt`;
    // mixing it with the German `ppocr_keys_v1` dict produces garbage text.
    NSString *detPath   = OCRSDKResourcePath(bundle, @"det", @"nb");
    NSString *clsPath   = OCRSDKResourcePath(bundle, @"cls", @"nb");
    NSString *recPath   = OCRSDKResourcePath(bundle, @"rec-en", @"nb");
    NSString *classPath = OCRSDKResourcePath(bundle, @"License_plate_classification_arm", @"nb");
    NSString *labelPath = OCRSDKResourcePath(bundle, @"label", @"txt");
    NSString *dictPath  = OCRSDKResourcePath(bundle, @"rec-en", @"txt");
    NSString *configPath= OCRSDKResourcePath(bundle, @"config", @"txt");
    NSString *yoloPath  = OCRSDKResourcePath(bundle, @"yolo", @"nb");

    if (!(detPath && clsPath && recPath && classPath && labelPath && dictPath
          && configPath && yoloPath)) {
        NSLog(@"[OCRSDK] bundle=%@", bundle.bundlePath);
        NSLog(@"[OCRSDK] .nb in root=%@ models=%@",
              [bundle pathsForResourcesOfType:@"nb" inDirectory:nil],
              [bundle pathsForResourcesOfType:@"nb" inDirectory:@"models"]);
        NSLog(@"[OCRSDK] .txt in root=%@ labels=%@",
              [bundle pathsForResourcesOfType:@"txt" inDirectory:nil],
              [bundle pathsForResourcesOfType:@"txt" inDirectory:@"labels"]);
    }
    NSAssert(detPath && clsPath && recPath && classPath && labelPath && dictPath
             && configPath && yoloPath,
             @"OCRSDK model resources are missing (iOS US model set): det=%@ cls=%@ rec=%@ class=%@ label=%@ dict=%@ config=%@ yolo=%@",
             detPath ?: @"MISSING", clsPath ?: @"MISSING", recPath ?: @"MISSING",
             classPath ?: @"MISSING", labelPath ?: @"MISSING", dictPath ?: @"MISSING",
             configPath ?: @"MISSING", yoloPath ?: @"MISSING");

    yoloPredictor = new YoloPredictor([yoloPath UTF8String], "LITE_POWER_HIGH", 1);

    // iOS release-line constructor: det, cls, rec, state-classifier, power,
    // threads, config, dict, labels.
    pipe_ = new Pipeline(
        [detPath UTF8String],
        [clsPath UTF8String],
        [recPath UTF8String],
        [classPath UTF8String],
        "LITE_POWER_HIGH",
        1,
        [configPath UTF8String],
        [dictPath UTF8String],
        [labelPath UTF8String]
    );
}

- (instancetype)initWithPreview:(UIViewController *)viewController{
    self = [super init];
    if (self) {
        self.parentViewController = viewController;
        self.processingQueue = dispatch_queue_create("com.itlogica.ocr.external-frame", DISPATCH_QUEUE_SERIAL);

        // 创建 rootViewController
        UIViewController *rootVC = [[UIViewController alloc] init];
        self.currentViewController = rootVC;
        self.currentViewController.view.backgroundColor = [UIColor whiteColor];
        
        // 创建容器视图
        UIEdgeInsets safeAreaInsets = self.parentViewController.view.safeAreaInsets;
        UIView *containerView = [[UIView alloc] initWithFrame:CGRectMake(0, safeAreaInsets.top, [UIScreen mainScreen].bounds.size.width, [UIScreen mainScreen].bounds.size.height - safeAreaInsets.top)];
        containerView.backgroundColor = [UIColor whiteColor];
        containerView.clipsToBounds = YES;
        
        // 添加 UIImageView
        UIImageView *imageView = [[UIImageView alloc] initWithFrame:CGRectMake(0, safeAreaInsets.top, containerView.bounds.size.width, containerView.bounds.size.height - 100 - safeAreaInsets.top)];
        self.currentImageView = imageView;
        [containerView addSubview:self.currentImageView];
        
        // 添加关闭按钮
        UIButton *closeButton = [UIButton buttonWithType:UIButtonTypeSystem];
        closeButton.frame = CGRectMake(50, CGRectGetMaxY(imageView.frame) + 20, containerView.frame.size.width - 100, 44);
        [closeButton setTitle:@"Cancel" forState:UIControlStateNormal];
        [closeButton addTarget:self action:@selector(closeCameraUI:) forControlEvents:UIControlEventTouchUpInside];
        [containerView addSubview:closeButton];
        
        [self.currentViewController.view addSubview:containerView];
        self.currentViewController.modalPresentationStyle = UIModalPresentationFullScreen;
        
        self.videoCamera = [[CvVideoCamera alloc] initWithParentView:self.previewImageView];
        self.videoCamera.delegate = self;
        self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
        self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset1280x720;
        self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
        self.videoCamera.rotateVideo = 0;
        self.videoCamera.defaultFPS = 30;
        self.cvimg.create(1280, 720, CV_8UC3);
        
        // 加载模型路径（与外部送帧入口共用同一初始化逻辑）
        [self initializePipeline];
    }
    return self;
}

#pragma mark - External frame API

- (NSError *)invalidPixelBufferError:(NSString *)message {
    return [NSError errorWithDomain:@"OCRSDK"
                               code:1001
                           userInfo:@{NSLocalizedDescriptionKey: message}];
}

- (BOOL)makeRGBMatFromPixelBuffer:(CVPixelBufferRef)pixelBuffer
                               mat:(cv::Mat &)rgbMat
                             error:(NSError **)error {
    OSType pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
    size_t width = CVPixelBufferGetWidth(pixelBuffer);
    size_t height = CVPixelBufferGetHeight(pixelBuffer);

    if (width == 0 || height == 0) {
        if (error) *error = [self invalidPixelBufferError:@"The pixel buffer has an invalid size."];
        return NO;
    }

    if (pixelFormat == kCVPixelFormatType_32BGRA) {
        void *baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer);
        if (!baseAddress) {
            if (error) *error = [self invalidPixelBufferError:@"The BGRA pixel buffer has no base address."];
            return NO;
        }
        cv::Mat bgra((int)height, (int)width, CV_8UC4, baseAddress,
                     CVPixelBufferGetBytesPerRow(pixelBuffer));
        cv::cvtColor(bgra, rgbMat, cv::COLOR_BGRA2RGB);
        return YES;
    }

    if (pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ||
        pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) {
        void *yAddress = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
        void *uvAddress = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1);
        if (!yAddress || !uvAddress) {
            if (error) *error = [self invalidPixelBufferError:@"The NV12 pixel buffer is missing a plane."];
            return NO;
        }
        cv::Mat y((int)height, (int)width, CV_8UC1, yAddress,
                  CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0));
        cv::Mat uv((int)height / 2, (int)width / 2, CV_8UC2, uvAddress,
                   CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1));
        cv::cvtColorTwoPlane(y, uv, rgbMat, cv::COLOR_YUV2RGB_NV12);
        return YES;
    }

    if (error) {
        *error = [self invalidPixelBufferError:
                  [NSString stringWithFormat:@"Unsupported pixel format: %u", (unsigned)pixelFormat]];
    }
    return NO;
}

/// Mirrors the original demo's center strip: ~74% of the frame width, 2:1
/// plate aspect ratio, centered. Camera frames need this because the demo's
/// purple crop box was applied before OCR; stills/videos do not.
static cv::Rect USCenterPlateCrop(const cv::Mat &image) {
    int cropW = (int)(image.cols * 0.74);
    int cropH = cropW / 2;
    if (cropW < 32 || cropH < 16) return cv::Rect(0, 0, image.cols, image.rows);
    cropW = std::min(cropW, image.cols);
    cropH = std::min(cropH, image.rows);
    return cv::Rect((image.cols - cropW) / 2, (image.rows - cropH) / 2, cropW, cropH);
}

- (void)processPixelBuffer:(CVPixelBufferRef)pixelBuffer
                completion:(void (^)(NSArray<OCRPlateResult *> *results,
                                     NSError * _Nullable error))completion {
    [self processPixelBuffer:pixelBuffer useCenterCrop:YES completion:completion];
}

- (void)processPixelBuffer:(CVPixelBufferRef)pixelBuffer
            useCenterCrop:(BOOL)useCenterCrop
                completion:(void (^)(NSArray<OCRPlateResult *> *results,
                                     NSError * _Nullable error))completion {
    if (!pixelBuffer) {
        if (completion) completion(@[], [self invalidPixelBufferError:@"pixelBuffer must not be nil."]);
        return;
    }

    @synchronized (self) {
        // One frame in flight plus at most one queued. Live frames beyond that are
        // rejected WITH a callback so a Swift `await` can never hang forever; a
        // one-shot image/video frame always gets a slot even if a camera pass is
        // still running.
        if (self.pendingFrames >= 2) {
            if (completion) {
                completion(@[], [NSError errorWithDomain:@"OCRSDK"
                                                   code:2
                                               userInfo:@{NSLocalizedDescriptionKey:
                                                              @"US OCRSDK is busy; frame dropped."}]);
            }
            return;
        }
        self.pendingFrames += 1;
    }

    CVPixelBufferRef retainedBuffer = (CVPixelBufferRef)CFRetain(pixelBuffer);
    dispatch_async(self.processingQueue, ^{
        NSMutableArray<OCRPlateResult *> *output = [NSMutableArray array];
        NSError *processingError = nil;

        CVPixelBufferLockBaseAddress(retainedBuffer, kCVPixelBufferLock_ReadOnly);
        cv::Mat rgbImage;
        BOOL converted = [self makeRGBMatFromPixelBuffer:retainedBuffer
                                                     mat:rgbImage
                                                   error:&processingError];
        // Search region: camera frames are limited to the demo's center plate
        // strip; stills/videos are searched over the whole image. In BOTH cases
        // YOLO localizes the plate before OCR — the original demo never OCRed
        // an unlocalized frame, and doing so is what made stills read scenery.
        cv::Rect searchRect = useCenterCrop
            ? USCenterPlateCrop(rgbImage)
            : cv::Rect(0, 0, rgbImage.cols, rgbImage.rows);
        if (searchRect.width <= 0 || searchRect.height <= 0) {
            searchRect = cv::Rect(0, 0, rgbImage.cols, rgbImage.rows);
        }
        cv::Mat searchImage = rgbImage(searchRect).clone();

        // Locate the plate with YOLO and crop to its box. Clip the box back into
        // the search image so a slightly overshooting detection still yields a
        // valid ROI instead of a black-padded crop that confuses OCR.
        cv::Rect plateBox;
        if (yoloPredictor && !searchImage.empty()) {
            try {
                std::vector<YoloResult> yoloResults = yoloPredictor->Predict(searchImage, 640, 640);
                YoloResult best;
                best.score = -1.f;
                for (const auto &object : yoloResults) {
                    // Camera frames kept the strict score gate from the demo;
                    // stills use a lower floor so a small plate is not discarded.
                    const float floor = useCenterCrop ? 0.82f : 0.5f;
                    if (object.score > best.score && object.score >= floor) best = object;
                }
                if (best.score >= 0.f && best.w > 8 && best.h > 8) {
                    int x = std::max(0, (int)best.x);
                    int y = std::max(0, (int)best.y);
                    int w = std::min((int)best.w, searchImage.cols - x);
                    int h = std::min((int)best.h, searchImage.rows - y);
                    if (w > 8 && h > 8) plateBox = cv::Rect(x, y, w, h);
                }
            } catch (...) {
                // A YOLO failure falls back to searching the whole region.
            }
        }

        cv::Mat inferenceImage = plateBox.width > 0
            ? searchImage(plateBox).clone()
            : searchImage;

        if (converted && !inferenceImage.empty() && pipe_) {
            std::vector<PlateRecognition> recognitions = pipe_->ProcessAll(inferenceImage);
            for (const PlateRecognition &recognition : recognitions) {
                OCRPlateResult *result = [OCRPlateResult new];
                result.text = [NSString stringWithUTF8String:recognition.text.c_str()] ?: @"";
                result.confidence = recognition.confidence;

                CGFloat minX = CGFLOAT_MAX;
                CGFloat minY = CGFLOAT_MAX;
                CGFloat maxX = 0;
                CGFloat maxY = 0;
                for (const std::vector<int> &point : recognition.box) {
                    if (point.size() < 2) continue;
                    minX = MIN(minX, point[0]);
                    minY = MIN(minY, point[1]);
                    maxX = MAX(maxX, point[0]);
                    maxY = MAX(maxY, point[1]);
                }
                // ProcessAll returns the plate-box outline in ROI coordinates;
                // add the ROI and search offsets so overlays align to the frame.
                result.boundingBox = (minX == CGFLOAT_MAX)
                    ? CGRectZero
                    : CGRectMake(minX + plateBox.x + searchRect.x,
                                 minY + plateBox.y + searchRect.y,
                                 maxX - minX, maxY - minY);
                [output addObject:result];
            }
        }
        CVPixelBufferUnlockBaseAddress(retainedBuffer, kCVPixelBufferLock_ReadOnly);
        CFRelease(retainedBuffer);

        @synchronized (self) {
            if (self.pendingFrames > 0) self.pendingFrames -= 1;
        }
        if (completion) completion(output, processingError);
    });
}

- (void)processImage:(cv::Mat &)image{
    dispatch_async(dispatch_get_main_queue(), ^{
        cv::Mat croppedImage;
        if (image.channels() == 4) {
            cvtColor(image, croppedImage, COLOR_RGBA2RGB);
        }
        
        // Define the region to crop (e.g., center region)
        CGSize imageSize = CGSizeMake(image.cols, image.rows); // 640x480
        CGSize viewSize = self.currentImageView.bounds.size;    // e.g., 375x667
        CGFloat scaleX = imageSize.width / viewSize.width;
        CGFloat scaleY = imageSize.height / viewSize.height;

        CGFloat layerHeight= (self.currentImageView.bounds.size.width - 100) / 2.0;
        
        CGFloat cropX = 50 * scaleX;
        CGFloat cropY = (self.currentImageView.bounds.size.height - layerHeight) / 2.0 * scaleY;
        CGFloat cropW = (viewSize.width - 100) * scaleX;
        CGFloat cropH = layerHeight * scaleY;
        cv::Rect roi(cropX, cropY, cropW, cropH);
        cv::Mat croppedRegion = croppedImage(roi).clone();  // Cropped image
        
        std::vector<YoloResult> result = yoloPredictor->Predict(croppedRegion, 640, 640);
        bool isWholePlate = false;
        if(result.size() > 0)
        {
            int plateBoundDiff = 5;
            for (const auto &object : result) {
                //std::cout << "x:" << object.x << ",y:" << object.y<< ",w:" << object.w<< ",h:" << object.h << std::endl;
                
                if(object.x >= plateBoundDiff
                   && object.y >= plateBoundDiff
                   && (object.x + object.w + plateBoundDiff) <= cropW
                   && (object.y + object.h + plateBoundDiff) <= cropH)
                {
                    isWholePlate = true;
                    yoloPredictor->VisualizeResults(result, &croppedRegion);
                }
            }
            
            if(isWholePlate)
            {
                std::vector<std::string> res_txt;
                cv::Mat img_vis;
                std::vector<std::string> processedText;
                std::tie(img_vis, processedText) = pipe_->Process(croppedRegion, "output_img_result.jpg", res_txt);
                img_vis.copyTo(croppedImage(roi));
                cvtColor(croppedImage, self->_cvimg, COLOR_RGB2BGR);
                
                self.currentImageView.image = MatToUIImage(self->_cvimg);
                if(processedText.size() >=2)
                {
                    NSDate *currentTime = [NSDate date];
                    NSTimeInterval diffInSeconds = [currentTime timeIntervalSinceDate:self.startTime];
                    if(diffInSeconds < 2)
                    {
                        return;
                    }
                    
                    //std::cout << processedText[0] << "\t" << processedText[1] << std::endl;
                    if(self.myArray.count < 4)
                    {
                        [self.myArray addObject:[NSString stringWithCString:processedText[0].c_str() encoding:NSUTF8StringEncoding]];
                        [self.myArray addObject:[NSString stringWithCString:processedText[1].c_str() encoding:NSUTF8StringEncoding]];
                        cv::Mat tempCroppedRegion;
                        cvtColor(croppedRegion, tempCroppedRegion, COLOR_RGB2BGR);
                        self.myImage = MatToUIImage(tempCroppedRegion);
                    }
                }
            }
        }
        if(!isWholePlate){
            cvtColor(croppedImage, self->_cvimg, COLOR_RGB2BGR);
            self.currentImageView.image = MatToUIImage(self->_cvimg);
        }
    });
}

NSUInteger levenshteinDistance(NSString *str1, NSString *str2) {
    NSUInteger len1 = [str1 length];
    NSUInteger len2 = [str2 length];
    
    // 创建二维数组
    NSUInteger *matrix = (NSUInteger *)malloc(sizeof(NSUInteger) * (len1 + 1) * (len2 + 1));
    
    // 初始化矩阵
    for (NSUInteger i = 0; i <= len1; i++) {
        matrix[i * (len2 + 1)] = i;
    }
    for (NSUInteger j = 0; j <= len2; j++) {
        matrix[j] = j;
    }
    
    // 填充矩阵
    for (NSUInteger i = 1; i <= len1; i++) {
        for (NSUInteger j = 1; j <= len2; j++) {
            NSUInteger cost = [str1 characterAtIndex:i - 1] == [str2 characterAtIndex:j - 1] ? 0 : 1;
            NSUInteger deleteMatrix = matrix[(i - 1) * (len2 + 1) + j] + 1;
            NSUInteger insert = matrix[i * (len2 + 1) + j - 1] + 1;
            NSUInteger replace = matrix[(i - 1) * (len2 + 1) + j - 1] + cost;
            
            matrix[i * (len2 + 1) + j] = MIN(MIN(deleteMatrix, insert), replace);
        }
    }
    
    // 获取 Levenshtein 距离
    NSUInteger distance = matrix[len1 * (len2 + 1) + len2];
    
    // 释放内存
    free(matrix);
    
    return distance;
}

// 计算字符串相似度，返回 0 到 1 之间的值
CGFloat similarity(NSString *str1, NSString *str2) {
    NSUInteger distance = levenshteinDistance(str1, str2);
    NSUInteger maxLength = MAX([str1 length], [str2 length]);
    
    if (maxLength == 0) {
        return 1.0; // 两个空字符串视为完全相同
    }
    
    // 计算相似度，距离越小，相似度越高
    return 1.0 - ((CGFloat)distance / (CGFloat)maxLength);
}

- (void)addHighlightOverlayInView:(UIView *)container withHighlightRect:(CGRect)highlightRect {
    // 移除已有的遮罩层（通过 tag）
    const NSInteger kOverlayTag = 9999;
    UIView *oldOverlay = [container viewWithTag:kOverlayTag];
    if (oldOverlay) {
        [oldOverlay removeFromSuperview];
    }

    // 创建遮罩层
    UIView *overlay = [[UIView alloc] initWithFrame:container.bounds];
    overlay.backgroundColor = [[UIColor blackColor] colorWithAlphaComponent:0.5]; // 半透明黑色
    overlay.tag = kOverlayTag;

    // 创建遮罩路径（中间透明）
    CAShapeLayer *maskLayer = [CAShapeLayer layer];
    UIBezierPath *path = [UIBezierPath bezierPathWithRect:overlay.bounds];
    UIBezierPath *highlightPath = [UIBezierPath bezierPathWithRect:highlightRect];
    [path appendPath:highlightPath];
    maskLayer.path = path.CGPath;
    maskLayer.fillRule = kCAFillRuleEvenOdd;

    overlay.layer.mask = maskLayer;
    [container addSubview:overlay];
}

- (void)startRecognize:(void (^)(NSMutableArray<NSString *> *resultArray, UIImage *processedImage))completion{
    self.myArray = [NSMutableArray array];
    [self.videoCamera start];
    
    CGFloat layerHeight= (self.currentImageView.bounds.size.width - 100) / 2.0;
    UIView *box = [[UIView alloc] initWithFrame:CGRectMake(50, (self.currentImageView.bounds.size.height - layerHeight) / 2.0, self.currentImageView.bounds.size.width - 100, layerHeight)];
    box.layer.borderColor = [UIColor purpleColor].CGColor;
    box.layer.borderWidth = 2.0;
    [self.currentImageView addSubview:box];
    [self addHighlightOverlayInView:self.currentImageView withHighlightRect:box.frame];
    
    [self.parentViewController presentViewController:self.currentViewController animated:YES completion:nil];
    
    self.startTime = [NSDate date];
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        while (self.myArray.count < 5) {
            if(self.myArray.count == 4)
            {
                CGFloat stateSame = similarity(self.myArray[0], self.myArray[2]);
                CGFloat carSame = similarity(self.myArray[1], self.myArray[3]);
                if(stateSame < 0.95 || carSame < 0.95)
                {
                    //NSLog(@"Diff Results: %@", self.myArray);
                    self.myArray = [NSMutableArray array];
                }
                else
                {
                    break;
                }
            }
            [NSThread sleepForTimeInterval:0.1]; // 避免占用过多CPU资源
        }
        
        dispatch_async(dispatch_get_main_queue(), ^{
            [self.videoCamera stop];
            [self.currentViewController dismissViewControllerAnimated:YES completion:^{}];
            if (completion) {
                completion(self.myArray, self.myImage);
            }
        });
    });
}

- (void)closeCameraUI:(UIButton *)sender {
    [self.videoCamera stop];
    [self.currentViewController dismissViewControllerAnimated:YES completion:^{}];
}

@end

