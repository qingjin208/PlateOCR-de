#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

NS_ASSUME_NONNULL_BEGIN

@interface NativePlateBox : NSObject
@property(nonatomic) CGRect rect;
@property(nonatomic) float confidence;
@end

/// Serializes access to the bundled Paddle Lite plate detector.
@interface NativePlateDetector : NSObject
- (nullable instancetype)initWithModelPath:(NSString *)path error:(NSError **)error;
- (nullable NSArray<NativePlateBox *> *)detectImage:(CGImageRef)image error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
