Pod::Spec.new do |s|
  # Local pod that compiles the US-plate OCR SDK from source together with the
  # app. Because the sources live in this repo, "Run" recompiles them automatically
  # after any SDK edit — no separate xcframework archive step is required.
  # Derived from VINRecognizeSDK branch feat/external-frame-api (based on dev,
  # the US-plate model), with the external-frame API added.
  s.name         = 'OCRSDK'
  s.version      = '1.0.0'
  s.summary      = 'US license-plate recognition (Objective-C++ / PaddleLite / OpenCV).'
  s.homepage     = 'https://example.local/OCRSDK'
  s.license      = { :type => 'Proprietary' }
  s.author       = { 'ITLogica' => '' }
  s.platform     = :ios, '18.0'

  # Relative to this podspec (OCR/USPlateOCRSDK).
  s.source       = { :path => '.' }

  # 10 implementation files + headers. All compile as Objective-C++ / C++.
  s.source_files = 'Sources/*.{h,hpp,mm,cc,cpp}'
  s.public_header_files = 'Sources/OCRSDK.h'

  # Model bundles. Loaded via [NSBundle bundleForClass:] at runtime — inside a
  # static pod that is the app's main bundle, so these land in the app resources.
  s.resources    = 'Resources/**/*'

  # arm64-only static libraries (simulator slices stripped; arm_neon.h requires ARM).
  s.vendored_libraries = [
    'ThirdParty/lib/libpaddle_api_light_bundled.a',
    'ThirdParty/lib/libopencv2.a'
  ]

  # <opencv2/...> resolves under ThirdParty/opencv_headers (contains opencv2/).
  # <paddle_api.h> and friends resolve under PaddleLite4/include.
  # The SDK's own headers sit beside each .mm/.cc in Sources.
  s.pod_target_xcconfig = {
    'HEADER_SEARCH_PATHS' => '$(PODS_TARGET_SRCROOT)/Sources $(PODS_TARGET_SRCROOT)/ThirdParty/opencv_headers $(PODS_TARGET_SRCROOT)/ThirdParty/PaddleLite4/include',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'gnu++20',
    'CLANG_CXX_LIBRARY' => 'libc++',
    'GCC_PREPROCESSOR_DEFINITIONS' => '$(inherited)'
  }

  s.libraries    = ['c++', 'z']
  s.frameworks   = ['AVFoundation', 'CoreMedia', 'CoreVideo', 'CoreGraphics', 'QuartzCore', 'UIKit']
  s.requires_arc = true
  # arm_neon.h + arm64-only static libs => device (arm64) builds only.
end
