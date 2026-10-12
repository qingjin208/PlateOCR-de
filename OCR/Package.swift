// swift-tools-version:6.0
import PackageDescription
import Foundation

let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let paddleRoot = packageRoot + "/ThirdParty/PaddleLite"

// Standalone definition of the License Plate SDK.
// The demo app in `LicensePlateOCR/` already compiles these same sources directly,
// so this package is provided for when you want to extract the engine into its own
// importable framework / Swift Package later.
let package = Package(
    name: "LicensePlateSDK",
    platforms: [
        .iOS(.v18),
        .macOS(.v15)
    ],
    products: [
        .library(name: "LicensePlateSDK", targets: ["LicensePlateSDK"])
    ],
    targets: [
        .target(
            name: "LicensePlateSDK",
            dependencies: ["NativePlateDetector"],
            path: "LicensePlateOCR/LicensePlateSDK",
            resources: [.process("Resources")]
        ),
        .testTarget(name: "LicensePlateSDKTests", dependencies: ["LicensePlateSDK"],
                    path: "Tests/LicensePlateSDKTests"),
        .target(
            name: "NativePlateDetector",
            path: "NativePlateDetector",
            publicHeadersPath: "include",
            cxxSettings: [.unsafeFlags(["-I" + paddleRoot + "/include"])],
            linkerSettings: [
                .unsafeFlags(["-L" + paddleRoot + "/lib", "-lpaddle_api_light_bundled"], .when(platforms: [.iOS])),
                .linkedLibrary("c++"),
                .linkedFramework("Accelerate"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("Foundation")
            ]
        )
    ],
    swiftLanguageModes: [.v5],
    cxxLanguageStandard: .cxx17
)
