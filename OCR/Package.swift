// swift-tools-version:6.0
import PackageDescription

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
            path: "LicensePlateOCR/LicensePlateSDK",
            resources: [.process("Resources")]
        ),
        .testTarget(name: "LicensePlateSDKTests", dependencies: ["LicensePlateSDK"],
                    path: "Tests/LicensePlateSDKTests", resources: [.process("Fixtures")])
    ],
    swiftLanguageModes: [.v5]
)
