// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "OSCDCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "OSCDCore", targets: ["OSCDCore"])],
    targets: [
        .target(name: "OSCDCore"),
        .testTarget(name: "OSCDCoreTests", dependencies: ["OSCDCore"]),
        .executableTarget(name: "oscd-harness", dependencies: ["OSCDCore"]),
        .executableTarget(name: "oscd-objects", dependencies: ["OSCDCore"]),
    ],
    swiftLanguageModes: [.v5]
)
