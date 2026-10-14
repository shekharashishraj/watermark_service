// swift-tools-version:6.0
import PackageDescription

// O-SCD change fusion (spec §6.4–6.5) on splatx-metal. splatx is a local checkout patched with
// oscd_mobile/patches/splatx-oscd-fusion.patch (see oscd_mobile/README.md).
let package = Package(
    name: "OSCDFusion",
    platforms: [.macOS("15.0"), .iOS("26.0")],
    products: [.library(name: "OSCDFusion", targets: ["OSCDFusion"])],
    dependencies: [
        .package(path: "../../third_party/splatx-metal"),
        .package(path: "../OSCDCore"),
    ],
    targets: [
        .target(name: "OSCDFusion", dependencies: [
            .product(name: "SplatXTrainC", package: "splatx-metal"), "OSCDCore",
        ]),
        .testTarget(name: "OSCDFusionTests", dependencies: ["OSCDFusion"]),
        .executableTarget(name: "oscd-fusion-eval", dependencies: ["OSCDFusion", "OSCDCore"]),
    ],
    swiftLanguageModes: [.v5]
)
