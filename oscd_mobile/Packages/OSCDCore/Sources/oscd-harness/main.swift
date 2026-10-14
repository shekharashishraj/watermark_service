import Foundation
import OSCDCore
import simd

// Usage: oscd-harness anchors <session-dir> <XFeat_480x352.mlpackage> [depth1 depth2]
// Builds O-SCD anchors for a recorded session and prints the anchor yield. The optional
// disambiguation depths test spec deviation D11 (O-SCD uses 1 and 10 world units).
let args = CommandLine.arguments
guard args.count >= 4, args[1] == "anchors" else {
    print("usage: oscd-harness anchors <session-dir> <XFeat_480x352.mlpackage> [depth1 depth2]")
    exit(2)
}
let store = try SessionStore(open: URL(fileURLWithPath: args[2]))
let xfeat = try XFeatModel(compiledModelURL: try XFeatModel.compile(URL(fileURLWithPath: args[3])))
let depths: (Float, Float) = args.count >= 6 ? (Float(args[4])!, Float(args[5])!) : (1, 10)
var views: [(pose: Pose, keypoints: DescribedKeypoints)] = []
for r in store.manifest.keyframes {
    let pose = Pose.fromARKitCameraToWorld(simd_float4x4(columnMajor: r.arkitCameraToWorld))
    views.append((pose, try xfeat.detect(try store.rgb(.keyframes, id: r.id))))
}
let t0 = Date()
let anchors = AnchorBuilder.build(keyframes: views, intrinsics: store.manifest.intrinsics, disambiguationDepths: depths)
print(String(format: "keyframes %d  anchors %d  yield %.3f  depths (%.2f, %.2f)  build %.1f s",
             views.count, anchors.anchorCount, anchors.yield, depths.0, depths.1, Date().timeIntervalSince(t0)))
