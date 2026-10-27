import Foundation
import OSCDCore
import simd

// Usage:
//   oscd-harness anchors <session-dir> <XFeat_480x352.mlpackage> [depth1 depth2 [neighbours]]
//     Builds O-SCD anchors for a recorded session and prints the anchor yield. The optional
//     disambiguation depths test spec deviation D11 (O-SCD uses 1 and 10 world units); `neighbours`
//     matches each keyframe with that many nearest views instead of every keyframe.
//   oscd-harness pose <reference-session> <query-session> <XFeat_480x352.mlpackage> [depth1 depth2]
//     Builds anchors from the reference session, estimates every query keyframe's pose with
//     O-SCD's XFeat + P4P + mini-BA, and compares it with ARKit (both sessions must share one
//     ARKit world frame, i.e. come from one app run).
//   oscd-harness colmap <session-dir> <out-dir>
//     Writes the session as the COLMAP dataset the phone trains its reference from (ColmapExport),
//     for comparison with tools/session_to_colmap.py.
//   oscd-harness features <session-dir> <XFeat_480x352.mlpackage> <out-dir>
//     Writes every keyframe's XFeat keypoints and descriptors as <id>_kpts.npy (N × 2, working-image
//     pixels) and <id>_feats.npy (N × 64), for tools/refine_poses.py.
let usage = """
usage: oscd-harness anchors <session-dir> <XFeat_480x352.mlpackage> [depth1 depth2 [neighbours]]
       oscd-harness pose <reference-session> <query-session> <XFeat_480x352.mlpackage> [depth1 depth2]
       oscd-harness colmap <session-dir> <out-dir>
       oscd-harness features <session-dir> <XFeat_480x352.mlpackage> <out-dir>
"""
let args = CommandLine.arguments
if args.count == 4, args[1] == "colmap" {
    let r = try ColmapExport.write(session: try SessionStore(open: URL(fileURLWithPath: args[2])), to: URL(fileURLWithPath: args[3]))
    print("\(r.images) images, \(r.points) points → \(args[3])")
    exit(0)
}
if args.count == 5, args[1] == "features" {
    let store = try SessionStore(open: URL(fileURLWithPath: args[2]))
    let xfeat = try XFeatModel(compiledModelURL: try XFeatModel.compile(URL(fileURLWithPath: args[3])))
    let out = URL(fileURLWithPath: args[4])
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    for r in store.manifest.keyframes {
        let k = try xfeat.detect(try store.rgb(.keyframes, id: r.id))
        let name = String(format: "%06d", r.id)
        try NPYArray(floats: k.kpts.flatMap { [$0.x, $0.y] }, shape: [k.count, 2]).write(to: out.appendingPathComponent(name + "_kpts.npy"))
        try NPYArray(floats: k.feats, shape: [k.count, DescribedKeypoints.dim]).write(to: out.appendingPathComponent(name + "_feats.npy"))
    }
    print("\(store.manifest.keyframes.count) keyframes → \(args[4])")
    exit(0)
}
guard args.count >= 4, ["anchors", "pose"].contains(args[1]) else { print(usage); exit(2) }

func keyframes(_ store: SessionStore, _ xfeat: XFeatModel) throws -> [(pose: Pose, keypoints: DescribedKeypoints)] {
    try store.manifest.keyframes.map { r in
        (Pose.fromARKitCameraToWorld(simd_float4x4(columnMajor: r.arkitCameraToWorld)), try xfeat.detect(try store.rgb(.keyframes, id: r.id)))
    }
}

if args[1] == "anchors" {
    let store = try SessionStore(open: URL(fileURLWithPath: args[2]))
    let xfeat = try XFeatModel(compiledModelURL: try XFeatModel.compile(URL(fileURLWithPath: args[3])))
    let depths: (Float, Float) = args.count >= 6 ? (Float(args[4])!, Float(args[5])!) : (1, 10)
    let neighbours = args.count >= 7 ? Int(args[6]) : nil
    let views = try keyframes(store, xfeat)
    let t0 = Date()
    let anchors = AnchorBuilder.build(keyframes: views, intrinsics: store.manifest.intrinsics, disambiguationDepths: depths,
                                      neighbours: neighbours)
    print(String(format: "keyframes %d  anchors %d  yield %.3f  depths (%.2f, %.2f)  neighbours %@  build %.1f s",
                 views.count, anchors.anchorCount, anchors.yield, depths.0, depths.1, neighbours.map(String.init) ?? "all",
                 Date().timeIntervalSince(t0)))
    // Where do the anchors sit? ARKit world is y-up; the ROI's floor is 2 cm under the tapped table point.
    if let roi = store.manifest.roi {
        let tableY = roi.min.y + 0.02
        var onTable = 0, onObjects = 0, offTable = 0
        for kf in anchors.keyframes {
            for i in 0..<kf.keypoints.count where kf.hasPt3d[i] {
                let p = kf.pts3d[i]
                let worldY = p.y       // anchors live in the ARKit world frame (only camera axes are flipped)
                let inside = abs(p.x - (roi.min.x + roi.max.x) / 2) <= (roi.max.x - roi.min.x) / 2
                    && abs(p.z - (roi.min.z + roi.max.z) / 2) <= (roi.max.z - roi.min.z) / 2
                if !inside || worldY < tableY - 0.03 { offTable += 1 }
                else if worldY < tableY + 0.01 { onTable += 1 } else { onObjects += 1 }
            }
        }
        let n = Double(max(anchors.anchorCount, 1))
        print(String(format: "anchors on the table surface %.0f%%, on objects above it %.0f%%, off the table %.0f%%",
                     100 * Double(onTable) / n, 100 * Double(onObjects) / n, 100 * Double(offTable) / n))
    }
} else {
    guard args.count >= 5 else { print(usage); exit(2) }
    let ref = try SessionStore(open: URL(fileURLWithPath: args[2]))
    let query = try SessionStore(open: URL(fileURLWithPath: args[3]))
    let xfeat = try XFeatModel(compiledModelURL: try XFeatModel.compile(URL(fileURLWithPath: args[4])))
    let depths: (Float, Float) = args.count >= 7 ? (Float(args[5])!, Float(args[6])!) : (1, 10)
    let anchors = AnchorBuilder.build(keyframes: try keyframes(ref, xfeat), intrinsics: ref.manifest.intrinsics, disambiguationDepths: depths)
    print(String(format: "reference: %d keyframes, %d anchors, yield %.3f, depths (%.2f, %.2f)",
                 anchors.keyframes.count, anchors.anchorCount, anchors.yield, depths.0, depths.1))
    let estimator = PoseEstimator(anchors: anchors)
    var rotErr: [Float] = [], posErr: [Float] = [], times: [Double] = [], reasons: [String: Int] = [:]
    for (i, r) in query.manifest.keyframes.enumerated() {
        let arkit = Pose.fromARKitCameraToWorld(simd_float4x4(columnMajor: r.arkitCameraToWorld))
        let frame = try xfeat.detect(try query.rgb(.keyframes, id: r.id))
        let t0 = Date()
        let est = estimator.estimate(frame)
        times.append(Date().timeIntervalSince(t0) * 1000)
        if let p = est.pose {
            rotErr.append(p.rotationDegrees(to: arkit)); posErr.append(simd_distance(p.centre, arkit.centre) * 1000)
            print(String(format: "  frame %3d  kpts %4d  corr %4d  pnp %4d  ba-pts %4d  rot %.2f°  pos %.1f mm",
                         i, frame.count, est.nCorrespondences, est.nPnPInliers, est.nBAInlierComponents / 2, rotErr.last!, posErr.last!))
        } else {
            reasons[est.failReason ?? "unknown", default: 0] += 1
            print(String(format: "  frame %3d  kpts %4d  corr %4d  REJECTED (%@)", i, frame.count, est.nCorrespondences, est.failReason ?? "unknown"))
        }
    }
    func median<T: Comparable>(_ v: [T]) -> T? { v.isEmpty ? nil : v.sorted()[v.count / 2] }
    print(String(format: "accepted %d / %d  |  rotation median %.2f°  |  position median %.1f mm  |  estimate median %.0f ms (Mac)",
                 rotErr.count, query.manifest.keyframes.count, median(rotErr) ?? .nan, median(posErr) ?? .nan, median(times) ?? .nan))
    if !reasons.isEmpty { print("rejections:", reasons) }
}
