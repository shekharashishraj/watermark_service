import Foundation
import OSCDCore
import OSCDFusion
import SplatXTrainC
import simd

// Usage: oscd-fusion-eval <reference.ply> <before-session> <after-session> <cues f32 npy [N,H,W]> <out npy>
// Runs O-SCD's online fusion over the after-scan frames in order (cue + D3 validity = cue > 0, ARKit
// poses), then refine, and writes the fused change map of every frame [N,H,W] plus timings.
let a = CommandLine.arguments
// oscd-fusion-eval depth <reference.ply> <session> <keyframe index> <out.npy>: Method B's depth inputs for one
// frame at its ARKit pose — LiDAR depth, the reference's expected depth, and C_depth — written as [3,H,W].
if a.count == 6, a[1] == "depth" {
    let s = try SessionStore(open: URL(fileURLWithPath: a[3]))
    let r = s.manifest.keyframes[Int(a[4])!], K = s.manifest.intrinsics, m = s.manifest
    let pose = Pose.fromARKitCameraToWorld(simd_float4x4(columnMajor: r.arkitCameraToWorld))
    let dRen = try ReferenceDepth(plyPath: a[2]).expectedDepth(pose: pose, intrinsics: K)
    let (d, c) = try s.depth(.keyframes, id: r.id)
    let dLive = CuesB.upsampleDepth(d, confidence: c, width: m.depthWidth, height: m.depthHeight, toWidth: K.width, toHeight: K.height)
    let both = (0..<dLive.count).filter { dLive[$0].isFinite && dRen[$0].isFinite }
    let cue = CuesB.depthCue(live: dLive, rendered: dRen, valid: [Bool](repeating: true, count: dLive.count)).cue
    let diff = both.map { dLive[$0] - dRen[$0] }.sorted()
    func q(_ p: Double) -> Float { diff[min(diff.count - 1, Int(p * Double(diff.count)))] }
    print(String(format: "pixels with both depths %d of %d · live−render: p10 %.4f p50 %.4f p90 %.4f m · C_depth > 0 on %.1f%% of them, ≥ 0.5 on %.1f%%",
                 both.count, dLive.count, q(0.1), q(0.5), q(0.9), 100 * Double(both.filter { cue[$0] > 0 }.count) / Double(both.count),
                 100 * Double(both.filter { cue[$0] >= 0.5 }.count) / Double(both.count)))
    // Colour cue inside the judged table area (D3), with the reference RGB render at the same pose.
    var err = [CChar](repeating: 0, count: 512)
    let ks = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("../../../../third_party/splatx-metal/Sources/SplatXKernelSrc").standardized.path
    let session: OpaquePointer? = a[2].withCString { ply in ks.withCString { k in
        var args = SxRenderSessionArgs(); args.ply_path = ply; args.kernel_src_dir = k
        return sx_render_session_open(&args, &err, Int32(err.count)) } }
    guard let session else { print("render open failed: \(String(cString: err))"); exit(1) }
    let n = K.width * K.height
    var rgba = [UInt8](repeating: 0, count: n * 4)
    let R = pose.R, t = pose.t
    let vm: [Float] = [R[0][0], R[1][0], R[2][0], t.x, R[0][1], R[1][1], R[2][1], t.y, R[0][2], R[1][2], R[2][2], t.z, 0, 0, 0, 1]
    _ = sx_render_session_frame(session, vm, K.f, K.f, K.cx, K.cy, Int32(K.width), Int32(K.height), &rgba, &err, Int32(err.count))
    sx_render_session_close(session)
    var planar = [Float](repeating: 0, count: n * 3)
    for i in 0..<n { for ch in 0..<3 { planar[ch * n + i] = Float(rgba[i * 4 + ch]) / 255 } }
    let ren = FloatImage(width: K.width, height: K.height, channels: 3, data: planar)
    let live = try s.rgb(.keyframes, id: r.id)
    guard let roi = m.roi else { exit(1) }
    let valid = PreviewValidity.mask(depth: d, confidence: c, depthWidth: m.depthWidth, depthHeight: m.depthHeight,
                                     intrinsics: K, pose: pose, roi: roi, render: ren)
    let colour = CuesB.colorCue(live: live, render: ren, liveDepth: dLive, renderDepth: dRen, valid: valid)
    let dcue = CuesB.depthCue(live: dLive, rendered: dRen, valid: valid).cue
    let judged = valid.filter { $0 }.count
    func pct(_ x: [Float], _ th: Float) -> Double { 100 * Double((0..<n).filter { valid[$0] && x[$0] >= th }.count) / Double(max(judged, 1)) }
    print(String(format: "judged %d px · C_color ≥ 0.5 on %.1f%%, > 0 on %.1f%% · C_depth ≥ 0.5 on %.1f%% · C_depth+C_color ≥ 1 on %.1f%%",
                 judged, pct(colour, 0.5), pct(colour, 1e-6), pct(dcue, 0.5), pct(zip(colour, dcue).map(+), 1)))
    try NPYArray(floats: dLive + dRen + cue + colour, shape: [4, K.height, K.width]).write(to: URL(fileURLWithPath: a[5]))
    exit(0)
}
guard a.count == 6 else { print("usage: oscd-fusion-eval <ply> <before-session> <after-session> <cues.npy> <out.npy>"); exit(2) }
let before = try SessionStore(open: URL(fileURLWithPath: a[2]))
let after = try SessionStore(open: URL(fileURLWithPath: a[3]))
let cues = try NPYArray.load(URL(fileURLWithPath: a[4]))
let n = cues.shape[0], H = cues.shape[1], W = cues.shape[2], px = H * W
let C = cues.floats()
let K = after.manifest.intrinsics
let centres = before.manifest.keyframes.map { Pose.fromARKitCameraToWorld(simd_float4x4(columnMajor: $0.arkitCameraToWorld)).centre }
let fusion = try ChangeFusion(plyPath: a[1], referenceCentres: centres)
let poses = after.manifest.keyframes.prefix(n).map { Pose.fromARKitCameraToWorld(simd_float4x4(columnMajor: $0.arkitCameraToWorld)) }
var online = [Float](), times = [Double]()
for i in 0..<n {
    let cue = Array(C[(i * px)..<((i + 1) * px)])
    let t0 = Date()
    try fusion.addFrame(pose: poses[i], intrinsics: K, cue: cue, valid: cue.map { $0 > 0 })
    let loss = try fusion.stepFrame()
    times.append(Date().timeIntervalSince(t0) * 1000)
    online += try fusion.changeMap(pose: poses[i], intrinsics: K)       // the online mask O-SCD writes for frame i
    if i % 10 == 0 { print(String(format: "frame %2d  loss %.4f  gaussians %d  %.0f ms", i, loss, fusion.count, times.last!)) }
}
let t0 = Date()
try fusion.refine()
let refineSeconds = Date().timeIntervalSince(t0)
var refined = [Float]()
for i in 0..<n { refined += try fusion.changeMap(pose: poses[i], intrinsics: K) }
try NPYArray(floats: online + refined, shape: [2, n, H, W]).write(to: URL(fileURLWithPath: a[5]))
print(String(format: "online: median %.0f ms per frame (16 steps) | refine to %d steps: %.1f s | final gaussians %d",
             times.sorted()[times.count / 2], fusion.totalSteps, refineSeconds, fusion.count))
