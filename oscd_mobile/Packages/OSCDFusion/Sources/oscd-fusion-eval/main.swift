import Foundation
import OSCDCore
import OSCDFusion
import simd

// Usage: oscd-fusion-eval <reference.ply> <before-session> <after-session> <cues f32 npy [N,H,W]> <out npy>
// Runs O-SCD's online fusion over the after-scan frames in order (cue + D3 validity = cue > 0, ARKit
// poses), then refine, and writes the fused change map of every frame [N,H,W] plus timings.
let a = CommandLine.arguments
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
