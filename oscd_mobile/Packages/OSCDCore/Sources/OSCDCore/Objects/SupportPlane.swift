import Foundation
import simd

/// The height of the surface the scene's objects rest on (the table), in world y (+y up). Changes whose points lie
/// within a centimetre of it are the sheet itself (a cloth's wrinkles, LiDAR and model depth disagreeing on a flat
/// surface), not objects (spec §4.4).
public enum SupportPlane {
    /// The mode of y over the points inside `roi` (all points when nil): a `binSize` histogram, refined by the median
    /// y of the winning bin. Nil when no point is inside.
    public static func height(points: [SIMD3<Float>], roi: AABB?, binSize: Float = 0.01) -> Float? {
        guard binSize > 0 else { return nil }
        var bins: [Int: [Float]] = [:]
        for p in points where p.y.isFinite {
            if let roi, !roi.contains(p) { continue }
            bins[Int((p.y / binSize).rounded(.down)), default: []].append(p.y)
        }
        // Most points wins; on a tie, the lower bin (deterministic).
        guard let winner = bins.max(by: { ($0.value.count, -$0.key) < ($1.value.count, -$1.key) }) else { return nil }
        let ys = winner.value.sorted()
        return ys.count % 2 == 1 ? ys[ys.count / 2] : (ys[ys.count / 2 - 1] + ys[ys.count / 2]) / 2
    }

    /// From a scan's high-confidence LiDAR (as `SceneRegion.estimate` reads it), at most 60 keyframes evenly strided.
    public static func height(of store: SessionStore, roi: AABB?) throws -> Float? {
        let m = store.manifest
        let stride = max((m.keyframes.count + 59) / 60, 1)
        var points: [SIMD3<Float>] = []
        for (i, kf) in m.keyframes.enumerated() where i % stride == 0 {
            let rgb = try store.array(.keyframes, id: kf.id, name: "rgb").uint8()
            let (depth, conf) = try store.depth(.keyframes, id: kf.id)
            points += ColmapExport.lidarPoints(depth: depth, confidence: conf, depthWidth: m.depthWidth, depthHeight: m.depthHeight,
                                               intrinsics: m.intrinsics, arkitCameraToWorld: kf.arkitCameraToWorld, rgb: rgb)
                .map { SIMD3<Float>($0.xyz) }
        }
        return height(points: points, roi: roi)
    }
}
