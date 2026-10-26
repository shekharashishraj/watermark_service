import Foundation
import simd

/// The before-scan's own LiDAR seen from a live frame's pose: the reference depth the object layer compares the live
/// LiDAR with when it is given (`FrameContext.dBefore`), in place of the splat model's rendered depth (spec §4.4).
/// Both depths come from the same sensor, so the model's 1–2 cm bias over the sheet does not enter the sign.
public enum BeforeDepth {
    /// High-confidence LiDAR points (confidence 2) of every keyframe of a scan, in its ARKit world, kept inside `roi`
    /// (all when nil) and reduced to one point per `voxel` cube: the mean of the points that fell in it, in voxel-key order.
    public static func points(of store: SessionStore, roi: AABB?, voxel: Float = 0.005) throws -> [SIMD3<Float>] {
        let m = store.manifest
        var sums: [SIMD3<Int32>: (sum: SIMD3<Float>, n: Float)] = [:]
        for kf in m.keyframes {
            let (depth, conf) = try store.depth(.keyframes, id: kf.id)
            for p in lidar(depth: depth, confidence: conf, depthWidth: m.depthWidth, depthHeight: m.depthHeight,
                           intrinsics: m.intrinsics, arkitCameraToWorld: kf.arkitCameraToWorld) {
                if let roi, !roi.contains(p) { continue }
                let k = SIMD3<Int32>((p / voxel).rounded(.down))
                let e = sums[k] ?? (.zero, 0)
                sums[k] = (e.sum + p, e.n + 1)
            }
        }
        return sums.sorted { a, b in
            a.key.x != b.key.x ? a.key.x < b.key.x : a.key.y != b.key.y ? a.key.y < b.key.y : a.key.z < b.key.z
        }.map { $0.value.sum / $0.value.n }
    }

    /// Depth (camera z, metres) per pixel of a W×H view: each point in front of the camera covers the `splat`×`splat`
    /// pixels nearest its projection, and every pixel keeps the nearest depth that covers it (a z-buffer). NaN where no
    /// point lands.
    public static func splat(points: [SIMD3<Float>], pose: Pose, intrinsics k: Intrinsics, splat: Int = 2) -> [Float] {
        let w = k.width, h = k.height
        var z = [Float](repeating: .infinity, count: w * h)
        let s = max(splat, 1), half = Float(s - 1) / 2
        for p in points {
            let c = pose.apply(p)
            guard c.z > 0 else { continue }
            let uv = k.project(c)
            guard uv.x.isFinite, uv.y.isFinite else { continue }
            let x0 = Int((uv.x - half).rounded(.toNearestOrAwayFromZero)), y0 = Int((uv.y - half).rounded(.toNearestOrAwayFromZero))
            guard x0 + s > 0, y0 + s > 0, x0 < w, y0 < h else { continue }
            for y in max(y0, 0)..<min(y0 + s, h) {
                for x in max(x0, 0)..<min(x0 + s, w) where c.z < z[y * w + x] { z[y * w + x] = c.z }
            }
        }
        for i in z.indices where z[i] == .infinity { z[i] = .nan }
        return z
    }

    /// One keyframe's high-confidence LiDAR points in world coordinates (as `ColmapExport.lidarPoints`, without colour).
    static func lidar(depth: [Float], confidence: [UInt8], depthWidth dw: Int, depthHeight dh: Int,
                      intrinsics K: Intrinsics, arkitCameraToWorld v: [Float]) -> [SIMD3<Float>] {
        let c2w = ColmapExport.cameraToWorld(v)
        let s = Double(dw) / Double(K.width)
        let fd = Double(K.f) * s, cxd = (Double(K.cx) + 0.5) * s - 0.5, cyd = (Double(K.cy) + 0.5) * s - 0.5
        var out: [SIMD3<Float>] = []
        out.reserveCapacity(dw * dh)
        for v in 0..<dh {
            for u in 0..<dw {
                let i = v * dw + u
                let z = Double(depth[i])
                guard confidence[i] >= 2, z.isFinite, z > 0 else { continue }
                let w = c2w * SIMD4((Double(u) - cxd) / fd * z, (Double(v) - cyd) / fd * z, z, 1)
                out.append(SIMD3(Float(w.x), Float(w.y), Float(w.z)))
            }
        }
        return out
    }
}
