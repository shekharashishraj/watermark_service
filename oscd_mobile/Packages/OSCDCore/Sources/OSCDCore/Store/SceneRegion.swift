import Foundation
import simd

/// The scanned area, worked out from the scan itself: the box holding the 1st–99th percentile of the
/// scan's high-confidence LiDAR points on each axis (98 % of them, so stray far readings drop out),
/// plus a margin. It replaces the tapped table box, so a scan can be a tabletop or a whole room.
public enum SceneRegion {
    public static func bounds(of points: [SIMD3<Double>], lowQuantile: Double = 0.01, highQuantile: Double = 0.99,
                              margin: Double = 0.1) -> AABB {
        var lo = SIMD3<Float>(), hi = SIMD3<Float>()
        for axis in 0..<3 {
            let v = points.map { $0[axis] }.sorted()
            func q(_ p: Double) -> Double {
                let x = p * Double(v.count - 1), i = Int(x.rounded(.down)), f = x - Double(i)
                return i + 1 < v.count ? v[i] * (1 - f) + v[i + 1] * f : v[i]
            }
            lo[axis] = Float(q(lowQuantile) - margin)
            hi[axis] = Float(q(highQuantile) + margin)
        }
        return AABB(min: lo, max: hi)
    }

    /// From every `stride`-th keyframe's LiDAR depth (confidence high). Nil when the scan has no depth.
    public static func estimate(_ session: SessionStore, keyframeStride: Int = 1) throws -> AABB? {
        let m = session.manifest
        var pts: [SIMD3<Double>] = []
        for (i, kf) in m.keyframes.enumerated() where i % max(keyframeStride, 1) == 0 {
            let rgb = try session.array(.keyframes, id: kf.id, name: "rgb").uint8()
            let (depth, conf) = try session.depth(.keyframes, id: kf.id)
            pts += ColmapExport.lidarPoints(depth: depth, confidence: conf, depthWidth: m.depthWidth, depthHeight: m.depthHeight,
                                            intrinsics: m.intrinsics, arkitCameraToWorld: kf.arkitCameraToWorld, rgb: rgb).map(\.xyz)
        }
        return pts.isEmpty ? nil : bounds(of: pts)
    }
    /// The scan's saved box, or one estimated from its LiDAR for a scan saved without one.
    public static func region(of session: SessionStore) throws -> AABB? {
        if let roi = session.manifest.roi { return roi }
        return try estimate(session, keyframeStride: max(session.manifest.keyframes.count / 60, 1))
    }

    /// The point the scan's cameras look at: the least-squares closest point to every keyframe's optical axis,
    /// kept inside `box`. A sweep round a table converges on the table; a room scanned from its middle or its
    /// walls converges on the room. Falls back to the box centre when the axes are nearly parallel.
    public static func focus(arkitCameraToWorld poses: [simd_float4x4], within box: AABB) -> SIMD3<Float> {
        var A = simd_double3x3(), b = SIMD3<Double>()
        for m in poses {
            let c = SIMD3<Double>(SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z))
            let d = simd_normalize(SIMD3<Double>(SIMD3(-m.columns.2.x, -m.columns.2.y, -m.columns.2.z)))   // ARKit looks along −z
            let P = matrix_identity_double3x3 - simd_double3x3(rows: [d * d.x, d * d.y, d * d.z])
            A += P
            b += P * c
        }
        let centre = (box.min + box.max) / 2
        guard poses.count >= 2, abs(A.determinant) > 1e-6 * pow(Double(poses.count), 3) else { return centre }
        let p = SIMD3<Float>(A.inverse * b)
        return simd_clamp(p, box.min, box.max)
    }
}
