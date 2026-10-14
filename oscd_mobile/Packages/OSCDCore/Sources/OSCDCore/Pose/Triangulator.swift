import simd

/// O-SCD poses/triangulator.py for one keyframe. For each keypoint, intersect its ray with the
/// matched ray in each other camera, keep the candidate with the most parallax ("disambiguation":
/// pixel distance in the other view between the ray's points at two depths), and accept it if that
/// parallax exceeds `minDisambiguation`.
public enum Triangulator {
    public struct Output: Sendable {
        public var pts3d: [SIMD3<Float>]          // world coordinates
        public var depth: [Float]                 // camera z in this keyframe
        public var bestDisambiguation: [Float]
        public var valid: [Bool]
    }

    /// - uvsOthers[c][i]: keypoint i's match in other camera c, or (-1, -1).
    /// - disambiguationDepths: O-SCD uses 1 and 10 world units. Our world is metric; Task 17 measures
    ///   whether that leaves enough anchors on a tabletop (spec deviation D11 if it doesn't).
    public static func run(uv: [SIMD2<Float>], uvsOthers: [[SIMD2<Float>]], pose: Pose, posesOthers: [Pose],
                           intrinsics K: Intrinsics, maxError: Float, minDisambiguation: Float,
                           disambiguationDepths: (Float, Float) = (1, 10)) -> Output {
        let n = uv.count
        var pts = [SIMD3<Float>](repeating: .zero, count: n)
        var best = [Float](repeating: 0, count: n)
        for (c, other) in posesOthers.enumerated() {
            let rel = pose.compose(other.inverse)                 // other camera → this camera
            let R = rel.R, p2 = rel.t, Rt = R.transpose
            let others = uvsOthers[c]
            for i in 0..<n {
                let o = others[i]
                guard min(o.x, o.y) > 0 else { continue }
                let d1 = simd_normalize(K.unproject(uv[i], depth: 1))
                let d2 = simd_normalize(R * K.unproject(o, depth: 1))
                let nrm = simd_cross(d1, d2), n2 = simd_cross(d2, nrm)
                let xyz = d1 * (simd_dot(n2, p2) / simd_dot(n2, d1))
                let q1 = Rt * (d1 * disambiguationDepths.0 - p2)
                let q2 = Rt * (d1 * disambiguationDepths.1 - p2)
                let dis = simd_length(K.project(q1) - K.project(q2))
                let err = simd_length(K.project(Rt * (xyz - p2)) - o)
                if xyz.z > 1e-6 && dis > best[i] && err < maxError {
                    pts[i] = xyz
                    best[i] = dis
                }
            }
        }
        let Rkf = pose.R.transpose
        return Output(pts3d: pts.map { Rkf * ($0 - pose.t) }, depth: pts.map(\.z),
                      bestDisambiguation: best, valid: best.map { $0 > minDisambiguation })
    }
}
