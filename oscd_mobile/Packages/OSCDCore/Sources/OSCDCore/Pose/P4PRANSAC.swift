import Foundation
import simd

/// O-SCD RANSACEstimator(EstimatorType.P4P): every hypothesis is a 20-iteration pose-only MiniBA
/// on 4 random correspondences, started from the same initial pose (the top-retrieved keyframe).
public enum P4PRANSAC {
    public struct Result {
        public var r6: [Double]
        public var t: SIMD3<Double>
        public var inliers: [Bool]
        public var bestIndex: Int
        public var inlierScores: [Double]
        public var models: [(R: simd_double3x3, t: SIMD3<Double>)]
    }

    @inline(__always) static func isInlier(_ R: simd_double3x3, _ t: SIMD3<Double>, _ X: SIMD3<Double>,
                                           _ uv: SIMD2<Double>, _ f: Double, _ c: SIMD2<Double>, _ maxError: Double) -> Bool {
        let q = R * X + t
        let px = SIMD2(f * q.x / q.z, f * q.y / q.z) + c
        return simd_length(uv - px) < maxError
    }

    public static func run<S: IndexSampler>(uv: [SIMD2<Double>], xyz: [SIMD3<Double>], confs: [Double],
                                            f: Double, centre: SIMD2<Double>, initR6: [Double], initT: SIMD3<Double>,
                                            hypotheses: Int = 2000, maxError: Double, sampler: inout S) -> Result? {
        let n = uv.count
        guard n >= 4, hypotheses > 0 else { return nil }
        let idx = (0..<hypotheses).map { _ in sampler.sample(4, from: n) }
        var models = [(R: simd_double3x3, t: SIMD3<Double>)](repeating: (matrix_identity_double3x3, .zero), count: hypotheses)
        var scores = [Double](repeating: 0, count: hypotheses)
        models.withUnsafeMutableBufferPointer { M in
            scores.withUnsafeMutableBufferPointer { S in
                DispatchQueue.concurrentPerform(iterations: hypotheses) { h in
                    let sel = idx[h]
                    let res = MiniBA.solve(r6: initR6, t: initT, f: f, centre: centre,
                                           xyz: sel.map { xyz[$0] }, uv: sel.map { uv[$0] }, config: .p4p)
                    let R = SixD.toMatrix(res.r6)
                    var s = 0.0
                    for k in 0..<n where isInlier(R, res.t, xyz[k], uv[k], f, centre, maxError) { s += confs[k] }
                    M[h] = (R, res.t)
                    S[h] = s
                }
            }
        }
        var best = 0
        for h in 1..<hypotheses where scores[h] > scores[best] { best = h }
        let (R, t) = models[best]
        let inliers = (0..<n).map { isInlier(R, t, xyz[$0], uv[$0], f, centre, maxError) && confs[$0] > 0 }
        return Result(r6: SixD.fromMatrix(R), t: t, inliers: inliers, bestIndex: best, inlierScores: scores, models: models)
    }
}
