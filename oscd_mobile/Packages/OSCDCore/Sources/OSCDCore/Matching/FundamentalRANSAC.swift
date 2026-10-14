import Foundation

/// 8-point fundamental-matrix RANSAC as in O-SCD poses/ransac.cu (OpenCV usac solver, no point
/// normalisation) with Sampson-error inliers.
public enum FundamentalRANSAC {
    public struct Result: Sendable {
        public var F: [Float]
        public var inliers: [Bool]
        public var bestIndex: Int
        public var models: [[Float]]
        public var inlierCounts: [Int]
    }

    /// Row-major F with F[8] = 1, from 8 correspondences selected by `idx`.
    public static func estimate(_ p1: [SIMD2<Float>], _ p2: [SIMD2<Float>], _ idx: [Int]) -> [Float] {
        let m = 8, n = 9
        var a = [Float](repeating: 0, count: m * n)
        for row in 0..<m {
            let u1 = p1[idx[row]], u2 = p2[idx[row]]
            let x1 = u1.x, y1 = u1.y, x2 = u2.x, y2 = u2.y
            a[row * 9 + 0] = x2 * x1; a[row * 9 + 1] = x2 * y1; a[row * 9 + 2] = x2
            a[row * 9 + 3] = y2 * x1; a[row * 9 + 4] = y2 * y1; a[row * 9 + 5] = y2
            a[row * 9 + 6] = x1;      a[row * 9 + 7] = y1;      a[row * 9 + 8] = 1
        }
        // eliminateUpperTriangular
        for r in 0..<m {
            var pivot = a[r * n + r]
            var rowWithPivot = r
            for k in (r + 1)..<m where abs(pivot) < abs(a[k * n + r]) {
                pivot = a[k * n + r]
                rowWithPivot = k
            }
            if abs(pivot) < 1e-8 { continue }
            for c in r..<n { a.swapAt(rowWithPivot * n + c, r * n + c) }
            for j in (r + 1)..<m {
                let fac = a[j * n + r] / pivot
                a[j * n + r] = 0
                for c in (r + 1)..<n { a[j * n + c] -= fac * a[r * n + c] }
            }
        }
        // back-substitution with f[8] = 1
        var f = [Float](repeating: 0, count: 9)
        f[8] = 1
        for i in stride(from: m - 1, through: 0, by: -1) {
            var acc: Float = 0
            for j in (i + 1)..<n { acc -= a[i * n + j] * f[j] }
            f[i] = acc / a[i * n + i]
        }
        return f
    }

    @inline(__always) public static func sampsonError(_ F: [Float], _ u1: SIMD2<Float>, _ u2: SIMD2<Float>) -> Float {
        let x1 = u1.x, y1 = u1.y, x2 = u2.x, y2 = u2.y
        let fx = F[0] * x1 + F[1] * y1 + F[2], fy = F[3] * x1 + F[4] * y1 + F[5]
        let gx = x2 * F[0] + y2 * F[3] + F[6], gy = x2 * F[1] + y2 * F[4] + F[7]
        let e = x2 * fx + y2 * fy + F[6] * x1 + F[7] * y1 + F[8]
        return e * e / (fx * fx + fy * fy + gx * gx + gy * gy)
    }

    /// Nil when fewer than 8 correspondences (O-SCD would raise here; P1 skips the frame).
    public static func run<S: IndexSampler>(_ p1: [SIMD2<Float>], _ p2: [SIMD2<Float>], hypotheses: Int = 2000,
                                            maxError: Float, sampler: inout S) -> Result? {
        let n = p1.count
        guard n >= 8, hypotheses > 0 else { return nil }
        let idx = (0..<hypotheses).map { _ in sampler.sample(8, from: n) }
        let threshold = maxError * maxError
        var models = [[Float]](repeating: [], count: hypotheses)
        var counts = [Int](repeating: 0, count: hypotheses)
        models.withUnsafeMutableBufferPointer { M in
            counts.withUnsafeMutableBufferPointer { C in
                DispatchQueue.concurrentPerform(iterations: hypotheses) { h in
                    let F = estimate(p1, p2, idx[h])
                    var c = 0
                    for k in 0..<n where sampsonError(F, p1[k], p2[k]) < threshold { c += 1 }
                    M[h] = F
                    C[h] = c
                }
            }
        }
        var best = 0
        for h in 1..<hypotheses where counts[h] > counts[best] { best = h }
        let F = models[best]
        return Result(F: F, inliers: (0..<n).map { sampsonError(F, p1[$0], p2[$0]) < threshold },
                      bestIndex: best, models: models, inlierCounts: counts)
    }
}
