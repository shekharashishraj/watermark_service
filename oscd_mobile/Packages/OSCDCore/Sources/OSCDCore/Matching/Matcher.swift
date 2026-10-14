import Accelerate
import Foundation

/// Corresponding keypoint indices between two keypoint sets.
public struct MatchSet: Sendable, Equatable {
    public var idx: [Int]
    public var idxOther: [Int]
    public init(idx: [Int], idxOther: [Int]) { self.idx = idx; self.idxOther = idxOther }
    public var count: Int { idx.count }
}

/// O-SCD poses/matcher.py.
public enum Matcher {
    public static let minCosSim: Float = 0.82

    /// For every row of `x`: best cosine similarity against `y` and its index (first on ties).
    static func rowArgmax(_ x: DescribedKeypoints, _ y: DescribedKeypoints) -> (best: [Float], arg: [Int]) {
        let nx = x.count, ny = y.count, d = DescribedKeypoints.dim
        var best = [Float](repeating: -.infinity, count: nx)
        var arg = [Int](repeating: 0, count: nx)
        guard nx > 0, ny > 0 else { return (best, arg) }
        let block = 512
        var sim = [Float](repeating: 0, count: block * ny)
        x.feats.withUnsafeBufferPointer { X in
            y.feats.withUnsafeBufferPointer { Y in
                sim.withUnsafeMutableBufferPointer { S in
                    var start = 0
                    while start < nx {
                        let rows = min(block, nx - start)
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(rows), Int32(ny), Int32(d),
                                    1, X.baseAddress! + start * d, Int32(d), Y.baseAddress!, Int32(d),
                                    0, S.baseAddress!, Int32(ny))
                        for r in 0..<rows {
                            var m: Float = 0
                            var mi: vDSP_Length = 0
                            vDSP_maxvi(S.baseAddress! + r * ny, 1, &m, &mi, vDSP_Length(ny))
                            best[start + r] = m
                            arg[start + r] = Int(mi)
                        }
                        start += rows
                    }
                }
            }
        }
        return (best, arg)
    }

    /// O-SCD `match`: match12 for every row of `a` and the mutual-NN + cos > minCosSim mask.
    public static func mutualNearest(_ a: DescribedKeypoints, _ b: DescribedKeypoints,
                                     minCosSim: Float = minCosSim) -> (match12: [Int], mask: [Bool]) {
        let (best12, m12) = rowArgmax(a, b)
        let (_, m21) = rowArgmax(b, a)
        var mask = [Bool](repeating: false, count: a.count)
        for i in 0..<a.count where !m21.isEmpty { mask[i] = m21[m12[i]] == i && best12[i] > minCosSim }
        return (m12, mask)
    }

    public static func match(_ a: DescribedKeypoints, _ b: DescribedKeypoints) -> MatchSet {
        let (m12, mask) = mutualNearest(a, b)
        var s = MatchSet(idx: [], idxOther: [])
        for i in 0..<a.count where mask[i] {
            s.idx.append(i)
            s.idxOther.append(m12[i])
        }
        return s
    }

    /// O-SCD Matcher(..., remove_outliers=True): mutual NN, then keep the 8-point F-RANSAC inliers.
    public static func matchFiltered<S: IndexSampler>(_ a: DescribedKeypoints, _ b: DescribedKeypoints,
                                                      maxError: Float, hypotheses: Int = 2000,
                                                      sampler: inout S) -> MatchSet {
        let m = match(a, b)
        let p1 = m.idx.map { a.kpts[$0] }, p2 = m.idxOther.map { b.kpts[$0] }
        guard let res = FundamentalRANSAC.run(p1, p2, hypotheses: hypotheses, maxError: maxError, sampler: &sampler) else {
            return MatchSet(idx: [], idxOther: [])
        }
        var s = MatchSet(idx: [], idxOther: [])
        for k in 0..<m.count where res.inliers[k] {
            s.idx.append(m.idx[k])
            s.idxOther.append(m.idxOther[k])
        }
        return s
    }

    /// O-SCD `Matcher.evaluate_match`: number of mutual-NN matches (used for retrieval).
    public static func countMutual(_ a: DescribedKeypoints, _ b: DescribedKeypoints) -> Int {
        mutualNearest(a, b).mask.reduce(0) { $0 + ($1 ? 1 : 0) }
    }
}
