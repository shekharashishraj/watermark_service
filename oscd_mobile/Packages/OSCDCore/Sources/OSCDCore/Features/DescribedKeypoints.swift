import Foundation

/// XFeat keypoints with 64-d L2-normalised descriptors (O-SCD `DescribedKeypoints`).
/// An all-zero descriptor row marks an invalid keypoint (O-SCD zeroes features with score ≤ 0).
public struct DescribedKeypoints: Sendable {
    public static let dim = 64
    public var kpts: [SIMD2<Float>]
    public var feats: [Float]          // count × 64, row-major

    public init(kpts: [SIMD2<Float>], feats: [Float]) {
        precondition(feats.count == kpts.count * Self.dim, "feats must hold 64 values per keypoint")
        self.kpts = kpts
        self.feats = feats
    }

    public var count: Int { kpts.count }

    public func isValid(_ i: Int) -> Bool {
        for c in 0..<Self.dim where feats[i * Self.dim + c] != 0 { return true }
        return false
    }

    public static func load(kpts: NPYArray, feats: NPYArray) -> DescribedKeypoints {
        let k = kpts.floats()
        return DescribedKeypoints(kpts: stride(from: 0, to: k.count, by: 2).map { SIMD2(k[$0], k[$0 + 1]) },
                                  feats: feats.floats())
    }
}
