import Foundation
import simd

public struct PoseEstimate: Sendable {
    public var pose: Pose?
    public var accepted = false
    public var retrieved: [Int] = []
    public var nCorrespondences = 0
    public var nPnPInliers = 0
    /// O-SCD n_ba_inliers: mask components, 2 per point.
    public var nBAInlierComponents = 0
    public var failReason: String?
}

/// O-SCD PoseInitializer.init_inference_pose + get_reference_keyframes, with P1 semantics:
/// a rejected frame returns a reason instead of crashing.
public final class PoseEstimator {
    public let anchors: AnchorSet
    public var numReference = 4
    public var pnpHypotheses = 2000
    public var numPtsMiniBA = 2000
    public var minNumInliers = 100          // compared with the component count, as O-SCD does
    private var rng: SeededSampler

    public init(anchors: AnchorSet, seed: UInt64 = 0) {
        self.anchors = anchors
        rng = SeededSampler(seed: seed)
    }

    public func estimate(_ frame: DescribedKeypoints) -> PoseEstimate {
        var est = PoseEstimate()
        let kfs = anchors.keyframes

        // 1. Retrieval: the keyframes with the most mutual-NN matches (no geometric check).
        var counts = [Int](repeating: 0, count: kfs.count)
        counts.withUnsafeMutableBufferPointer { c in
            DispatchQueue.concurrentPerform(iterations: kfs.count) { k in c[k] = Matcher.countMutual(kfs[k].keypoints, frame) }
        }
        est.retrieved = Array(counts.indices.sorted { counts[$0] != counts[$1] ? counts[$0] > counts[$1] : $0 < $1 }.prefix(numReference))
        guard let top = est.retrieved.first else { est.failReason = "no_reference"; return est }

        // 2. 2D–3D correspondences through each retrieved keyframe's anchors.
        var uv: [SIMD2<Double>] = [], xyz: [SIMD3<Double>] = [], conf: [Double] = []
        for k in est.retrieved {
            let kf = kfs[k]
            let m = Matcher.matchFiltered(frame, kf.keypoints, maxError: anchors.maxError, hypotheses: 2000, sampler: &rng)
            for (a, b) in zip(m.idx, m.idxOther) where kf.hasPt3d[b] {
                uv.append(SIMD2<Double>(frame.kpts[a]))
                xyz.append(SIMD3<Double>(kf.pts3d[b]))
                conf.append(Double(kf.conf[b]))
            }
        }
        est.nCorrespondences = uv.count
        if uv.count > 2 * numPtsMiniBA {
            let sel = rng.subset(numPtsMiniBA, of: uv.count)
            uv = sel.map { uv[$0] }; xyz = sel.map { xyz[$0] }; conf = sel.map { conf[$0] }
        }

        // 3. P4P-RANSAC started from the top-retrieved keyframe's pose.
        let K = anchors.intrinsics
        let f = Double(K.f), c = SIMD2<Double>(K.centre)
        guard let ransac = P4PRANSAC.run(uv: uv, xyz: xyz, confs: conf, f: f, centre: c,
                                         initR6: kfs[top].pose.r6, initT: kfs[top].pose.tDouble,
                                         hypotheses: pnpHypotheses, maxError: Double(anchors.maxError), sampler: &rng) else {
            est.failReason = "too_few_correspondences"
            return est
        }
        var uvIn: [SIMD2<Double>] = [], xyzIn: [SIMD3<Double>] = []
        for k in 0..<uv.count where ransac.inliers[k] { uvIn.append(uv[k]); xyzIn.append(xyz[k]) }
        est.nPnPInliers = uvIn.count

        // 4. Mini-BA on exactly numPtsMiniBA points: a random subset, or padded with masked dummies.
        if uvIn.count >= numPtsMiniBA {
            let sel = rng.subset(numPtsMiniBA, of: uvIn.count)
            uvIn = sel.map { uvIn[$0] }; xyzIn = sel.map { xyzIn[$0] }
        } else {
            let pad = numPtsMiniBA - uvIn.count
            uvIn += [SIMD2<Double>](repeating: SIMD2(-1, -1), count: pad)
            xyzIn += [SIMD3<Double>](repeating: .zero, count: pad)
        }
        let ba = MiniBA.solve(r6: ransac.r6, t: ransac.t, f: f, centre: c, xyz: xyzIn, uv: uvIn, config: .poseRefine)
        est.nBAInlierComponents = ba.inlierComponents

        // 5. O-SCD gate: more than min_num_inliers residual components (> 50 points).
        if ba.inlierComponents > minNumInliers {
            est.pose = Pose(r6: ba.r6, t: ba.t)
            est.accepted = true
        } else {
            est.failReason = "too_few_inliers"
        }
        return est
    }
}
