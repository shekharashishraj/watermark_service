import Foundation
import simd

/// One reference keyframe: XFeat features, stored F-RANSAC-inlier matches, triangulated anchors.
public struct KeyframeAnchors: Sendable {
    public let index: Int
    public let pose: Pose
    public let keypoints: DescribedKeypoints
    public var matches: [Int: MatchSet] = [:]          // other keyframe → (idx here, idx there)
    public var hasPt3d: [Bool]
    public var pts3d: [SIMD3<Float>]
    public var conf: [Float]

    public init(index: Int, pose: Pose, keypoints: DescribedKeypoints) {
        self.index = index
        self.pose = pose
        self.keypoints = keypoints
        hasPt3d = [Bool](repeating: false, count: keypoints.count)
        pts3d = [SIMD3<Float>](repeating: .zero, count: keypoints.count)
        conf = [Float](repeating: 0, count: keypoints.count)
    }
}

public struct AnchorSet: Sendable {
    public var keyframes: [KeyframeAnchors]
    public let intrinsics: Intrinsics
    public let maxError: Float
    public var anchorCount: Int
    /// Fraction of valid keypoints that received a 3D anchor.
    public var yield: Double
}

/// O-SCD reference preparation (oscd.py): exhaustive pairwise matching with F-RANSAC, then
/// Keyframe.update_3dpts for every keyframe.
public enum AnchorBuilder {
    /// O-SCD `max_error = max(match_max_error * width, 1.5)`.
    public static func maxError(width: Int, matchMaxError: Float = 2e-3) -> Float {
        max(matchMaxError * Float(width), 1.5)
    }

    public static func build(keyframes input: [(pose: Pose, keypoints: DescribedKeypoints)], intrinsics: Intrinsics,
                             triangulationCams: Int = 8, hypotheses: Int = 2000, seed: UInt64 = 0,
                             disambiguationDepths: (Float, Float) = (1, 10)) -> AnchorSet {
        var kfs = input.enumerated().map { KeyframeAnchors(index: $0.offset, pose: $0.element.pose, keypoints: $0.element.keypoints) }
        let maxErr = maxError(width: intrinsics.width)
        let snapshot = kfs
        let pairs = (0..<kfs.count).flatMap { i in ((i + 1)..<kfs.count).map { (i, $0) } }
        var results = [MatchSet](repeating: MatchSet(idx: [], idxOther: []), count: pairs.count)
        results.withUnsafeMutableBufferPointer { out in
            DispatchQueue.concurrentPerform(iterations: pairs.count) { p in
                let (i, j) = pairs[p]
                var sampler = SeededSampler(seed: seed &+ UInt64(p) &* 0x9E37_79B9_7F4A_7C15)
                out[p] = Matcher.matchFiltered(snapshot[i].keypoints, snapshot[j].keypoints,
                                               maxError: maxErr, hypotheses: hypotheses, sampler: &sampler)
            }
        }
        for (p, (i, j)) in pairs.enumerated() {
            kfs[i].matches[j] = results[p]
            kfs[j].matches[i] = MatchSet(idx: results[p].idxOther, idxOther: results[p].idx)
        }

        var anchors = 0, validKeypoints = 0
        for k in 0..<kfs.count {
            let kf = kfs[k]
            let chosen = kf.matches
                .sorted { $0.value.count != $1.value.count ? $0.value.count > $1.value.count : $0.key < $1.key }
                .prefix(triangulationCams)
            var uvs: [[SIMD2<Float>]] = [], poses: [Pose] = []
            for (other, m) in chosen {
                var row = [SIMD2<Float>](repeating: SIMD2(-1, -1), count: kf.keypoints.count)
                for (a, b) in zip(m.idx, m.idxOther) { row[a] = kfs[other].keypoints.kpts[b] }
                uvs.append(row)
                poses.append(kfs[other].pose)
            }
            let out = Triangulator.run(uv: kf.keypoints.kpts, uvsOthers: uvs, pose: kf.pose, posesOthers: poses,
                                       intrinsics: intrinsics, maxError: maxErr, minDisambiguation: 30 * maxErr,
                                       disambiguationDepths: disambiguationDepths)
            for i in 0..<kf.keypoints.count where out.valid[i] {
                kfs[k].hasPt3d[i] = true
                kfs[k].pts3d[i] = out.pts3d[i]
                kfs[k].conf[i] = 1
                anchors += 1
            }
            validKeypoints += (0..<kf.keypoints.count).filter { kf.keypoints.isValid($0) }.count
        }
        return AnchorSet(keyframes: kfs, intrinsics: intrinsics, maxError: maxErr, anchorCount: anchors,
                         yield: validKeypoints > 0 ? Double(anchors) / Double(validKeypoints) : 0)
    }
}
