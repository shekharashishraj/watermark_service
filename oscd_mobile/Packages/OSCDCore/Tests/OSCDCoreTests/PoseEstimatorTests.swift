import XCTest
import simd
@testable import OSCDCore

final class PoseEstimatorTests: XCTestCase {
    func loadFixtureAnchors() throws -> AnchorSet {
        let d = try Fixtures.dir("anchors")
        let info = try Fixtures.json(d.appendingPathComponent("info.json"))
        let K = Intrinsics(f: Float(info["f"] as! Double), cx: Float(info["cx"] as! Double), cy: Float(info["cy"] as! Double),
                           width: info["width"] as! Int, height: info["height"] as! Int)
        var kfs: [KeyframeAnchors] = []
        for i in 0..<(info["count"] as! Int) {
            let p = String(format: "kf_%04d_", i)
            var kf = KeyframeAnchors(index: i, pose: Fixtures.pose(rowMajor4x4: try Fixtures.npy(d, p + "Rt").floats()),
                                     keypoints: DescribedKeypoints.load(kpts: try Fixtures.npy(d, p + "kpts"), feats: try Fixtures.npy(d, p + "feats")))
            kf.hasPt3d = try Fixtures.npy(d, p + "has_pt3d").bools()
            kf.pts3d = Fixtures.vec3s(try Fixtures.npy(d, p + "pts3d"))
            kf.conf = kf.hasPt3d.map { $0 ? 1 : 0 }
            kfs.append(kf)
        }
        return AnchorSet(keyframes: kfs, intrinsics: K, maxError: Float(info["max_error"] as! Double),
                         anchorCount: kfs.reduce(0) { $0 + $1.hasPt3d.filter { $0 }.count }, yield: 0)
    }

    /// Review Focus 1: a frame with almost no features is rejected with a reason.
    func testFrameWithTooFewFeaturesIsRejectedCleanly() {
        let K = Intrinsics(f: 360, cx: 239.5, cy: 179.5, width: 480, height: 360)
        var kf = KeyframeAnchors(index: 0, pose: .identity, keypoints: DescribedKeypoints(kpts: [SIMD2(10, 10)], feats: [1] + [Float](repeating: 0, count: 63)))
        kf.hasPt3d = [true]; kf.pts3d = [SIMD3(0, 0, 1)]; kf.conf = [1]
        let est = PoseEstimator(anchors: AnchorSet(keyframes: [kf], intrinsics: K, maxError: 1.5, anchorCount: 1, yield: 1))
        let r = est.estimate(DescribedKeypoints(kpts: [SIMD2(10, 10)], feats: [1] + [Float](repeating: 0, count: 63)))
        XCTAssertFalse(r.accepted)
        XCTAssertNil(r.pose)
        XCTAssertEqual(r.failReason, "too_few_correspondences")
        // Retrieval and matching ran (and are timed); PnP and bundle adjustment never started.
        XCTAssertGreaterThan(r.timings.retrievalMs, 0)
        XCTAssertGreaterThan(r.timings.matchingMs, 0)
        XCTAssertEqual(r.timings.pnpMs, 0)
        XCTAssertEqual(r.timings.bundleAdjustMs, 0)
    }

    /// End to end against O-SCD's own init_inference_pose (its own randomness, so poses are
    /// compared with tolerances rather than bit-exactly).
    func testParityPoseEstimator() throws {
        let anchors = try loadFixtureAnchors()
        let centres = anchors.keyframes.map { $0.pose.centre }
        let mean = centres.reduce(.zero, +) / Float(centres.count)
        let sceneScale = centres.map { simd_distance($0, mean) }.sorted()[centres.count / 2]
        let estimator = PoseEstimator(anchors: anchors)
        for k in 0..<3 {
            let d = Fixtures.root.appendingPathComponent(String(format: "pose/frame_%03d", k))
            guard FileManager.default.fileExists(atPath: d.path) else { continue }
            let frame = DescribedKeypoints.load(kpts: try Fixtures.npy(d, "kpts"), feats: try Fixtures.npy(d, "feats"))
            let r = estimator.estimate(frame)
            XCTAssertEqual(Set(r.retrieved), Set(try Fixtures.npy(d, "retrieved").ints()), "frame \(k)")
            let wantAccepted = try Fixtures.npy(d, "oscd_accepted").bools()[0]
            XCTAssertEqual(r.accepted, wantAccepted, "frame \(k)")
            guard wantAccepted, let pose = r.pose else { continue }
            let want = Fixtures.pose(rowMajor4x4: try Fixtures.npy(d, "oscd_Rt").floats())
            XCTAssertLessThan(pose.rotationDegrees(to: want), 0.1, "frame \(k)")
            XCTAssertLessThan(simd_distance(pose.centre, want.centre), 0.005 * sceneScale, "frame \(k)")
        }
    }
}
