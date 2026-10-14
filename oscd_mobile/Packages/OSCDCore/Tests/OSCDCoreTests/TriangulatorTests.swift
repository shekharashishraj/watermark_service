import XCTest
import simd
@testable import OSCDCore

final class TriangulatorTests: XCTestCase {
    let K = Intrinsics(f: 360, cx: 239.5, cy: 179.5, width: 480, height: 360)

    func testTwoViewTriangulationAndParallaxGate() {
        let a = Pose.identity
        let b = Pose(R: matrix_identity_float3x3, t: SIMD3(-0.5, 0, 0))          // 0.5 m baseline
        let X = SIMD3<Float>(0.1, -0.05, 2)
        let out = Triangulator.run(uv: [K.project(a.apply(X))], uvsOthers: [[K.project(b.apply(X))]],
                                   pose: a, posesOthers: [b], intrinsics: K, maxError: 1.5, minDisambiguation: 45)
        XCTAssertTrue(out.valid[0])
        XCTAssertLessThan(simd_distance(out.pts3d[0], X), 1e-3)
        // A 2 cm baseline gives too little parallax between ray depths 1 and 10: rejected.
        let c = Pose(R: matrix_identity_float3x3, t: SIMD3(-0.02, 0, 0))
        let weak = Triangulator.run(uv: [K.project(X)], uvsOthers: [[K.project(c.apply(X))]],
                                    pose: a, posesOthers: [c], intrinsics: K, maxError: 1.5, minDisambiguation: 45)
        XCTAssertFalse(weak.valid[0])
    }

    /// Three views of a synthetic textured scene: anchors come back at the right 3D positions.
    func testAnchorBuilderOnSyntheticScene() {
        var s = SeededSampler(seed: 11)
        let points = (0..<300).map { _ in SIMD3(Float(s.uniform()) * 2 - 1, Float(s.uniform()) * 1 - 0.5, 3 + Float(s.uniform())) }
        let descriptors: [[Float]] = points.map { _ in
            let v = (0..<64).map { _ in Float(s.uniform() - 0.5) }
            let n = sqrt(v.reduce(0) { $0 + $1 * $1 }); return v.map { $0 / n }
        }
        let poses = [Pose.identity, Pose(R: matrix_identity_float3x3, t: SIMD3(-0.6, 0, 0)),
                     Pose(R: simd_float3x3(simd_quatf(angle: 0.1, axis: SIMD3(0, 1, 0))), t: SIMD3(0.6, 0.1, 0))]
        let views = poses.map { pose -> (pose: Pose, keypoints: DescribedKeypoints) in
            var kpts: [SIMD2<Float>] = [], feats: [Float] = []
            for (X, d) in zip(points, descriptors) {
                let uv = K.project(pose.apply(X))
                guard uv.x > 1, uv.y > 1, uv.x < 478, uv.y < 358 else { continue }
                kpts.append(uv); feats += d
            }
            return (pose, DescribedKeypoints(kpts: kpts, feats: feats))
        }
        let anchors = AnchorBuilder.build(keyframes: views, intrinsics: K, hypotheses: 300)
        XCTAssertGreaterThan(anchors.yield, 0.8)
        let kf = anchors.keyframes[0]
        var worst: Float = 0
        for i in 0..<kf.keypoints.count where kf.hasPt3d[i] {
            let truth = points.min { simd_distance(K.project($0), kf.keypoints.kpts[i]) < simd_distance(K.project($1), kf.keypoints.kpts[i]) }!
            worst = max(worst, simd_distance(kf.pts3d[i], truth))
        }
        XCTAssertLessThan(worst, 0.01)
    }

    func testParityTriangulator() throws {
        let d = try Fixtures.dir("triangulate")
        let uvs = try Fixtures.npy(d, "uvs_others"), flat = Fixtures.vec2s(uvs)
        let nPts = uvs.shape[1]
        let others = (0..<uvs.shape[0]).map { Array(flat[($0 * nPts)..<(($0 + 1) * nPts)]) }
        let rts = try Fixtures.npy(d, "Rts_others").floats()
        let posesOthers = (0..<uvs.shape[0]).map { Fixtures.pose(rowMajor4x4: Array(rts[($0 * 16)..<($0 * 16 + 16)])) }
        let centre = try Fixtures.npy(d, "centre").floats()
        let Kf = Intrinsics(f: try Fixtures.npy(d, "f").floats()[0], cx: centre[0], cy: centre[1], width: 0, height: 0)
        let out = Triangulator.run(uv: Fixtures.vec2s(try Fixtures.npy(d, "uv")), uvsOthers: others,
                                   pose: Fixtures.pose(rowMajor4x4: try Fixtures.npy(d, "Rt").floats()),
                                   posesOthers: posesOthers, intrinsics: Kf,
                                   maxError: try Fixtures.npy(d, "max_error").floats()[0],
                                   minDisambiguation: try Fixtures.npy(d, "min_dis").floats()[0])
        let wantValid = try Fixtures.npy(d, "valid").bools(), wantPts = Fixtures.vec3s(try Fixtures.npy(d, "pts3d"))
        XCTAssertGreaterThanOrEqual(Double(zip(out.valid, wantValid).filter { $0 == $1 }.count) / Double(nPts), 0.995)
        var far = 0, both = 0
        for i in 0..<nPts where out.valid[i] && wantValid[i] {
            both += 1
            if simd_distance(out.pts3d[i], wantPts[i]) > 1e-3 * max(1, simd_length(wantPts[i])) { far += 1 }
        }
        XCTAssertLessThanOrEqual(Double(far) / Double(max(both, 1)), 0.005)
    }

    /// Full anchor building from O-SCD's own keypoints and poses: the same keypoints get anchors.
    func testParityAnchorBuilder() throws {
        let d = try Fixtures.dir("anchors")
        let info = try Fixtures.json(d.appendingPathComponent("info.json"))
        let count = info["count"] as! Int
        let K = Intrinsics(f: Float(info["f"] as! Double), cx: Float(info["cx"] as! Double), cy: Float(info["cy"] as! Double),
                           width: info["width"] as! Int, height: info["height"] as! Int)
        var views: [(pose: Pose, keypoints: DescribedKeypoints)] = []
        for i in 0..<count {
            let p = String(format: "kf_%04d_", i)
            views.append((Fixtures.pose(rowMajor4x4: try Fixtures.npy(d, p + "Rt").floats()),
                          DescribedKeypoints.load(kpts: try Fixtures.npy(d, p + "kpts"), feats: try Fixtures.npy(d, p + "feats"))))
        }
        let anchors = AnchorBuilder.build(keyframes: views, intrinsics: K)
        var agree = 0, total = 0
        for i in 0..<count {
            let want = try Fixtures.npy(d, String(format: "kf_%04d_has_pt3d", i)).bools()
            for k in 0..<want.count where views[i].keypoints.isValid(k) {
                total += 1
                if anchors.keyframes[i].hasPt3d[k] == want[k] { agree += 1 }
            }
        }
        XCTAssertGreaterThanOrEqual(Double(agree) / Double(total), 0.95)
    }
}
