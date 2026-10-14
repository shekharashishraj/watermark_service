import XCTest
import simd
@testable import OSCDCore

final class MiniBATests: XCTestCase {
    func synthetic(n: Int, outliers: Int, seed: UInt64) -> (truth: Pose, xyz: [SIMD3<Double>], uv: [SIMD2<Double>]) {
        var s = SeededSampler(seed: seed)
        let truth = Pose(R: simd_float3x3(simd_quatf(angle: 0.4, axis: simd_normalize(SIMD3(0.2, 1, 0.1)))), t: SIMD3(0.05, -0.1, 0.8))
        let K = Intrinsics(f: 360, cx: 239.5, cy: 179.5, width: 480, height: 360)
        var xyz: [SIMD3<Double>] = [], uv: [SIMD2<Double>] = []
        for i in 0..<n {
            let X = SIMD3(Float(s.uniform()) - 0.5, Float(s.uniform()) * 0.3, Float(s.uniform()) - 0.5)
            var p = K.project(truth.apply(X))
            if i < n - outliers { p += SIMD2(Float(s.uniform() - 0.5) * 0.6, Float(s.uniform() - 0.5) * 0.6) }
            else { p += SIMD2(40, -30) }
            xyz.append(SIMD3<Double>(X)); uv.append(SIMD2<Double>(p))
        }
        return (truth, xyz, uv)
    }

    func testRecoversPoseAndMasksOutliers() {
        let (truth, xyz, uv) = synthetic(n: 300, outliers: 30, seed: 3)
        let start = Pose(R: truth.R * simd_float3x3(simd_quatf(angle: 0.03, axis: SIMD3(1, 0, 0))), t: truth.t + SIMD3(0.02, 0, -0.02))
        let res = MiniBA.solve(r6: start.r6, t: start.tDouble, f: 360, centre: SIMD2(239.5, 179.5), xyz: xyz, uv: uv, config: .poseRefine)
        let got = Pose(r6: res.r6, t: res.t)
        XCTAssertLessThan(got.rotationDegrees(to: truth), 0.05)
        XCTAssertLessThan(simd_distance(got.t, truth.t), 0.002)
        XCTAssertEqual(res.mask[(2 * 270)...].reduce(0, +), 0)            // the 30 outliers are masked
        XCTAssertGreaterThan(res.inlierComponents, 2 * 260)
    }

    /// Review Focus 2: padded dummies at depth 0 and a fully padded problem never produce NaN.
    func testDegenerateInputsStayFinite() {
        let r6 = Pose.identity.r6
        let allPadded = MiniBA.solve(r6: r6, t: SIMD3(0, 0, 0), f: 360, centre: SIMD2(239.5, 179.5),
                                     xyz: [SIMD3<Double>](repeating: .zero, count: 50),
                                     uv: [SIMD2<Double>](repeating: SIMD2(-1, -1), count: 50), config: .poseRefine)
        XCTAssertTrue(allPadded.r6.allSatisfy(\.isFinite))
        XCTAssertTrue([allPadded.t.x, allPadded.t.y, allPadded.t.z].allSatisfy(\.isFinite))
        XCTAssertEqual(allPadded.inlierComponents, 0)

        var (_, xyz, uv) = synthetic(n: 60, outliers: 0, seed: 9)
        xyz += [SIMD3<Double>](repeating: .zero, count: 40); uv += [SIMD2<Double>](repeating: SIMD2(-1, -1), count: 40)
        let res = MiniBA.solve(r6: r6, t: SIMD3(0, 0, 0), f: 360, centre: SIMD2(239.5, 179.5), xyz: xyz, uv: uv, config: .poseRefine)
        XCTAssertTrue(res.r6.allSatisfy(\.isFinite))
    }

    func testParityMiniBA() throws {
        _ = try Fixtures.dir("pose")
        for k in 0..<3 {
            let d = Fixtures.root.appendingPathComponent(String(format: "pose/frame_%03d", k))
            guard FileManager.default.fileExists(atPath: d.path) else { continue }
            let meta = try Fixtures.json(Fixtures.root.appendingPathComponent("meta.json"))
            let f = meta["f"] as! Double, c = SIMD2(meta["cx"] as! Double, meta["cy"] as! Double)
            let xyz = Fixtures.vec3s(try Fixtures.npy(d, "ba_in_xyz")).map { SIMD3<Double>($0) }
            let uv = Fixtures.vec2s(try Fixtures.npy(d, "ba_in_uv")).map { SIMD2<Double>($0) }
            let r6 = try Fixtures.npy(d, "ba_in_r6").floats().map(Double.init)
            let t = try Fixtures.npy(d, "ba_in_t").floats().map(Double.init)
            let res = MiniBA.solve(r6: r6, t: SIMD3(t[0], t[1], t[2]), f: f, centre: c, xyz: xyz, uv: uv, config: .poseRefine)
            let want = Pose(r6: try Fixtures.npy(d, "ba_out_r6").floats().map(Double.init),
                            t: SIMD3<Double>(Fixtures.vec3s(try Fixtures.npy(d, "ba_out_t"))[0]))
            let got = Pose(r6: res.r6, t: res.t)
            XCTAssertLessThan(got.rotationDegrees(to: want), 0.1, "frame \(k)")
            XCTAssertLessThan(simd_distance(got.t, want.t), 1e-3 * max(1, simd_length(want.t)), "frame \(k)")
            let wantMask = try Fixtures.npy(d, "ba_mask").floats()
            let agree = zip(res.mask, wantMask).filter { Float($0) == $1 }.count
            XCTAssertGreaterThanOrEqual(Double(agree) / Double(wantMask.count), 0.99, "frame \(k)")
        }
    }
}
