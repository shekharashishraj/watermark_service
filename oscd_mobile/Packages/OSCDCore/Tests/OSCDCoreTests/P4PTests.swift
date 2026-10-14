import XCTest
import simd
@testable import OSCDCore

final class P4PTests: XCTestCase {
    func testRecoversPoseWithHalfOutliers() {
        var s = SeededSampler(seed: 5)
        let truth = Pose(R: simd_float3x3(simd_quatf(angle: 0.5, axis: simd_normalize(SIMD3(0, 1, 0.3)))), t: SIMD3(0.1, 0.05, 0.9))
        let K = Intrinsics(f: 360, cx: 239.5, cy: 179.5, width: 480, height: 360)
        var uv: [SIMD2<Double>] = [], xyz: [SIMD3<Double>] = []
        for i in 0..<400 {
            let X = SIMD3(Float(s.uniform()) - 0.5, Float(s.uniform()) * 0.3, Float(s.uniform()) - 0.5)
            var p = K.project(truth.apply(X))
            if i % 2 == 1 { p = SIMD2(Float(s.uniform()) * 480, Float(s.uniform()) * 360) }
            uv.append(SIMD2<Double>(p)); xyz.append(SIMD3<Double>(X))
        }
        // Start 6° / 5 cm away, as from the top-retrieved reference keyframe.
        let start = Pose(R: truth.R * simd_float3x3(simd_quatf(angle: 0.1, axis: SIMD3(0, 0, 1))), t: truth.t + SIMD3(0.05, 0, 0))
        let res = P4PRANSAC.run(uv: uv, xyz: xyz, confs: [Double](repeating: 1, count: 400), f: 360,
                                centre: SIMD2(239.5, 179.5), initR6: start.r6, initT: start.tDouble,
                                hypotheses: 500, maxError: 1.5, sampler: &s)!
        XCTAssertLessThan(Pose(r6: res.r6, t: res.t).rotationDegrees(to: truth), 0.5)
        XCTAssertGreaterThanOrEqual((0..<400).filter { $0 % 2 == 0 && res.inliers[$0] }.count, 190)
    }

    func testFewerThanFourPointsIsNil() {
        var s = SeededSampler(seed: 0)
        XCTAssertNil(P4PRANSAC.run(uv: [SIMD2(1, 1)], xyz: [SIMD3(0, 0, 1)], confs: [1], f: 360, centre: .zero,
                                   initR6: Pose.identity.r6, initT: .zero, maxError: 1.5, sampler: &s))
    }

    func testParityP4P() throws {
        _ = try Fixtures.dir("pose")
        let meta = try Fixtures.json(Fixtures.root.appendingPathComponent("meta.json"))
        let f = meta["f"] as! Double, c = SIMD2(meta["cx"] as! Double, meta["cy"] as! Double)
        let maxErr = meta["max_error"] as! Double
        for k in 0..<3 {
            let d = Fixtures.root.appendingPathComponent(String(format: "pose/frame_%03d", k))
            guard FileManager.default.fileExists(atPath: d.path) else { continue }
            let uv = Fixtures.vec2s(try Fixtures.npy(d, "corr_uv")).map { SIMD2<Double>($0) }
            let xyz = Fixtures.vec3s(try Fixtures.npy(d, "corr_xyz")).map { SIMD3<Double>($0) }
            let conf = try Fixtures.npy(d, "corr_conf").floats().map(Double.init)
            let idxs = try Fixtures.npy(d, "p4p_idxs"), rows = idxs.ints()
            var sampler = InjectedSampler(rows: (0..<idxs.shape[0]).map { Array(rows[($0 * 4)..<($0 * 4 + 4)]) })
            let initT = try Fixtures.npy(d, "p4p_init_t").floats().map(Double.init)
            let res = P4PRANSAC.run(uv: uv, xyz: xyz, confs: conf, f: f, centre: c,
                                    initR6: try Fixtures.npy(d, "p4p_init_r6").floats().map(Double.init),
                                    initT: SIMD3(initT[0], initT[1], initT[2]),
                                    hypotheses: idxs.shape[0], maxError: maxErr, sampler: &sampler)!
            // Hypothesis models against O-SCD's (row-major 3×4 [R | t]). Only hypotheses that could win
            // (at least half the best inlier count) are compared: LM on outlier-contaminated 4-point
            // samples is ill-conditioned, and fp32 (torch) and fp64 (here) legitimately diverge there.
            let models = try Fixtures.npy(d, "p4p_models").floats()
            let wantInl = try Fixtures.npy(d, "p4p_inliers").floats(), n = uv.count
            let best = try Fixtures.npy(d, "p4p_best").ints()[0]
            let counts = (0..<idxs.shape[0]).map { h in (0..<n).reduce(0) { $0 + (wantInl[h * n + $1] > 0 ? 1 : 0) } }
            var close = 0, contenders = 0
            for h in 0..<idxs.shape[0] where counts[h] * 2 >= counts[best] {
                contenders += 1
                let m = Array(models[(h * 12)..<(h * 12 + 12)])
                let want = Fixtures.pose(rowMajor4x4: m + [0, 0, 0, 1])
                let got = Pose(R: simd_float3x3(converting: res.models[h].R), t: SIMD3<Float>(res.models[h].t))
                if got.rotationDegrees(to: want) < 0.05 { close += 1 }
            }
            XCTAssertGreaterThanOrEqual(Double(close) / Double(max(contenders, 1)), 0.98, "frame \(k)")
            XCTAssertLessThanOrEqual(abs(res.inlierScores[res.bestIndex] - Double(counts[best])), max(2, Double(counts[best]) / 100), "frame \(k)")
            let agree = (0..<n).filter { res.inliers[$0] == (wantInl[best * n + $0] > 0) }.count
            XCTAssertGreaterThanOrEqual(Double(agree) / Double(n), 0.99, "frame \(k)")
        }
    }
}
