import XCTest
import simd
@testable import OSCDCore

final class MatcherTests: XCTestCase {
    func unit(_ v: [Float]) -> [Float] { let n = sqrt(v.reduce(0) { $0 + $1 * $1 }); return v.map { $0 / n } }

    func testMutualNearestAndThreshold() {
        var a = [Float](repeating: 0, count: 3 * 64), b = [Float](repeating: 0, count: 3 * 64)
        a[0 * 64 + 0] = 1; a[1 * 64 + 1] = 1                 // a2 stays all-zero (invalid)
        b[0 * 64 + 1] = 1                                     // b0 ↔ a1
        b[1 * 64 + 0] = 1                                     // b1 ↔ a0
        let w = unit([0.5, 0.5] + [Float](repeating: 0, count: 62))   // cos 0.707 with a0/a1: below 0.82
        for c in 0..<64 { b[2 * 64 + c] = w[c] }
        let ka = DescribedKeypoints(kpts: [.zero, .zero, .zero], feats: a)
        let kb = DescribedKeypoints(kpts: [.zero, .zero, .zero], feats: b)
        let m = Matcher.match(ka, kb)
        XCTAssertEqual(m.idx, [0, 1]); XCTAssertEqual(m.idxOther, [1, 0])
        XCTAssertEqual(Matcher.countMutual(ka, kb), 2)
    }

    func testEightPointRecoversEpipolarGeometry() {
        // Two cameras 0.3 m apart; F-RANSAC must keep every true correspondence and drop the outliers.
        let K = Intrinsics(f: 300, cx: 240, cy: 180, width: 480, height: 360)
        let cam2 = Pose(R: simd_float3x3(simd_quatf(angle: 0.1, axis: SIMD3(0, 1, 0))), t: SIMD3(-0.3, 0, 0))
        var sampler = SeededSampler(seed: 1)
        var p1: [SIMD2<Float>] = [], p2: [SIMD2<Float>] = []
        for _ in 0..<200 {
            let X = SIMD3(Float(sampler.uniform()) - 0.5, Float(sampler.uniform()) - 0.5, 2 + Float(sampler.uniform()))
            p1.append(K.project(X)); p2.append(K.project(cam2.apply(X)))
        }
        for _ in 0..<40 { p1.append(SIMD2(Float(sampler.uniform()) * 480, Float(sampler.uniform()) * 360)); p2.append(SIMD2(Float(sampler.uniform()) * 480, Float(sampler.uniform()) * 360)) }
        let res = FundamentalRANSAC.run(p1, p2, hypotheses: 2000, maxError: 1.5, sampler: &sampler)!
        XCTAssertGreaterThanOrEqual(res.inliers[0..<200].filter { $0 }.count, 195)
        XCTAssertLessThanOrEqual(res.inliers[200...].filter { $0 }.count, 4)
    }

    /// Review Focus 1: fewer than 8 matches is a clean "no model", not a crash.
    func testTooFewMatchesReturnsNil() {
        var s = SeededSampler(seed: 0)
        let pts = [SIMD2<Float>](repeating: SIMD2(1, 2), count: 5)
        XCTAssertNil(FundamentalRANSAC.run(pts, pts, hypotheses: 10, maxError: 1.5, sampler: &s))
        let kp = DescribedKeypoints(kpts: [], feats: [])
        XCTAssertEqual(Matcher.matchFiltered(kp, kp, maxError: 1.5, sampler: &s).count, 0)
    }

    func testParityMutualNN() throws {
        let d = try Fixtures.dir("match")
        let a = DescribedKeypoints.load(kpts: try Fixtures.npy(d, "kpts_a"), feats: try Fixtures.npy(d, "feats_a"))
        let b = DescribedKeypoints.load(kpts: try Fixtures.npy(d, "kpts_b"), feats: try Fixtures.npy(d, "feats_b"))
        let (m12, mask) = Matcher.mutualNearest(a, b)
        let wantMask = try Fixtures.npy(d, "mask").bools(), want12 = try Fixtures.npy(d, "match12").ints()
        var agree = 0, total = 0
        for i in 0..<a.count where mask[i] || wantMask[i] {
            total += 1
            if mask[i] == wantMask[i] && m12[i] == want12[i] { agree += 1 }
        }
        XCTAssertGreaterThanOrEqual(Double(agree) / Double(max(total, 1)), 0.999)
    }

    func testParityFundamentalRANSAC() throws {
        let d = try Fixtures.dir("fransac")
        let p1 = Fixtures.vec2s(try Fixtures.npy(d, "kpts1")), p2 = Fixtures.vec2s(try Fixtures.npy(d, "kpts2"))
        let idxs = try Fixtures.npy(d, "idxs"), rows = idxs.ints()
        var sampler = InjectedSampler(rows: (0..<idxs.shape[0]).map { Array(rows[($0 * 8)..<($0 * 8 + 8)]) })
        let res = FundamentalRANSAC.run(p1, p2, hypotheses: idxs.shape[0], maxError: try Fixtures.npy(d, "max_error").floats()[0], sampler: &sampler)!
        let wantInl = try Fixtures.npy(d, "inliers").bools(), n = p1.count
        let wantBest = try Fixtures.npy(d, "best_id").ints()[0]
        let wantCount = (0..<n).filter { wantInl[wantBest * n + $0] }.count
        XCTAssertLessThanOrEqual(abs(res.inlierCounts[res.bestIndex] - wantCount), max(2, wantCount / 100))
        var agree = 0
        for k in 0..<n where res.inliers[k] == wantInl[wantBest * n + k] { agree += 1 }
        XCTAssertGreaterThanOrEqual(Double(agree) / Double(n), 0.995)
    }
}
