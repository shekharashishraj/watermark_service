import simd
import XCTest
@testable import OSCDCore

final class PlanarICPTests: XCTestCase {
    /// Surface points of an axis-aligned box centred at the origin, on a `step` grid per face.
    func boxSurface(size: SIMD3<Float>, step: Float) -> [SIMD3<Float>] {
        let h = size / 2
        func axis(_ half: Float) -> [Float] {
            let n = max(1, Int((2 * half / step).rounded()))
            return (0...n).map { -half + 2 * half * Float($0) / Float(n) }
        }
        var pts: [SIMD3<Float>] = []
        for x in axis(h.x) { for y in axis(h.y) { pts.append([x, y, -h.z]); pts.append([x, y, h.z]) } }
        for x in axis(h.x) { for z in axis(h.z) { pts.append([x, -h.y, z]); pts.append([x, h.y, z]) } }
        for y in axis(h.y) { for z in axis(h.z) { pts.append([-h.x, y, z]); pts.append([h.x, y, z]) } }
        return pts
    }

    /// Roughly even points on a sphere (Fibonacci spiral).
    func sphereSurface(radius: Float, n: Int) -> [SIMD3<Float>] {
        let golden = Float.pi * (3 - Float(5).squareRoot())
        return (0..<n).map { i in
            let y = 1 - 2 * (Float(i) + 0.5) / Float(n)
            let r = (1 - y * y).squareRoot()
            let a = golden * Float(i)
            return radius * SIMD3<Float>(r * cos(a), y, r * sin(a))
        }
    }

    func moved(_ pts: [SIMD3<Float>], yawDeg: Float, by offset: SIMD3<Float>) -> [SIMD3<Float>] {
        let R = PlanarICP.rotationY(yawDeg)
        return pts.map { R * $0 + offset }
    }

    let box = SIMD3<Float>(0.10, 0.04, 0.05)
    let offset = SIMD3<Float>(0.20, 0.0, 0.05)

    func assertTranslation(_ m: Motion, _ expected: SIMD3<Float>, accuracy: Float, file: StaticString = #filePath, line: UInt = #line) {
        for k in 0..<3 { XCTAssertEqual(m.translation[k], expected[k], accuracy: accuracy, file: file, line: line) }
    }

    /// Distance in the table plane between the fitted and the true displacement.
    func horizontalError(_ m: Motion, _ expected: SIMD3<Float>) -> Float {
        simd_length(SIMD2<Float>(m.translation.x - expected.x, m.translation.z - expected.z))
    }

    /// Either no yaw claimed, or one near `expected`.
    func assertYawNilOrNear(_ m: Motion, _ expected: Float, accuracy: Float, file: StaticString = #filePath, line: UInt = #line) {
        if let yaw = m.yawDeg { XCTAssertEqual(yaw, expected, accuracy: accuracy, file: file, line: line) }
    }

    func testRotationYIsRightHandedAboutUp() {
        let p = PlanarICP.rotationY(90) * SIMD3<Float>(1, 0, 0)
        XCTAssertEqual(p.x, 0, accuracy: 1e-5); XCTAssertEqual(p.z, -1, accuracy: 1e-5)
        let up = PlanarICP.rotationY(37) * SIMD3<Float>(0, 1, 0)
        XCTAssertEqual(up.y, 1, accuracy: 1e-6)
    }

    func testRecoversTranslationAndYawOfABox() throws {
        let src = boxSurface(size: box, step: 0.005)
        let m = try XCTUnwrap(PlanarICP.fit(source: src, target: moved(src, yawDeg: 40, by: offset)))
        XCTAssertEqual(try XCTUnwrap(m.yawDeg), 40, accuracy: 5)
        assertTranslation(m, offset, accuracy: 0.01)
        XCTAssertEqual(m.horizontalDistance, 0.206, accuracy: 0.01)
        XCTAssertGreaterThan(m.inlierFraction, 0.9)
    }

    func testHalfTurnIsReportedAsTheSmallerRotation() throws {
        let src = boxSurface(size: box, step: 0.005)
        let m = try XCTUnwrap(PlanarICP.fit(source: src, target: moved(src, yawDeg: 220, by: offset)))
        XCTAssertEqual(try XCTUnwrap(m.yawDeg), 40, accuracy: 5)
        assertTranslation(m, offset, accuracy: 0.01)
    }

    func testYawIsUnknownForASphere() throws {
        let src = sphereSurface(radius: 0.033, n: 800)
        let by = SIMD3<Float>(0.15, 0, 0)
        let m = try XCTUnwrap(PlanarICP.fit(source: src, target: src.map { $0 + by }))
        XCTAssertNil(m.yawDeg)
        assertTranslation(m, by, accuracy: 0.01)
    }

    func testYawIsUnknownForACube() throws {
        let src = boxSurface(size: SIMD3<Float>(repeating: 0.06), step: 0.005)
        let m = try XCTUnwrap(PlanarICP.fit(source: src, target: moved(src, yawDeg: 40, by: offset)))
        XCTAssertNil(m.yawDeg)
        assertTranslation(m, offset, accuracy: 0.01)
    }

    func testTooFewPointsGivesNil() {
        let src = boxSurface(size: box, step: 0.005)
        XCTAssertNil(PlanarICP.fit(source: Array(src.prefix(5)), target: src))
        XCTAssertNil(PlanarICP.fit(source: src, target: []))
    }

    func testPartialOverlapStillFits() throws {
        let src = boxSurface(size: box, step: 0.005)
        // LiDAR sees one side only: drop the half of the target that lies beyond its centre along z.
        let target = moved(src, yawDeg: 0, by: offset).filter { $0.z <= offset.z }
        let m = try XCTUnwrap(PlanarICP.fit(source: src, target: target))
        XCTAssertLessThan(horizontalError(m, offset), 0.015)
    }

    func testPartialOverlapWithYawStillFits() throws {
        let src = boxSurface(size: box, step: 0.005)
        let target = moved(src, yawDeg: 40, by: offset).filter { $0.z <= offset.z }
        let m = try XCTUnwrap(PlanarICP.fit(source: src, target: target))
        XCTAssertLessThan(horizontalError(m, offset), 0.02)
        assertYawNilOrNear(m, 40, accuracy: 10)
    }

    /// Both clouds are one-sided: the reference render and the LiDAR each see only the half facing -z.
    func testBothSidesPartialStillFits() throws {
        let full = boxSurface(size: box, step: 0.005)
        let src = full.filter { $0.z <= 0 }
        let target = moved(full, yawDeg: 40, by: offset).filter { $0.z <= offset.z }
        let m = try XCTUnwrap(PlanarICP.fit(source: src, target: target))
        XCTAssertLessThan(horizontalError(m, offset), 0.02)
        assertYawNilOrNear(m, 40, accuracy: 10)
    }

    /// A table patch 2.5 cm under the box is in the target but not in the source.
    func testFloorClutterDoesNotBreakTheFit() throws {
        let src = boxSurface(size: box, step: 0.005)
        var target = moved(src, yawDeg: 40, by: offset)
        let floorY = offset.y - box.y / 2 - 0.025
        for i in 0...20 { for j in 0...20 {
            target.append(SIMD3<Float>(offset.x - 0.05 + 0.005 * Float(i), floorY, offset.z - 0.05 + 0.005 * Float(j)))
        } }
        let m = try XCTUnwrap(PlanarICP.fit(source: src, target: target))
        assertYawNilOrNear(m, 40, accuracy: 5)
        XCTAssertLessThan(horizontalError(m, offset), 0.005)
    }

    /// The turn is about the source centroid, so `translation` is the centroid's displacement wherever the box sits.
    func testRotatesAboutTheSourceCentroid() throws {
        let c = SIMD3<Float>(0.3, 0.02, -0.2), shift = SIMD3<Float>(0.15, 0.01, 0.05)
        let src = boxSurface(size: box, step: 0.005).map { $0 + c }
        let R = PlanarICP.rotationY(40)
        let target = src.map { R * ($0 - c) + c + shift }
        let m = try XCTUnwrap(PlanarICP.fit(source: src, target: target))
        assertTranslation(m, shift, accuracy: 0.01)
        XCTAssertEqual(try XCTUnwrap(m.yawDeg), 40, accuracy: 5)
    }

    func testUnusableConfigGivesNil() {
        let src = boxSurface(size: box, step: 0.005)
        let target = moved(src, yawDeg: 0, by: offset)
        var c = PlanarICP.Config(); c.nnVoxel = 0
        XCTAssertNil(PlanarICP.fit(source: src, target: target, config: c))
        c = PlanarICP.Config(); c.iterations = -1
        XCTAssertNil(PlanarICP.fit(source: src, target: target, config: c))
        c = PlanarICP.Config(); c.yawCandidates = 0
        XCTAssertNil(PlanarICP.fit(source: src, target: target, config: c))
    }

    /// `maxIterationPoints` is a hard cap, also between one and two times the cap, and the sample spans the array.
    func testIterationPointCapIsHonoured() {
        let pts = (0..<1199).map { SIMD3<Float>(Float($0), 0, 0) }
        let sample = PlanarICP.iterationSample(pts, cap: 600)
        XCTAssertEqual(sample.count, 600)
        XCTAssertEqual(sample.first?.x, 0)
        XCTAssertGreaterThan(sample.last?.x ?? 0, 1190)
        XCTAssertEqual(PlanarICP.iterationSample(Array(pts.prefix(600)), cap: 600).count, 600)
        XCTAssertEqual(Set(sample.map { Int($0.x) % 2 }).count, 2)
    }

    /// Non-finite points are dropped before fitting; the fit uses the rest.
    func testNonFinitePointsAreIgnored() throws {
        var src = boxSurface(size: box, step: 0.005)
        let target = moved(src, yawDeg: 40, by: offset)
        src.append(SIMD3<Float>(.nan, 0, 0)); src.append(SIMD3<Float>(0, .infinity, 0))
        let m = try XCTUnwrap(PlanarICP.fit(source: src, target: target))
        XCTAssertTrue(m.translation.x.isFinite && m.translation.y.isFinite && m.translation.z.isFinite)
        XCTAssertTrue(m.horizontalDistance.isFinite && m.fitRMS.isFinite && m.inlierFraction.isFinite)
        assertTranslation(m, offset, accuracy: 0.01)
        // Nothing finite left: too few points.
        XCTAssertNil(PlanarICP.fit(source: [SIMD3<Float>](repeating: SIMD3<Float>(.nan, 0, 0), count: 50), target: target))
    }

    // MARK: - Seeded noise matrix

    /// Smallest difference between two yaws modulo a half turn, in degrees (0...90).
    func yawGap(_ a: Float, _ b: Float) -> Float {
        var d = (a - b).truncatingRemainder(dividingBy: 180)
        if d < 0 { d += 180 }
        return min(d, 180 - d)
    }

    func noisy(_ pts: [SIMD3<Float>], sigma: Float, rng: inout SeededSampler) -> [SIMD3<Float>] {
        guard sigma > 0 else { return pts }
        func gauss() -> Float {
            let u1 = max(rng.uniform(), 1e-12), u2 = rng.uniform()
            return Float((-2 * log(u1)).squareRoot() * cos(2 * Double.pi * u2))
        }
        return pts.map { $0 + sigma * SIMD3<Float>(gauss(), gauss(), gauss()) }
    }

    struct MatrixOutcome { var failures: [String] = []; var wrong = 0; var boxFullCorrect0 = 0; var boxFullCorrectNoisy = 0 }

    /// Shapes x views x yaws x noise levels; a reported yaw must be within 10 deg of the truth modulo 180 (never reported
    /// for a sphere or a whole cube), box fits must stay within 2 cm and the full-view box must be recognised.
    func runMatrix(config: PlanarICP.Config) -> MatrixOutcome {
        let shapes: [(name: String, pts: [SIMD3<Float>])] = [
            ("box", boxSurface(size: box, step: 0.005)),
            ("cube", boxSurface(size: SIMD3<Float>(repeating: 0.06), step: 0.005)),
            ("sphere", sphereSurface(radius: 0.033, n: 800))]
        var out = MatrixOutcome()
        var seed: UInt64 = 1
        for shape in shapes { for view in ["full", "half", "both"] { for yaw: Float in [17, 40, 130] { for sigma: Float in [0, 0.002] {
            seed += 1
            var rng = SeededSampler(seed: seed)
            let srcClean = view == "both" ? shape.pts.filter { $0.z <= 0 } : shape.pts
            var tgtClean = moved(shape.pts, yawDeg: yaw, by: offset)
            if view != "full" { tgtClean = tgtClean.filter { $0.z <= offset.z } }
            let src = noisy(srcClean, sigma: sigma, rng: &rng), tgt = noisy(tgtClean, sigma: sigma, rng: &rng)
            let name = "\(shape.name)/\(view)/yaw \(Int(yaw))/sigma \(sigma)"
            guard let m = PlanarICP.fit(source: src, target: tgt, config: config) else {
                out.failures.append("\(name): fit returned nil"); continue
            }
            if shape.name == "box", horizontalError(m, offset) >= 0.02 {
                out.failures.append("\(name): horizontal error \(horizontalError(m, offset))")
            }
            guard let reported = m.yawDeg else { continue }
            // A sphere never has a yaw, and a cube is 4-fold symmetric when seen whole; a clipped cube may show one.
            let right = shape.name == "box" || (shape.name == "cube" && view != "full") ? yawGap(reported, yaw) <= 10 : false
            if !right { out.wrong += 1; out.failures.append("\(name): wrong yaw \(reported) (fitRMS \(m.fitRMS))") }
            if shape.name == "box", view == "full", right { if sigma == 0 { out.boxFullCorrect0 += 1 } else { out.boxFullCorrectNoisy += 1 } }
        } } } }
        if out.boxFullCorrect0 < 3 { out.failures.append("box/full at sigma 0: only \(out.boxFullCorrect0) of 3 yaws found") }
        if out.boxFullCorrectNoisy < 2 { out.failures.append("box/full at sigma 0.002: only \(out.boxFullCorrectNoisy) of 3 yaws found") }
        return out
    }

    func testYawNeverWrongOnNoisyMatrix() throws {
        #if DEBUG
        throw XCTSkip("54 fits are too slow in Debug; run with -c release")
        #endif
        let out = runMatrix(config: PlanarICP.Config())
        if !out.failures.isEmpty { XCTFail("\(out.failures.count) problems:\n" + out.failures.joined(separator: "\n")) }
    }
}
