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
        assertTranslation(m, offset, accuracy: 0.02)
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
}
