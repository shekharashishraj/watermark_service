import XCTest
import simd
@testable import OSCDCore

final class SceneRegionTests: XCTestCase {
    func testBoxCoversTheBulkPlusMarginAndIgnoresStrayPoints() {
        var s = SeededSampler(seed: 9)
        // A 4 × 2.5 × 3 m room of points, plus 1 % far-away noise (LiDAR outliers).
        var pts: [SIMD3<Double>] = (0..<5000).map { _ in SIMD3(4 * s.uniform() - 2, 2.5 * s.uniform(), 3 * s.uniform() - 1) }
        pts += (0..<50).map { _ in SIMD3(40, 30, -40) }
        let box = SceneRegion.bounds(of: pts)
        XCTAssertEqual(box.min.x, -2 - 0.1, accuracy: 0.06)
        XCTAssertEqual(box.max.x, 2 + 0.1, accuracy: 0.06)
        XCTAssertEqual(box.max.y, 2.5 + 0.1, accuracy: 0.06)
        XCTAssertLessThan(box.max.z, 3)            // the outliers at z = −40 / x = 40 are excluded
        XCTAssertLessThan(box.max.x, 3)
    }

    func testEstimatesFromASessionsLiDAR() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("region-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let m = SessionManifest(deviceModel: "test", captureWidth: 128, captureHeight: 96, depthWidth: 8, depthHeight: 6,
                                intrinsics: Intrinsics(f: 40, cx: 15.5, cy: 11.5, width: 32, height: 24))
        let store = try SessionStore(create: dir, manifest: m)
        // Identity ARKit pose: a wall 1.5 m in front (ARKit −z).
        let pose: [Float] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        try store.append(.keyframes, record: FrameRecord(id: 0, timestamp: 0, arkitCameraToWorld: pose),
                         rgb: [UInt8](repeating: 100, count: 32 * 24 * 3), depth: [Float](repeating: 1.5, count: 48),
                         confidence: [UInt8](repeating: 2, count: 48))
        let box = try XCTUnwrap(SceneRegion.estimate(store))
        XCTAssertEqual(box.min.z, -1.5 - 0.1, accuracy: 0.01)
        XCTAssertEqual(box.max.z, -1.5 + 0.1, accuracy: 0.01)
        XCTAssertTrue(box.contains(SIMD3(0, 0, -1.5)))
    }
    /// ARKit camera-to-world for a camera at `eye` looking at `target` (y up, looking along −z).
    private func lookAt(_ eye: SIMD3<Float>, _ target: SIMD3<Float>) -> simd_float4x4 {
        let back = simd_normalize(eye - target)
        let right = simd_normalize(simd_cross(SIMD3(0, 1, 0), back))
        let up = simd_cross(back, right)
        return simd_float4x4(SIMD4(right, 0), SIMD4(up, 0), SIMD4(back, 0), SIMD4(eye, 1))
    }

    func testFocusIsWhereASweepLooks() {
        let table = SIMD3<Float>(0.2, 0.75, -1)
        let poses = stride(from: 0, to: 2 * Float.pi, by: 0.4).map {
            lookAt(table + SIMD3(0.4 * sin($0), 0.3, 0.4 * cos($0)), table)
        }
        let box = AABB(min: SIMD3(-3, -0.1, -4), max: SIMD3(3, 2.6, 2))
        let f = SceneRegion.focus(arkitCameraToWorld: poses, within: box)
        XCTAssertLessThan(simd_distance(f, table), 1e-3)
    }

    func testFocusFallsBackToTheBoxCentreForParallelViews() {
        let poses = (0..<5).map { lookAt(SIMD3(Float($0) * 0.1, 1, 0), SIMD3(Float($0) * 0.1, 1, -2)) }
        let box = AABB(min: SIMD3(-1, 0, -3), max: SIMD3(1, 2, 1))
        XCTAssertEqual(SceneRegion.focus(arkitCameraToWorld: poses, within: box), SIMD3(0, 1, -1))
    }
}
