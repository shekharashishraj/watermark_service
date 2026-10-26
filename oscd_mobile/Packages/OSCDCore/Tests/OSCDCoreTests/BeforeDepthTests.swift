import simd
import XCTest
@testable import OSCDCore

final class BeforeDepthTests: XCTestCase {
    let k = Intrinsics.oscdCentered(f: 30, width: 32, height: 24)

    func testSplatKeepsTheNearestDepthOverATwoByTwoBlock() {
        // Both points project to (20.4, 13.7): the block is x 20–21, y 13–14.
        let near = SIMD3<Float>((20.4 - k.cx) / k.f * 0.4, (13.7 - k.cy) / k.f * 0.4, 0.4)
        let far = near * 1.5
        let d = BeforeDepth.splat(points: [far, near, SIMD3(0, 0, -1)], pose: .identity, intrinsics: k, splat: 2)
        let covered = d.indices.filter { d[$0].isFinite }
        XCTAssertEqual(covered.map { [$0 % 32, $0 / 32] }, [[20, 13], [21, 13], [20, 14], [21, 14]])
        for i in covered { XCTAssertEqual(d[i], 0.4, accuracy: 1e-6) }
        // The default 3×3 block is centred on the nearest pixel (20, 14).
        let d3 = BeforeDepth.splat(points: [far, near], pose: .identity, intrinsics: k)
        let covered3 = d3.indices.filter { d3[$0].isFinite }
        XCTAssertEqual(Set(covered3.map { [$0 % 32, $0 / 32] }), Set((13...15).flatMap { y in (19...21).map { [$0, y] } }))
        for i in covered3 { XCTAssertEqual(d3[i], 0.4, accuracy: 1e-6) }
        // One pixel per point with splat 1; a point outside the view adds nothing.
        let one = BeforeDepth.splat(points: [near, SIMD3(5, 0, 0.4)], pose: .identity, intrinsics: k, splat: 1)
        XCTAssertEqual(one.indices.filter { one[$0].isFinite }, [14 * 32 + 20])
    }

    /// Points at the camera plane: one on the axis would write a depth of ≈ 0 at the image centre, and one off the axis
    /// projects to a uv so large that converting it to Int traps. Both add nothing; a normal point still splats.
    func testPointsAtTheCameraPlaneAreSkipped() {
        let onAxis = SIMD3<Float>(0, 0, 1e-6), offAxis = SIMD3<Float>(0.5, 0, 1e-6), huge = SIMD3<Float>(0.5, 0.5, 1e-30)
        let d = BeforeDepth.splat(points: [onAxis, offAxis, huge, SIMD3(0.5, 0.5, -1e-30)], pose: .identity, intrinsics: k)
        XCTAssertTrue(d.allSatisfy { $0.isNaN })
        let near = SIMD3<Float>((20.4 - k.cx) / k.f * 0.4, (13.7 - k.cy) / k.f * 0.4, 0.4)
        let mixed = BeforeDepth.splat(points: [huge, onAxis, near], pose: .identity, intrinsics: k)
        XCTAssertEqual(mixed.filter { $0.isFinite }.count, 9)
        XCTAssertEqual(mixed.compactMap { $0.isFinite ? $0 : nil }.min() ?? 0, 0.4, accuracy: 1e-6)
    }

    func testSplatUsesThePose() {
        // A camera 0.5 m above the plane y = 0, looking down: camera z = 0.5 − world y.
        let down = Pose(R: simd_float3x3(rows: [SIMD3(1, 0, 0), SIMD3(0, 0, 1), SIMD3(0, -1, 0)]), t: SIMD3(0, 0, 0.5))
        var pts: [SIMD3<Float>] = []
        for x in stride(from: Float(-0.3), through: 0.3, by: 0.005) {
            for z in stride(from: Float(-0.3), through: 0.3, by: 0.005) { pts.append(SIMD3(x, 0, z)) }
        }
        let d = BeforeDepth.splat(points: pts, pose: down, intrinsics: k)
        XCTAssertTrue(d.allSatisfy { $0.isFinite })
        for v in d { XCTAssertEqual(v, 0.5, accuracy: 1e-5) }
    }

    func testPointsFromAScanAreInsideTheRoiAndOnePerVoxel() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("before-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let m = SessionManifest(deviceModel: "test", captureWidth: 128, captureHeight: 96, depthWidth: 8, depthHeight: 6,
                                intrinsics: Intrinsics(f: 40, cx: 15.5, cy: 11.5, width: 32, height: 24))
        let store = try SessionStore(create: dir, manifest: m)
        // ARKit camera 0.5 m above the plane y = 0, looking straight down (camera +z = world +y).
        let pose: [Float] = [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0.5, 0, 1]
        var conf = [UInt8](repeating: 2, count: 48)
        conf[0] = 1                                                         // low confidence: ignored
        for id in 0..<2 {
            try store.append(.keyframes, record: FrameRecord(id: id, timestamp: Double(id), arkitCameraToWorld: pose),
                             rgb: [UInt8](repeating: 100, count: 32 * 24 * 3), depth: [Float](repeating: 0.5, count: 48),
                             confidence: conf)
        }
        let all = try BeforeDepth.points(of: store, roi: nil, voxel: 0.005)
        // 47 confident pixels per keyframe, the same in both: one point each (8 × 6 depth pixels are ≈ 5 cm apart).
        XCTAssertEqual(all.count, 47)
        for p in all { XCTAssertEqual(p.y, 0, accuracy: 1e-5) }
        let roi = AABB(min: SIMD3(0, -0.1, -1), max: SIMD3(1, 0.1, 1))
        let inside = try BeforeDepth.points(of: store, roi: roi)
        XCTAssertFalse(inside.isEmpty)
        XCTAssertLessThan(inside.count, all.count)
        XCTAssertTrue(inside.allSatisfy { roi.contains($0) })
    }
}
