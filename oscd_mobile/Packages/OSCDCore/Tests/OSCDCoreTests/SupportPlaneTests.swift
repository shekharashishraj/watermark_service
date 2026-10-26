import simd
import XCTest
@testable import OSCDCore

final class SupportPlaneTests: XCTestCase {
    let roi = AABB(min: SIMD3(-0.5, -0.47, -0.5), max: SIMD3(0.5, -0.1, 0.5))

    /// A table at y = −0.43 (±2 mm LiDAR noise) plus 20 % of the points scattered above it (objects, clutter).
    func tablePoints(_ s: inout SeededSampler) -> [SIMD3<Float>] {
        var pts: [SIMD3<Float>] = (0..<4000).map { _ in
            SIMD3(Float(s.uniform()) - 0.5, -0.43 + 0.004 * (Float(s.uniform()) - 0.5), Float(s.uniform()) - 0.5)
        }
        pts += (0..<1000).map { _ in SIMD3(Float(s.uniform()) - 0.5, -0.42 + 0.3 * Float(s.uniform()), Float(s.uniform()) - 0.5) }
        return pts
    }

    func testTableHeightIsTheModeOfY() throws {
        var s = SeededSampler(seed: 3)
        let y = try XCTUnwrap(SupportPlane.height(points: tablePoints(&s), roi: roi))
        XCTAssertEqual(y, -0.43, accuracy: 0.005)
    }

    func testPointsOutsideTheRoiAreIgnored() throws {
        var s = SeededSampler(seed: 4)
        // More floor points than table points, all below the ROI.
        let floor: [SIMD3<Float>] = (0..<9000).map { _ in SIMD3(Float(s.uniform()) - 0.5, -1.2, Float(s.uniform()) - 0.5) }
        let y = try XCTUnwrap(SupportPlane.height(points: tablePoints(&s) + floor, roi: roi))
        XCTAssertEqual(y, -0.43, accuracy: 0.005)
        XCTAssertEqual(try XCTUnwrap(SupportPlane.height(points: floor, roi: nil)), -1.2, accuracy: 0.005)
    }

    func testNoPointsGiveNil() {
        XCTAssertNil(SupportPlane.height(points: [], roi: roi))
        XCTAssertNil(SupportPlane.height(points: [SIMD3(0, -1.2, 0)], roi: roi))
    }

    /// Points spread over ten 1 cm bands, except that the band at y = `bandY` holds `share` of them.
    func bandedPoints(share: Float, bandY: Float = -0.43, count: Int = 4000, _ s: inout SeededSampler) -> [SIMD3<Float>] {
        let inBand = Int(Float(count) * share)
        var pts: [SIMD3<Float>] = (0..<inBand).map { _ in
            SIMD3(Float(s.uniform()) - 0.5, bandY + 0.002 + 0.006 * Float(s.uniform()), Float(s.uniform()) - 0.5)
        }
        // The rest evenly over twelve other bands (−0.40 … −0.29), each holding less than the band above.
        let rest = count - inBand
        pts += (0..<rest).map { k in
            let band = Float(k % 12)
            return SIMD3(Float(s.uniform()) - 0.5, -0.40 + 0.01 * band + 0.002 + 0.006 * Float(s.uniform()), Float(s.uniform()) - 0.5)
        }
        return pts
    }

    /// R37: the densest band must hold at least 15 % of the in-ROI points to be the table.
    func testDensestBandNeedsAMinimumShare() throws {
        var s = SeededSampler(seed: 7)
        let sparse = bandedPoints(share: 0.10, &s)
        let weak = try XCTUnwrap(SupportPlane.densestBand(points: sparse, roi: roi))
        XCTAssertEqual(weak.share, 0.10, accuracy: 0.005)
        XCTAssertEqual(weak.height, -0.425, accuracy: 0.005)
        XCTAssertEqual(weak.points, 4000)
        XCTAssertFalse(weak.isReliable)
        XCTAssertNil(SupportPlane.height(points: sparse, roi: roi))

        let dense = bandedPoints(share: 0.30, &s)
        let strong = try XCTUnwrap(SupportPlane.densestBand(points: dense, roi: roi))
        XCTAssertEqual(strong.share, 0.30, accuracy: 0.005)
        XCTAssertTrue(strong.isReliable)
        XCTAssertEqual(try XCTUnwrap(SupportPlane.height(points: dense, roi: roi)), -0.425, accuracy: 0.005)
    }

    func testShareCountsOnlyPointsInsideTheRoi() throws {
        var s = SeededSampler(seed: 8)
        let floor: [SIMD3<Float>] = (0..<20000).map { _ in SIMD3(Float(s.uniform()) - 0.5, -1.2, Float(s.uniform()) - 0.5) }
        let e = try XCTUnwrap(SupportPlane.densestBand(points: bandedPoints(share: 0.30, &s) + floor, roi: roi))
        XCTAssertEqual(e.share, 0.30, accuracy: 0.005)
        XCTAssertEqual(e.points, 4000)
    }

    func testHeightFromAScansLiDAR() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("support-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let m = SessionManifest(deviceModel: "test", captureWidth: 128, captureHeight: 96, depthWidth: 8, depthHeight: 6,
                                intrinsics: Intrinsics(f: 40, cx: 15.5, cy: 11.5, width: 32, height: 24))
        let store = try SessionStore(create: dir, manifest: m)
        // ARKit camera 0.5 m above the plane y = 0, looking straight down (camera +z = world +y).
        let pose: [Float] = [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0.5, 0, 1]
        for id in 0..<3 {
            try store.append(.keyframes, record: FrameRecord(id: id, timestamp: Double(id), arkitCameraToWorld: pose),
                             rgb: [UInt8](repeating: 100, count: 32 * 24 * 3), depth: [Float](repeating: 0.5, count: 48),
                             confidence: [UInt8](repeating: 2, count: 48))
        }
        XCTAssertEqual(try XCTUnwrap(SupportPlane.height(of: store, roi: nil)), 0, accuracy: 0.005)
        XCTAssertEqual(try XCTUnwrap(SupportPlane.densestBand(of: store, roi: nil)).share, 1, accuracy: 1e-6)
    }
}
