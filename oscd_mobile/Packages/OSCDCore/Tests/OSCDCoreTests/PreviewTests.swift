import XCTest
import simd
@testable import OSCDCore

final class PreviewTests: XCTestCase {
    /// 4×4 working image, depth at the same resolution, identity pose (reference world = camera).
    let K = Intrinsics(f: 4, cx: 1.5, cy: 1.5, width: 4, height: 4)
    let roi = AABB(min: SIMD3(-1, -1, 0.5), max: SIMD3(1, 1, 1.5))
    func covered() -> FloatImage { FloatImage(width: 4, height: 4, channels: 3, fill: 0.5) }

    func testValidityNeedsCoverageConfidentDepthAndROI() {
        var depth = [Float](repeating: 1, count: 16), conf = [UInt8](repeating: 2, count: 16)
        var render = covered()
        depth[1] = .nan          // missing LiDAR
        conf[2] = 1              // low confidence
        depth[3] = 3             // surface outside the ROI
        render[0, 1, 0] = 0; render[1, 1, 0] = 0; render[2, 1, 0] = 0   // pixel (x 0, y 1): reference has nothing
        let v = PreviewValidity.mask(depth: depth, confidence: conf, depthWidth: 4, depthHeight: 4,
                                     intrinsics: K, pose: .identity, roi: roi, render: render)
        XCTAssertTrue(v[0])
        XCTAssertFalse(v[1]); XCTAssertFalse(v[2]); XCTAssertFalse(v[3]); XCTAssertFalse(v[4])
        XCTAssertEqual(v.filter { $0 }.count, 12)     // 16 minus the 4 invalidated pixels
    }

    /// The PnP pose places LiDAR points in the reference world: moving the camera 2 m back puts
    /// the same surface outside the ROI.
    func testValidityUsesThePoseToReachTheReferenceWorld() {
        let back = Pose(R: matrix_identity_float3x3, t: SIMD3(0, 0, -2))   // world→camera: camera centre at z = +2
        let v = PreviewValidity.mask(depth: [Float](repeating: 1, count: 16), confidence: [UInt8](repeating: 2, count: 16),
                                     depthWidth: 4, depthHeight: 4, intrinsics: K, pose: back, roi: roi, render: covered())
        XCTAssertFalse(v.contains(true))
    }

    /// Lower-resolution LiDAR (2×2) is sampled by nearest neighbour.
    func testValidityUpsamplesDepthByNearestNeighbour() {
        let v = PreviewValidity.mask(depth: [1, 1, 1, 3], confidence: [2, 2, 2, 2], depthWidth: 2, depthHeight: 2,
                                     intrinsics: K, pose: .identity, roi: roi, render: covered())
        XCTAssertTrue(v[0]); XCTAssertFalse(v[15]); XCTAssertFalse(v[2 * 4 + 2])
    }

    func testOverlayTintsChangedAndDimsInvalid() {
        let rgb = [UInt8](repeating: 100, count: 3 * 3)                 // 3 pixels
        let out = PreviewComposite.overlay(rgb8: rgb, changed: [true, false, false], valid: [true, true, false])
        XCTAssertEqual(out.count, 3 * 4)
        XCTAssertGreaterThan(out[0], 150); XCTAssertLessThan(out[1], 100)  // red tint
        XCTAssertEqual(Array(out[4..<7]), [100, 100, 100])                // unchanged, valid
        XCTAssertEqual(Array(out[8..<11]), [35, 35, 35])                  // outside the judged area: dimmed
        XCTAssertEqual(out[3], 255)
    }
}
