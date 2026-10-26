import XCTest
import simd
@testable import OSCDCore

final class ObjectProjectionTests: XCTestCase {
    /// Camera at the origin looking down +z, f 400, 480×360, principal point at the centre.
    let K = Intrinsics.oscdCentered(f: 400, width: 480, height: 360)

    func testProjectionBehindCameraIsNil() {
        XCTAssertNil(ObjectProjection.project(SIMD3(0, 0, -0.5), pose: .identity, intrinsics: K, orientation: .none))
        XCTAssertNil(ObjectProjection.project(SIMD3(0, 0, 0.005), pose: .identity, intrinsics: K, orientation: .none))
        XCTAssertNotNil(ObjectProjection.project(SIMD3(0, 0, 0.5), pose: .identity, intrinsics: K, orientation: .none))
    }

    func testSensorRightMapsCornersToPortrait() {
        func d(_ u: Float, _ v: Float) -> SIMD2<Float> {
            ObjectProjection.displayed(SIMD2(u, v), width: 480, height: 360, orientation: .sensorRight)
        }
        XCTAssertEqual(d(0, 0), SIMD2(359, 0))
        XCTAssertEqual(d(479, 0), SIMD2(359, 479))
        XCTAssertEqual(d(0, 359), SIMD2(0, 0))
        let s = ObjectProjection.displayedSize(width: 480, height: 360, orientation: .sensorRight)
        XCTAssertEqual(s.width, 360); XCTAssertEqual(s.height, 480)
        let n = ObjectProjection.displayedSize(width: 480, height: 360, orientation: .none)
        XCTAssertEqual(n.width, 480); XCTAssertEqual(n.height, 360)
        XCTAssertEqual(ObjectProjection.displayed(SIMD2(12, 34), width: 480, height: 360, orientation: .none), SIMD2(12, 34))
    }

    func testFarOutsidePointIsNil() {
        // u = f x / z + cx: x = -0.85 at z = 1 gives u = -100.5 (within 25 % of 480); x = -1.1 gives u = -200.5.
        XCTAssertNotNil(ObjectProjection.project(SIMD3(-0.85, 0, 1), pose: .identity, intrinsics: K, orientation: .none))
        XCTAssertNil(ObjectProjection.project(SIMD3(-1.1, 0, 1), pose: .identity, intrinsics: K, orientation: .none))
        XCTAssertNil(ObjectProjection.project(SIMD3(0, 1.0, 1), pose: .identity, intrinsics: K, orientation: .none))   // v = 579 > 1.25·360
    }

    func testArrowEndpointsForAMovedEventProjectIntoTheDisplayedImage() {
        let tail = ObjectProjection.project(SIMD3(-0.05, 0, 0.5), pose: .identity, intrinsics: K, orientation: .sensorRight)
        let head = ObjectProjection.project(SIMD3(0.05, 0, 0.5), pose: .identity, intrinsics: K, orientation: .sensorRight)
        guard let tail, let head else { return XCTFail("both endpoints must project") }
        for p in [tail, head] {
            XCTAssertTrue(p.x >= 0 && p.x < 360, "x \(p.x) outside the displayed width")
            XCTAssertTrue(p.y >= 0 && p.y < 480, "y \(p.y) outside the displayed height")
        }
        XCTAssertLessThan(tail.y, head.y)   // +x in the sensor image is down in the portrait display
    }
}
