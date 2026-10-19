import XCTest
import simd
@testable import OSCDCore

final class OrbitCameraTests: XCTestCase {
    let target = SIMD3<Float>(0.1, -0.3, -0.2)

    func testLooksAtTheTargetFromItsEyeWithWorldUpUpInTheImage() {
        let cam = OrbitCamera(target: target, yaw: 0.8, pitch: 0.6, distance: 0.5)
        XCTAssertEqual(simd_distance(cam.pose.centre, cam.eye), 0, accuracy: 1e-5)
        XCTAssertEqual(simd_distance(cam.eye, target), 0.5, accuracy: 1e-5)
        let t = cam.pose.apply(target)                       // camera space, OpenCV axes
        XCTAssertEqual(t.x, 0, accuracy: 1e-5); XCTAssertEqual(t.y, 0, accuracy: 1e-5); XCTAssertEqual(t.z, 0.5, accuracy: 1e-5)
        XCTAssertLessThan(cam.pose.apply(target + SIMD3(0, 0.1, 0)).y, 0)   // above the target → up in the image (y down)
        XCTAssertGreaterThan(cam.eye.y, target.y)                           // positive pitch is above the table
    }

    func testStartsFromAKeyframesViewpoint() {
        let a = OrbitCamera(target: target, yaw: -1.2, pitch: 0.4, distance: 0.35)
        let b = OrbitCamera(lookingFrom: a.pose, at: target)
        XCTAssertEqual(b.yaw, a.yaw, accuracy: 1e-4)
        XCTAssertEqual(b.pitch, a.pitch, accuracy: 1e-4)
        XCTAssertEqual(b.distance, a.distance, accuracy: 1e-5)
    }

    func testOrbitAndZoomStayInRange() {
        var cam = OrbitCamera(target: target, yaw: 0, pitch: 0.5, distance: 0.4)
        cam.orbit(dYaw: 0.3, dPitch: 10)
        XCTAssertEqual(cam.yaw, 0.3, accuracy: 1e-6)
        XCTAssertEqual(cam.pitch, OrbitCamera.maxPitch, accuracy: 1e-6)
        cam.zoom(1000)
        XCTAssertEqual(cam.distance, OrbitCamera.minDistance, accuracy: 1e-6)
    }
}
