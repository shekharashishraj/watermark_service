import XCTest
import simd
@testable import OSCDCore

final class CameraTests: XCTestCase {
    func testProjectAndUnproject() {
        let K = Intrinsics(f: 100, cx: 50, cy: 40, width: 100, height: 80)
        XCTAssertEqual(K.project(SIMD3(1, 2, 4)), SIMD2(75, 90))
        let p = K.unproject(SIMD2(75, 90), depth: 4)
        XCTAssertEqual(p.x, 1, accuracy: 1e-5); XCTAssertEqual(p.y, 2, accuracy: 1e-5); XCTAssertEqual(p.z, 4)
    }

    func testOSCDCentreAndARKitScaling() {
        let c = Intrinsics.oscdCentered(f: 360, width: 480, height: 360)
        XCTAssertEqual(c.cx, 239.5); XCTAssertEqual(c.cy, 179.5)
        let k = Intrinsics.fromARKit(fx: 1440, fy: 1442, cx: 959.5, cy: 719.5,
                                     captureWidth: 1920, captureHeight: 1440, workingWidth: 480, workingHeight: 360)
        XCTAssertEqual(k.f, 360.25, accuracy: 1e-4)
        XCTAssertEqual(k.cx, 239.5, accuracy: 1e-4)
        XCTAssertEqual(k.cy, 179.5, accuracy: 1e-4)
    }

    func testSixDRoundTripAndOrthonormalisation() {
        let R = simd_double3x3(simd_quatd(angle: 0.7, axis: simd_normalize(SIMD3(1, 2, 3))))
        let back = SixD.toMatrix(SixD.fromMatrix(R))
        for c in 0..<3 { for r in 0..<3 { XCTAssertEqual(back[c][r], R[c][r], accuracy: 1e-12) } }
        // Non-orthonormal input: columns (2,0,0) and (0,3,0) in O-SCD's row-major (3,2) layout.
        let I = SixD.toMatrix([2, 0, 0, 3, 0, 0])
        XCTAssertEqual(I, matrix_identity_double3x3)
    }

    func testPoseInverseAndCentre() {
        let p = Pose(R: simd_float3x3(simd_quatf(angle: 0.3, axis: SIMD3(0, 1, 0))), t: SIMD3(0.1, -0.2, 0.5))
        let q = p.compose(p.inverse)
        XCTAssertLessThan(simd_length(q.t), 1e-6)
        XCTAssertLessThan(p.rotationDegrees(to: p), 1e-3)
        XCTAssertLessThan(simd_length(p.apply(p.centre)), 1e-6)
    }

    func testARKitAxesBecomeOpenCV() {
        // ARKit camera at the origin looking down -z. A point 2 m in front of it must land at z = +2.
        let pose = Pose.fromARKitCameraToWorld(matrix_identity_float4x4)
        let pc = pose.apply(SIMD3(0, 0, -2))
        XCTAssertEqual(pc, SIMD3(0, 0, 2))
        // A point above the camera (ARKit +y) is at negative image y in OpenCV axes.
        XCTAssertLessThan(pose.apply(SIMD3(0, 1, -2)).y, 0)
    }

    func testR6RoundTripThroughPose() {
        let p = Pose(R: simd_float3x3(simd_quatf(angle: 1.1, axis: simd_normalize(SIMD3(1, 1, 0)))), t: SIMD3(1, 2, 3))
        let q = Pose(r6: p.r6, t: p.tDouble)
        XCTAssertLessThan(p.rotationDegrees(to: q), 1e-3)
        XCTAssertEqual(q.t, p.t)
    }

    func testColumnMajorRoundTrip() {
        let m = Pose(R: simd_float3x3(simd_quatf(angle: 0.2, axis: SIMD3(1, 0, 0))), t: SIMD3(4, 5, 6)).matrix
        XCTAssertEqual(simd_float4x4(columnMajor: m.columnMajor), m)
    }
}
