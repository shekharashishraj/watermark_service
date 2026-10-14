import XCTest
import simd
@testable import OSCDCore

final class StoreTests: XCTestCase {
    func manifest() -> SessionManifest {
        SessionManifest(deviceModel: "test", captureWidth: 8, captureHeight: 4, depthWidth: 2, depthHeight: 1,
                        intrinsics: Intrinsics(f: 3, cx: 1.5, cy: 0.5, width: 4, height: 2))
    }

    func temp() -> URL { FileManager.default.temporaryDirectory.appendingPathComponent("oscd-\(UUID().uuidString)") }

    func testRoundTrip() throws {
        let root = temp()
        let store = try SessionStore(create: root, manifest: manifest())
        try store.setROI(AABB(min: SIMD3(-1, 0, -1), max: SIMD3(1, 0.5, 1)))
        let rec = FrameRecord(id: 0, timestamp: 1.5, arkitCameraToWorld: matrix_identity_float4x4.columnMajor)
        try store.append(.keyframes, record: rec, rgb: [UInt8](repeating: 255, count: 4 * 2 * 3), depth: [0.5, 0.6], confidence: [2, 1])
        let back = try SessionStore(open: root)
        XCTAssertEqual(back.manifest.keyframes, [rec])
        XCTAssertEqual(back.manifest.roi?.max.y, 0.5)
        XCTAssertEqual(try back.rgb(.keyframes, id: 0).data.max(), 1)
        XCTAssertEqual(try back.depth(.keyframes, id: 0).confidence, [2, 1])
    }

    /// Review Focus 5: files written for a frame the manifest never recorded are ignored, and a
    /// listed frame with a missing file throws a clear error instead of crashing.
    func testPartialWriteIsIgnoredAndMissingFileThrows() throws {
        let root = temp()
        let store = try SessionStore(create: root, manifest: manifest())
        let rec = FrameRecord(id: 0, timestamp: 0, arkitCameraToWorld: matrix_identity_float4x4.columnMajor)
        try store.append(.keyframes, record: rec, rgb: [UInt8](repeating: 0, count: 24), depth: [1, 1], confidence: [2, 2])
        try NPYArray(uint8: [UInt8](repeating: 0, count: 24), shape: [2, 4, 3])
            .write(to: root.appendingPathComponent("keyframes/000001_rgb.npy"))          // orphan, as after a crash
        let back = try SessionStore(open: root)
        XCTAssertEqual(back.manifest.keyframes.map(\.id), [0])
        try FileManager.default.removeItem(at: root.appendingPathComponent("keyframes/000000_depth.npy"))
        XCTAssertThrowsError(try back.depth(.keyframes, id: 0))
        XCTAssertThrowsError(try SessionStore(open: temp()))
    }

    func testKeyframeSelector() {
        var s = KeyframeSelector()
        func pose(x: Float) -> Pose { Pose(R: matrix_identity_float3x3, t: SIMD3(-x, 0, 0)) }   // centre at (x, 0, 0)
        XCTAssertTrue(s.consider(pose: pose(x: 0), time: 0, trackingNormal: true))
        XCTAssertFalse(s.consider(pose: pose(x: 0.02), time: 0.2, trackingNormal: true))      // too close
        XCTAssertTrue(s.consider(pose: pose(x: 0.06), time: 0.4, trackingNormal: true))       // 6 cm on
        XCTAssertFalse(s.consider(pose: pose(x: 0.30), time: 0.5, trackingNormal: true))      // 2.4 m/s: too fast
        XCTAssertFalse(s.consider(pose: pose(x: 0.40), time: 1.5, trackingNormal: false))     // tracking limited
    }

    func testCameraImageConversion() {
        // 4×4 luma, 2×2 interleaved CbCr, factor 2 → 2×2 RGB.
        let gray = CameraImage.rgbArea(y: [UInt8](repeating: 128, count: 16), yStride: 4, cbcr: [UInt8](repeating: 128, count: 8), cbcrStride: 4,
                                       width: 4, height: 4, factor: 2)
        XCTAssertEqual(gray, [UInt8](repeating: 128, count: 12))
        var cbcr = [UInt8](repeating: 0, count: 8)
        for i in stride(from: 0, to: 8, by: 2) { cbcr[i] = 85; cbcr[i + 1] = 255 }        // pure red in BT.601 full range
        let red = CameraImage.rgbArea(y: [UInt8](repeating: 76, count: 16), yStride: 4, cbcr: cbcr, cbcrStride: 4, width: 4, height: 4, factor: 2)
        XCTAssertGreaterThanOrEqual(red[0], 250); XCTAssertLessThanOrEqual(red[1], 3); XCTAssertLessThanOrEqual(red[2], 3)
    }

    /// Review fix: a capture size that is not an exact integer multiple of 480×360 must be rejected,
    /// not crash the recorder when the 480×360 RGB array is written.
    func testAreaFactorAcceptsOnlyExactMultiples() {
        XCTAssertEqual(CameraImage.areaFactor(captureWidth: 1920, captureHeight: 1440, workingWidth: 480, workingHeight: 360), 4)
        XCTAssertNil(CameraImage.areaFactor(captureWidth: 1920, captureHeight: 1080, workingWidth: 480, workingHeight: 360))
        XCTAssertNil(CameraImage.areaFactor(captureWidth: 1000, captureHeight: 750, workingWidth: 480, workingHeight: 360))
        XCTAssertNil(CameraImage.areaFactor(captureWidth: 240, captureHeight: 180, workingWidth: 480, workingHeight: 360))
    }
}
