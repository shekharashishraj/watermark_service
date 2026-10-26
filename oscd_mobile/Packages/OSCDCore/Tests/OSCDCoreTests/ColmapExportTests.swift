import XCTest
import simd
@testable import OSCDCore

final class ColmapExportTests: XCTestCase {
    /// ARKit camera-to-world (column-major, y-up camera looking down −z) for a camera at `c` turned by `yaw`.
    func arkitPose(yaw: Float, centre c: SIMD3<Float>) -> [Float] {
        let R = simd_float3x3(simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0)))
        let m = simd_float4x4(SIMD4(R.columns.0, 0), SIMD4(R.columns.1, 0), SIMD4(R.columns.2, 0), SIMD4(c, 1))
        return (0..<4).flatMap { col in (0..<4).map { m[col][$0] } }
    }

    func testPoseRoundTripsThroughTheColmapQuaternion() {
        let (R, t) = ColmapExport.worldToCamera(arkitCameraToWorld: arkitPose(yaw: 0.7, centre: SIMD3(0.1, 0.3, -0.2)))
        let q = ColmapExport.qvec(R)                                   // (w, x, y, z)
        let back = simd_double3x3(simd_quatd(ix: q.y, iy: q.z, iz: q.w, r: q.x))
        for c in 0..<3 { for r in 0..<3 { XCTAssertEqual(back[c][r], R[c][r], accuracy: 1e-6) } }
        XCTAssertEqual(simd_length(SIMD4(q.x, q.y, q.z, q.w)), 1, accuracy: 1e-6)   // ARKit poses are Float
        // The camera centre −Rᵀt is where ARKit put the camera.
        let centre = -(R.transpose * t)
        XCTAssertEqual(centre.x, 0.1, accuracy: 1e-6); XCTAssertEqual(centre.y, 0.3, accuracy: 1e-6); XCTAssertEqual(centre.z, -0.2, accuracy: 1e-6)
    }

    func testFlatDepthBackProjectsOntoThePlaneInFront() {
        // Camera at the origin looking down ARKit −z; a wall 0.5 m in front fills an 8×6 depth map.
        let K = Intrinsics(f: 40, cx: 15.5, cy: 11.5, width: 32, height: 24)
        let depth = [Float](repeating: 0.5, count: 8 * 6)
        var conf = [UInt8](repeating: 2, count: 8 * 6); conf[0] = 1           // one low-confidence pixel
        let rgb = [UInt8](repeating: 200, count: 32 * 24 * 3)
        let pts = ColmapExport.lidarPoints(depth: depth, confidence: conf, depthWidth: 8, depthHeight: 6, intrinsics: K,
                                           arkitCameraToWorld: arkitPose(yaw: 0, centre: .zero), rgb: rgb)
        XCTAssertEqual(pts.count, 47)
        for p in pts { XCTAssertEqual(p.xyz.z, -0.5, accuracy: 1e-9); XCTAssertEqual(p.rgb, SIMD3(200, 200, 200)) }
        // The depth map's centre pixel (cx, cy scaled by 8/32) is on the optical axis.
        let xs = pts.map(\.xyz.x), ys = pts.map(\.xyz.y)
        XCTAssertEqual((xs.min()! + xs.max()!) / 2, 0, accuracy: 1e-9)
        XCTAssertEqual((ys.min()! + ys.max()!) / 2, 0, accuracy: 1e-9)
    }

    func testRegionCropAndVoxelDedupKeepTheFirstPointPerVoxelInSortedVoxelOrder() {
        let p: [(SIMD3<Double>, UInt8)] = [
            (SIMD3(0.0101, 0, 0), 1), (SIMD3(0.0119, 0.001, 0), 2),   // same 4 mm voxel (2, 0, 0): keep 1
            (SIMD3(-0.003, 0, 0), 3),                                  // voxel (−1, 0, 0) sorts first
            (SIMD3(0.5, 0, 0), 4),                                     // outside the region
            (SIMD3(0.0101, 0.0041, 0), 5),                             // voxel (2, 1, 0)
        ]
        let pts = p.map { ColmapExport.Point(xyz: $0.0, rgb: SIMD3(repeating: $0.1)) }
        let roi = AABB(min: SIMD3(-0.1, -0.1, -0.1), max: SIMD3(0.1, 0.1, 0.1))
        XCTAssertEqual(ColmapExport.cropAndDedupe(pts, roi: roi, voxel: 0.004).map(\.rgb.x), [3, 1, 5])
    }

    func testWritesADatasetSplatxCanRead() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("colmap-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        var m = SessionManifest(deviceModel: "test", captureWidth: 128, captureHeight: 96, depthWidth: 8, depthHeight: 6,
                                intrinsics: Intrinsics(f: 40, cx: 15.5, cy: 11.5, width: 32, height: 24))
        m.roi = AABB(min: SIMD3(-1, -1, -1), max: SIMD3(1, 1, 1))
        let store = try SessionStore(create: dir.appendingPathComponent("scan"), manifest: m)
        for i in 0..<2 {
            try store.append(.keyframes, record: FrameRecord(id: i, timestamp: Double(i), arkitCameraToWorld: arkitPose(yaw: 0.1 * Float(i), centre: .zero)),
                             rgb: [UInt8](repeating: 128, count: 32 * 24 * 3), depth: [Float](repeating: 0.5, count: 48),
                             confidence: [UInt8](repeating: 2, count: 48))
        }
        let out = dir.appendingPathComponent("colmap")
        let r = try ColmapExport.write(session: store, to: out)
        XCTAssertEqual(r.images, 2)
        XCTAssertGreaterThan(r.points, 0)
        let sparse = out.appendingPathComponent("sparse/0")
        XCTAssertEqual(try String(contentsOf: sparse.appendingPathComponent("cameras.txt"), encoding: .utf8), "1 PINHOLE 32 24 40.0 40.0 15.5 11.5\n")
        let images = try String(contentsOf: sparse.appendingPathComponent("images.txt"), encoding: .utf8).split(separator: "\n")
        XCTAssertEqual(images.count, 2)
        XCTAssertTrue(images[0].hasPrefix("1 ") && images[0].hasSuffix(" 1 000000.png"))
        XCTAssertEqual(try String(contentsOf: sparse.appendingPathComponent("points3D.txt"), encoding: .utf8)
            .split(separator: "\n").count, r.points)
        XCTAssertTrue(FileManager.default.fileExists(atPath: out.appendingPathComponent("images/000001.png").path))
    }

    /// Full resolution: the keyframe's JPEG is the training image and the camera is scaled to the capture size;
    /// without full-size images the export falls back to the working-size PNGs.
    func testFullResolutionExportUsesTheJPEGsAndScalesTheCamera() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("full-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: dir) }
        let m = SessionManifest(deviceModel: "test", captureWidth: 128, captureHeight: 96, depthWidth: 8, depthHeight: 6,
                                intrinsics: Intrinsics(f: 40, cx: 15.5, cy: 11.5, width: 32, height: 24))
        let store = try SessionStore(create: dir.appendingPathComponent("scan"), manifest: m)
        try store.append(.keyframes, record: FrameRecord(id: 0, timestamp: 0, arkitCameraToWorld: arkitPose(yaw: 0, centre: .zero)),
                         rgb: [UInt8](repeating: 100, count: 32 * 24 * 3), depth: [Float](repeating: 1, count: 48),
                         confidence: [UInt8](repeating: 2, count: 48))
        XCTAssertFalse(store.hasFullImages)
        try ColmapExport.write(session: store, to: dir.appendingPathComponent("small"), fullResolution: true)
        XCTAssertTrue(fm.fileExists(atPath: dir.appendingPathComponent("small/images/000000.png").path))

        try ColmapExport.writeJPEG(rgb: [UInt8](repeating: 100, count: 128 * 96 * 3), width: 128, height: 96,
                                   to: store.fullImageURL(.keyframes, id: 0))
        XCTAssertTrue(store.hasFullImages)
        let out = dir.appendingPathComponent("full")
        try ColmapExport.write(session: store, to: out, fullResolution: true)
        XCTAssertTrue(fm.fileExists(atPath: out.appendingPathComponent("images/000000.jpg").path))
        let cameras = try String(contentsOf: out.appendingPathComponent("sparse/0/cameras.txt"), encoding: .utf8)
        XCTAssertEqual(cameras, "1 PINHOLE 128 96 160.0 160.0 63.5 47.5\n")
        let images = try String(contentsOf: out.appendingPathComponent("sparse/0/images.txt"), encoding: .utf8)
        XCTAssertTrue(images.contains(" 1 000000.jpg"))
    }
}
