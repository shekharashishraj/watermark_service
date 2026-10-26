import XCTest
import simd
@testable import OSCDCore

final class FrameContextIOTests: XCTestCase {
    private func tempDir() throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("framectx-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: d) }
        return d
    }

    private func pose(_ x: Float) -> Pose {
        let a: Float = 0.3
        return Pose(R: simd_float3x3(columns: (SIMD3(cos(a), 0, -sin(a)), SIMD3(0, 1, 0), SIMD3(sin(a), 0, cos(a)))),
                    t: SIMD3(x, -0.25, 0.75))
    }

    private func sample(id: Int) -> FrameContext {
        let feature = (0..<256).map { Float($0 % 7) * 0.1234 - 0.3 }
        var f = frame(id: id, pose: pose(Float(id) * 0.1),
                      dLive: { x, y in (x + y) % 5 == 0 ? .nan : 0.4 + Float(x) * 0.003 },
                      dRen: { x, y in x == 3 ? .nan : 0.5 - Float(y) * 0.002 },
                      changed: { x, y in (x * y) % 3 == 0 }, feature: feature)
        // Distinct colours so a channel mix-up shows.
        for i in f.live.data.indices { f.live.data[i] = Float(i % 251) / 251 }
        for i in f.render.data.indices { f.render.data[i] = Float((i * 7) % 253) / 253 }
        f.valid = f.valid.enumerated().map { $0.offset % 11 != 0 }
        return f
    }

    func testRoundTripPreservesEverything() throws {
        let dir = try tempDir()
        let f = sample(id: 42)
        try FrameContextIO.write(f, to: dir)
        let folder = dir.appendingPathComponent("000042")
        for name in ["changed", "valid", "dLive", "dRen", "live", "render", "embLive", "embRender"] {
            XCTAssertTrue(FileManager.default.fileExists(atPath: folder.appendingPathComponent(name + ".npy").path), name)
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: folder.appendingPathComponent("pose.json").path))

        let g = try FrameContextIO.read(frameDir: folder)
        XCTAssertEqual(g.frameID, 42)
        XCTAssertEqual(g.pose, f.pose)
        XCTAssertEqual(g.intrinsics, f.intrinsics)
        XCTAssertEqual(g.changed, f.changed)
        XCTAssertEqual(g.valid, f.valid)
        XCTAssertEqual(g.dLive.map(\.bitPattern), f.dLive.map(\.bitPattern))     // NaNs included
        XCTAssertEqual(g.dRen.map(\.bitPattern), f.dRen.map(\.bitPattern))
        XCTAssertEqual(g.live.data, f.live.data)
        XCTAssertEqual(g.render.data, f.render.data)
        XCTAssertEqual([g.live.width, g.live.height, g.live.channels], [32, 24, 3])
        XCTAssertEqual([g.render.width, g.render.height, g.render.channels], [32, 24, 3])
        XCTAssertEqual(g.embLive.count, 256 * 64 * 64)
        for (a, b) in zip(f.embLive, g.embLive) { XCTAssertLessThanOrEqual(abs(a - b), 1e-3 * abs(a) + 1e-7) }
        for (a, b) in zip(f.embRender, g.embRender) { XCTAssertLessThanOrEqual(abs(a - b), 1e-3 * abs(a) + 1e-7) }
    }

    func testBeforeDepthRoundTripsWhenPresentAndReadsNilWhenAbsent() throws {
        let dir = try tempDir()
        XCTAssertNil(try FrameContextIO.read(frameDir: { try FrameContextIO.write(sample(id: 1), to: dir); return dir.appendingPathComponent("000001") }()).dBefore)
        var f = sample(id: 2)
        f.dBefore = f.dRen.enumerated().map { $0.offset % 4 == 0 ? .nan : $0.element + 0.01 }
        try FrameContextIO.write(f, to: dir)
        let g = try FrameContextIO.read(frameDir: dir.appendingPathComponent("000002"))
        XCTAssertEqual(try XCTUnwrap(g.dBefore).map(\.bitPattern), f.dBefore!.map(\.bitPattern))
    }

    func testPoseJSONHoldsViewmatAndTrackerSettings() throws {
        let dir = try tempDir()
        let f = sample(id: 3)
        let roi = AABB(min: SIMD3(-1, -0.5, -1), max: SIMD3(1, 0.5, 1))
        try FrameContextIO.write(f, to: dir, supportHeight: -0.123, roi: roi)
        let folder = dir.appendingPathComponent("000003")
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("pose.json"))) as! [String: Any]
        XCTAssertEqual(json["frameID"] as? Int, 3)
        XCTAssertEqual(json["keyframeID"] as? Int, 3)
        let m = (json["viewmat"] as! [Double]).map(Float.init)
        XCTAssertEqual(m.count, 16)
        // Row-major world→camera: rotation in the top-left 3×3, translation in the last column, 0 0 0 1 below.
        XCTAssertEqual(m[3], f.pose.t.x); XCTAssertEqual(m[7], f.pose.t.y); XCTAssertEqual(m[11], f.pose.t.z)
        XCTAssertEqual(m[1], f.pose.R.columns.1.x); XCTAssertEqual(Array(m[12...]), [0, 0, 0, 1])
        XCTAssertNotNil(json["intrinsics"])
        XCTAssertEqual((json["supportHeight"] as? NSNumber)?.floatValue, -0.123)
        let meta = try FrameContextIO.readMeta(frameDir: folder)
        XCTAssertEqual(meta.supportHeight, -0.123)
        XCTAssertEqual(meta.roi, roi)
        XCTAssertEqual(meta.keyframeID, 3)
    }

    func testMissingSupportAndROIReadAsNil() throws {
        let dir = try tempDir()
        try FrameContextIO.write(sample(id: 1), to: dir)
        let meta = try FrameContextIO.readMeta(frameDir: dir.appendingPathComponent("000001"))
        XCTAssertNil(meta.supportHeight)
        XCTAssertNil(meta.roi)
    }

    func testReadAllSortsByFrameID() throws {
        let dir = try tempDir()
        for id in [7, 2, 10] { try FrameContextIO.write(sample(id: id), to: dir) }
        XCTAssertEqual(try FrameContextIO.readAll(in: dir).map(\.frameID), [2, 7, 10])
    }
}
