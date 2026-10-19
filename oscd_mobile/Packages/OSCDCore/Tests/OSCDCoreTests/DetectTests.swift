import XCTest
@testable import OSCDCore

final class StageTimingsTests: XCTestCase {
    func testMediansTotalsAndOrderOfFirstUse() {
        var t = StageTimings()
        t.add("XFeat", 9); t.add("Pose", 40); t.add("XFeat", 11); t.add("XFeat", 30); t.add("Pose", 20)
        XCTAssertEqual(t.stages, ["XFeat", "Pose"])
        XCTAssertEqual(t.median("XFeat"), 11)
        XCTAssertEqual(t.median("Pose"), 30)          // even count: mean of the middle two
        XCTAssertEqual(t.total("XFeat"), 50)
        XCTAssertEqual(t.count("Pose"), 2)
        XCTAssertNil(t.median("SAM2"))
    }

    func testMergeAppendsSamplesAndKeepsFirstUseOrder() {
        var a = StageTimings(); a.add("XFeat", 9); a.add("Fusion", 300)
        var b = StageTimings(); b.add("Fusion", 280); b.add("Mask", 7)
        a.merge(b)
        XCTAssertEqual(a.stages, ["XFeat", "Fusion", "Mask"])
        XCTAssertEqual(a.count("Fusion"), 2)
        XCTAssertEqual(a.median("Fusion"), 290)
        XCTAssertEqual(a.total("Mask"), 7)
    }

    func testRoundTripsThroughJSON() throws {
        var t = StageTimings()
        t.add("Fusion", 676.5); t.add("SAM2", 191)
        let back = try JSONDecoder().decode(StageTimings.self, from: JSONEncoder().encode(t))
        XCTAssertEqual(back, t)
        XCTAssertEqual(back.stages, ["Fusion", "SAM2"])
    }
}

final class ScanCatalogTests: XCTestCase {
    var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("catalog-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: dir) }

    func makeScan(_ folder: String, at date: Date, frames: Int, name: String? = nil) throws -> SessionStore {
        var m = SessionManifest(deviceModel: "iPhone18,1", captureWidth: 1920, captureHeight: 1440, depthWidth: 256, depthHeight: 192,
                                intrinsics: Intrinsics(f: 360, cx: 240, cy: 180, width: 480, height: 360), createdAt: date)
        m.name = name
        m.keyframes = (0..<frames).map { FrameRecord(id: $0, timestamp: Double($0), arkitCameraToWorld: [Float](repeating: 0, count: 16)) }
        return try SessionStore(create: dir.appendingPathComponent(folder), manifest: m)
    }

    func testScansAreNumberedOldestFirstAndKnowTheirModelAndResults() throws {
        let t0 = Date(timeIntervalSince1970: 1_000_000)
        _ = try makeScan("b", at: t0.addingTimeInterval(60), frames: 45)
        let first = try makeScan("a", at: t0, frames: 47)
        _ = try makeScan("c", at: t0.addingTimeInterval(120), frames: 40, name: "Moved ball")
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("not-a-scan"), withIntermediateDirectories: true)
        try Data("ply".utf8).write(to: first.root.appendingPathComponent(ScanCatalog.referenceFile))

        var summary = DetectionSummary(beforeID: "a", afterID: "b", method: "A", refined: false, frames: 45, framesWithPose: 31,
                                       judgedPixels: 1000, changedPixels: 140, totalSeconds: 58, timings: StageTimings())
        summary.finishedAt = t0.addingTimeInterval(200)
        try summary.save(in: dir.appendingPathComponent("b"))

        let scans = try ScanCatalog.list(dir)
        XCTAssertEqual(scans.map(\.id), ["a", "b", "c"])
        XCTAssertEqual(scans.map(\.label), ["Scan 1", "Scan 2", "Moved ball"])
        XCTAssertEqual(scans.map(\.frameCount), [47, 45, 40])
        XCTAssertEqual(scans.map(\.hasReference), [true, false, false])
        XCTAssertEqual(scans[1].latestDetection?.beforeID, "a")
        XCTAssertEqual(scans[1].latestDetection?.changedFraction ?? 0, 0.14, accuracy: 1e-9)
        XCTAssertNil(scans[0].latestDetection)
    }

    func testDetectionSummaryKeepsOnlyTheLatestRunPerBeforeScan() throws {
        let after = dir.appendingPathComponent("b")
        var s1 = DetectionSummary(beforeID: "a", afterID: "b", method: "A", refined: false, frames: 10, framesWithPose: 5,
                                  judgedPixels: 10, changedPixels: 1, totalSeconds: 1, timings: StageTimings())
        try s1.save(in: after)
        s1.framesWithPose = 7
        try s1.save(in: after)
        let all = try DetectionSummary.all(in: after)
        XCTAssertEqual(all.count, 1)
        XCTAssertEqual(all[0].framesWithPose, 7)
        XCTAssertEqual(DetectionSummary.folder(in: after, beforeID: "a").lastPathComponent, "a")
    }
}
