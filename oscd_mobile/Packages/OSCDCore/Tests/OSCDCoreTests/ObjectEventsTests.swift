import simd
import XCTest
@testable import OSCDCore

final class ObjectEventsTests: XCTestCase {
    func config() -> TrackerConfig {
        var c = TrackerConfig()
        c.minPixels = 4
        return c
    }

    let pose = Pose(R: matrix_identity_float3x3, t: .zero)

    func inBlock(_ x: Int, _ y: Int, x0: Int, y0: Int, size: Int) -> Bool {
        x >= x0 && x < x0 + size && y >= y0 && y < y0 + size
    }

    /// A table at 0.50 m with a "left" block (live at table level, rendered closer) at x0 = 6 and an "arrived"
    /// block (live closer, rendered at table level) at `arrivedX0`; either can be left out.
    func frameWith(id: Int, left: Bool = true, arrived: Bool = true, arrivedX0: Int = 20, arrivedY0: Int = 9,
                   arrivedSize: Int = 6, flatX0: Int? = nil) -> FrameContext {
        let l = { (x: Int, y: Int) in left && self.inBlock(x, y, x0: 6, y0: 9, size: 6) }
        let a = { (x: Int, y: Int) in arrived && self.inBlock(x, y, x0: arrivedX0, y0: arrivedY0, size: arrivedSize) }
        let f = { (x: Int, y: Int) in flatX0.map { self.inBlock(x, y, x0: $0, y0: 9, size: 6) } ?? false }
        return frame(id: id, pose: pose,
                     dLive: { a($0, $1) ? 0.45 : 0.50 }, dRen: { l($0, $1) ? 0.45 : 0.50 },
                     changed: { l($0, $1) || a($0, $1) || f($0, $1) })
    }

    /// Three frames: one confirmed left (id 0) and one confirmed arrived (id 1).
    func pairTracker() -> ObjectTracker {
        let t = ObjectTracker(config: config())
        for i in 0..<3 { t.ingest(frameWith(id: i)) }
        XCTAssertEqual(t.instances.map(\.sign), [-1, 1])
        XCTAssertTrue(t.instances.allSatisfy { $0.isConfirmed(t.config) })
        return t
    }

    func unit(_ i: Int) -> [Float] {
        var v = [Float](repeating: 0, count: 256)
        v[i] = 1
        return v
    }

    // MARK: Hungarian and score

    func testHungarianPicksBestTotal() {
        XCTAssertEqual(Hungarian.assign(score: [[0.9, 0.8], [0.85, 0.1]]), [1, 0])
    }

    func testEmptySideGivesNoMatches() {
        XCTAssertEqual(Hungarian.assign(score: []), [])
        XCTAssertEqual(Hungarian.assign(score: [[]]), [nil])
        let tall = Hungarian.assign(score: [[0.2], [0.9], [0.5]])
        XCTAssertEqual(tall, [nil, 0, nil])
        XCTAssertEqual(Hungarian.assign(score: [[0.1, 0.9, 0.3]]), [1])
    }

    func testScoreFormula() {
        func inst(feature: [Float], bin: Int, voxels: Int) -> TrackedInstance {
            var i = TrackedInstance(id: 0, sign: 1)
            i.feature = feature
            i.histogram = [Float](repeating: 0, count: LabHistogram.bins)
            i.histogram[bin] = 1
            for k in 0..<voxels { i.voxels[VoxelKey(SIMD3(Float(k), 0, 0), size: 1)] = 1 }
            return i
        }
        let same = InstanceMatcher.score(inst(feature: unit(3), bin: 5, voxels: 20), inst(feature: unit(3), bin: 5, voxels: 20))
        XCTAssertEqual(same, 1, accuracy: 1e-4)
        let apart = InstanceMatcher.score(inst(feature: unit(3), bin: 5, voxels: 20), inst(feature: unit(4), bin: 6, voxels: 60))
        XCTAssertEqual(apart, 0, accuracy: 1e-4)
    }

    // MARK: Events

    func testMatchedPairBecomesMoved() {
        let t = pairTracker()
        t.setFeature(instance: 0, unit(2), source: .render)
        t.setFeature(instance: 1, unit(2), source: .live)
        let events = t.events()
        XCTAssertEqual(events.count, 1)
        let e = events[0]
        XCTAssertEqual(e.kind, .moved)
        XCTAssertEqual(e.id, 0)
        XCTAssertFalse(e.hint)
        XCTAssertGreaterThanOrEqual(e.confidence, 0.55)
        XCTAssertNotNil(e.before)
        XCTAssertNotNil(e.after)
        XCTAssertNil(e.motion)
        XCTAssertEqual(e.before?.frames, [0, 1, 2])
        XCTAssertLessThan(e.before!.centroid.x, e.after!.centroid.x)
        XCTAssertEqual(t.matches(), [Match(leftID: 0, arrivedID: 1, score: e.confidence)])
    }

    func testUnmatchedBecomeAddedAndRemoved() {
        let t = pairTracker()
        t.setFeature(instance: 0, unit(2), source: .render)
        t.setFeature(instance: 1, unit(2).map { -$0 }, source: .live)     // cosine −1: score 0
        let events = t.events()
        XCTAssertEqual(events.map(\.kind), [.removed, .added])
        XCTAssertEqual(events.map(\.id), [0, 1])
        XCTAssertNil(events[0].after); XCTAssertNotNil(events[0].before)
        XCTAssertNil(events[1].before); XCTAssertNotNil(events[1].after)
        XCTAssertEqual(events[0].confidence, 0.5, accuracy: 1e-6)         // 3 frames / 6
        XCTAssertTrue(t.matches().isEmpty)
    }

    func testLowScoreIsMovedWithLowConfidence() {
        let t = pairTracker()
        // Histograms are identical, so score = 0.5·cos + 0.3 + 0.2·size; pick the cosine that lands on 0.45.
        let size = InstanceMatcher.sizeSimilarity(t.instances[0].voxelCount, t.instances[1].voxelCount)
        let c = (0.45 - 0.3 - 0.2 * size) / 0.5
        var w = unit(2).map { $0 * c }
        w[3] = (1 - c * c).squareRoot()
        t.setFeature(instance: 0, unit(2), source: .render)
        t.setFeature(instance: 1, w, source: .live)
        XCTAssertEqual(InstanceMatcher.score(t.instances[0], t.instances[1]), 0.45, accuracy: 1e-4)
        let events = t.events()
        XCTAssertEqual(events.map(\.kind), [.moved])
        XCTAssertLessThan(events[0].confidence, 0.55)
        XCTAssertTrue(events[0].summary().hasPrefix("Object 0 · probably moved"))
    }

    func testMovedWithoutMotionIsStillMoved() {
        let t = pairTracker()
        t.setFeature(instance: 0, unit(2), source: .render)
        t.setFeature(instance: 1, unit(2), source: .live)
        let e = t.events()[0]
        XCTAssertNil(e.motion)
        XCTAssertEqual(e.summary(), "Object 0 · moved")
        XCTAssertFalse(e.summary().contains(" cm"))
    }

    func testBelowMatchLowIsNotAMatch() {
        let t = pairTracker()
        t.setFeature(instance: 0, unit(2), source: .render)
        t.setFeature(instance: 1, unit(2).map { -$0 * 0.5 }, source: .live)   // 0.5·(−1) + 0.3 + 0.2·size ≤ 0
        XCTAssertTrue(t.matches().isEmpty)
    }

    func testAlteredIsAHintOnly() {
        let t = ObjectTracker(config: config())
        for i in 0..<3 { t.ingest(frameWith(id: i, left: false, arrived: false, flatX0: 13)) }
        XCTAssertEqual(t.instances.map(\.sign), [0])
        let events = t.events()
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events[0].kind, .altered)
        XCTAssertTrue(events[0].hint)
        XCTAssertEqual(events[0].confidence, 0)
        XCTAssertTrue(t.matches().isEmpty)
        XCTAssertEqual(events[0].summary(), "Object 0 · look changed here?")
    }

    func testUnconfirmedInstancesProduceNoEvents() {
        let t = ObjectTracker(config: config())
        t.ingest(frameWith(id: 0))
        XCTAssertTrue(t.events().isEmpty)
    }

    func testEventsAreOrderedByFirstSeen() {
        let t = ObjectTracker(config: config())
        for i in 0..<3 { t.ingest(frameWith(id: 10 + i, left: false)) }     // arrived first seen at frame 10
        for i in 0..<3 { t.ingest(frameWith(id: i, left: true, arrived: false)) }  // left first seen at frame 0
        XCTAssertEqual(t.instances.map(\.sign), [1, -1])
        t.setFeature(instance: 0, unit(2), source: .live)
        t.setFeature(instance: 1, unit(2).map { -$0 }, source: .render)
        let events = t.events()
        XCTAssertEqual(events.map(\.kind), [.removed, .added])
        XCTAssertEqual(events.map(\.id), [1, 0])
    }

    func testSummaryStrings() {
        let motion = Motion(translation: SIMD3(0.14, 0, 0), horizontalDistance: 0.14, yawDeg: 30, fitRMS: 0.002, inlierFraction: 0.9)
        var e = ObjectEvent(id: 4, kind: .moved, confidence: 0.8, hint: false, label: "Ball", labelConfidence: 0.9, motion: motion)
        XCTAssertEqual(e.summary(), "Ball · moved 14 cm · turned 30°")
        e.label = "Cup"; e.labelConfidence = 0.4; e.motion?.yawDeg = nil
        XCTAssertEqual(e.summary(), "Cup? · moved 14 cm · rotation unknown")
        e.confidence = 0.5
        XCTAssertEqual(e.summary(), "Cup? · probably moved 14 cm · rotation unknown")
        XCTAssertEqual(e.summary(matchAccept: 0.4), "Cup? · moved 14 cm · rotation unknown")
        let r = ObjectEvent(id: 2, kind: .removed, confidence: 1, hint: false, label: "Glasses", labelConfidence: 0.8)
        XCTAssertEqual(r.summary(), "Glasses · removed")
        XCTAssertEqual(ObjectEvent(id: 3, kind: .added, confidence: 1, hint: false).summary(), "Object 3 · added")
    }

    // MARK: Report

    func tempFolder() throws -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("oscd-events-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    func testReportRoundTripsThroughJSONAndFolder() throws {
        let t = pairTracker()
        t.setFeature(instance: 0, unit(2), source: .render)
        t.setFeature(instance: 1, unit(2), source: .live)
        t.setMotion(leftID: 0, arrivedID: 1, Motion(translation: SIMD3(0.14, 0, 0.01), horizontalDistance: 0.14, yawDeg: 30,
                                                    fitRMS: 0.002, inlierFraction: 0.9))
        let report = t.report(beforeID: "before-1", afterID: "after-1", method: "B", timingsMs: ["icp": 12.5])
        XCTAssertEqual(report.worldFrame, "before-scan ARKit world, +y up")
        XCTAssertEqual(report.voxelSizeM, 0.01)
        XCTAssertEqual(report.events.first?.summary(), "Object 0 · moved 14 cm · turned 30°")

        let after = try tempFolder()
        try report.save(in: after)
        XCTAssertTrue(FileManager.default.fileExists(atPath: DetectionSummary.folder(in: after, beforeID: "before-1", method: "B")
            .appendingPathComponent("objects.json").path))
        let loaded = try XCTUnwrap(ObjectReport.load(in: after, beforeID: "before-1", method: "B"))
        XCTAssertEqual(loaded.beforeID, "before-1")
        XCTAssertEqual(loaded.afterID, "after-1")
        XCTAssertEqual(loaded.method, "B")
        XCTAssertEqual(loaded.timingsMs, ["icp": 12.5])
        XCTAssertEqual(try SessionStore.encoder.encode(loaded.events), try SessionStore.encoder.encode(report.events))
        XCTAssertEqual(loaded.events[0].motion?.yawDeg, 30)
        XCTAssertEqual(loaded.events[0].before?.bbox, report.events[0].before?.bbox)

        XCTAssertNil(ObjectReport.load(in: try tempFolder(), beforeID: "before-1", method: "B"))
        let bad = try tempFolder()
        let dir = DetectionSummary.folder(in: bad, beforeID: "before-1", method: "B")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("not json".utf8).write(to: dir.appendingPathComponent("objects.json"))
        XCTAssertNil(ObjectReport.load(in: bad, beforeID: "before-1", method: "B"))
    }

    // MARK: Motion cache

    func testMotionCacheClearsWhenInstanceGrows() {
        let t = pairTracker()
        t.setFeature(instance: 0, unit(2), source: .render)
        t.setFeature(instance: 1, unit(2), source: .live)
        XCTAssertEqual(t.pairsNeedingMotion().count, 1)
        let m = Motion(translation: SIMD3(0.1, 0, 0), horizontalDistance: 0.1, yawDeg: nil, fitRMS: 0.001, inlierFraction: 1)
        t.setMotion(leftID: 0, arrivedID: 1, m)
        XCTAssertTrue(t.pairsNeedingMotion().isEmpty)
        XCTAssertEqual(t.events()[0].motion?.horizontalDistance, 0.1)
        // A failed measurement is cached too.
        t.setMotion(leftID: 0, arrivedID: 1, nil)
        XCTAssertTrue(t.pairsNeedingMotion().isEmpty)
        XCTAssertNil(t.events()[0].motion)

        let before = t.instances[1].voxelCount
        t.ingest(frameWith(id: 3, arrivedX0: 18, arrivedY0: 7, arrivedSize: 10))
        XCTAssertGreaterThan(Float(t.instances[1].voxelCount), 1.2 * Float(before))
        XCTAssertEqual(t.pairsNeedingMotion().count, 1)
        XCTAssertNil(t.events()[0].motion)
    }
}
