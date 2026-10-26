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

    // MARK: Matching with an "unmatched" option (R36)

    /// L1→A1 0.72, L1→A2 0.50, L2→A1 0.60, L2→A2 0.30: plain assignment takes the cross pairs (1.10); with the
    /// opt-out L1→A1 + L2 unmatched (0.72 + 0.40) wins, so L2 is removed and A2 added.
    func testAssignDoesNotForceAPairToRaiseTheTotal() {
        let score: [[Float]] = [[0.72, 0.50], [0.60, 0.30]]
        XCTAssertEqual(Hungarian.assign(score: score), [1, 0])               // the temptation the opt-out removes
        XCTAssertEqual(InstanceMatcher.assign(score: score, low: 0.40), [0, nil])
    }

    func testAssignKeepsAPairExactlyAtLowAndDropsOneJustBelow() {
        XCTAssertEqual(InstanceMatcher.assign(score: [[0.40]], low: 0.40), [0])
        XCTAssertEqual(InstanceMatcher.assign(score: [[0.3999]], low: 0.40), [nil])
        XCTAssertEqual(InstanceMatcher.assign(score: [[0.40, 0.10], [0.10, 0.3999]], low: 0.40), [0, nil])
    }

    /// Three left × three arrived where pairing L1 with its second choice frees A1 for L2: the total rises (1.95 against
    /// 1.80) but is beaten by leaving L2 unmatched (2.00).
    func testAssignThreeByThreeResistsAForcedPair() {
        let score: [[Float]] = [[0.90, 0.45, 0.10], [0.80, 0.20, 0.10], [0.10, 0.10, 0.70]]
        XCTAssertEqual(Hungarian.assign(score: score), [1, 0, 2])
        XCTAssertEqual(InstanceMatcher.assign(score: score, low: 0.40), [0, nil, 2])
    }

    func testAssignRectangularAndNonFinite() {
        XCTAssertEqual(InstanceMatcher.assign(score: [], low: 0.4), [])
        XCTAssertEqual(InstanceMatcher.assign(score: [[]], low: 0.4), [nil])
        XCTAssertEqual(InstanceMatcher.assign(score: [[0.2], [0.9], [0.5]], low: 0.4), [nil, 0, nil])
        XCTAssertEqual(InstanceMatcher.assign(score: [[0.1, 0.9, 0.3]], low: 0.4), [1])
        XCTAssertEqual(InstanceMatcher.assign(score: [[.nan, 0.5], [0.9, .infinity]], low: 0.4), [1, 0])
        XCTAssertEqual(InstanceMatcher.assign(score: [[.nan]], low: 0.4), [nil])
        XCTAssertEqual(InstanceMatcher.assign(score: [[-.infinity, .nan]], low: -1), [nil])
    }

    /// The 2 × 2 case end to end: two left and two arrived blocks with features set so the scores tempt a cross
    /// pairing; the events are one move (L1 → A1), one removal (L2) and one addition (A2).
    func testTrackerReportsRemovedAndAddedRatherThanAForcedMove() throws {
        let t = ObjectTracker(config: config())
        func block(_ x: Int, _ y: Int, _ x0: Int, _ y0: Int) -> Bool { inBlock(x, y, x0: x0, y0: y0, size: 6) }
        for i in 0..<3 {
            let l = { (x: Int, y: Int) in block(x, y, 2, 2) || block(x, y, 2, 14) }
            let a = { (x: Int, y: Int) in block(x, y, 20, 2) || block(x, y, 20, 14) }
            t.ingest(frame(id: i, pose: pose, dLive: { a($0, $1) ? 0.45 : 0.50 }, dRen: { l($0, $1) ? 0.45 : 0.50 },
                           changed: { l($0, $1) || a($0, $1) }))
        }
        XCTAssertTrue(t.instances.allSatisfy { $0.isConfirmed(t.config) })
        // Image y grows downwards and the camera looks along +z with +y down, so the top blocks have the smaller world y.
        let lefts = t.instances.filter { $0.sign < 0 }.sorted { $0.centroid.y < $1.centroid.y }
        let arrived = t.instances.filter { $0.sign > 0 }.sorted { $0.centroid.y < $1.centroid.y }
        XCTAssertEqual(lefts.count, 2); XCTAssertEqual(arrived.count, 2)
        let (l1, l2, a1, a2) = (lefts[0].id, lefts[1].id, arrived[0].id, arrived[1].id)
        // Unit features: cos(L1, A1) 0.44, cos(L1, A2) 0, cos(L2, A1) 0.2, cos(L2, A2) −0.4. With equal colour and
        // size the scores are 0.5 + 0.5·cos: 0.72, 0.50, 0.60, 0.30.
        var fa1 = unit(0).map { $0 * 0.44 }; fa1[1] = 0.2; fa1[2] = (1 - 0.44 * 0.44 - 0.04).squareRoot()
        var fa2 = unit(1).map { $0 * -0.4 }; fa2[3] = (1 - 0.16).squareRoot()
        t.setFeature(instance: l1, unit(0), source: .render)
        t.setFeature(instance: l2, unit(1), source: .render)
        t.setFeature(instance: a1, fa1, source: .live)
        t.setFeature(instance: a2, fa2, source: .live)
        let byID = Dictionary(uniqueKeysWithValues: t.instances.map { ($0.id, $0) })
        let score = [l1, l2].map { l in [a1, a2].map { InstanceMatcher.score(byID[l]!, byID[$0]!) } }
        XCTAssertEqual(score[0][0], 0.72, accuracy: 0.02); XCTAssertEqual(score[0][1], 0.50, accuracy: 0.02)
        XCTAssertEqual(score[1][0], 0.60, accuracy: 0.02); XCTAssertEqual(score[1][1], 0.30, accuracy: 0.02)
        XCTAssertEqual(Hungarian.assign(score: score), [1, 0])

        XCTAssertEqual(t.matches().map { [$0.leftID, $0.arrivedID] }, [[l1, a1]])
        let events = t.events()
        XCTAssertEqual(events.filter { $0.kind == .moved }.map(\.id), [l1])
        XCTAssertEqual(events.filter { $0.kind == .removed }.map(\.id), [l2])
        XCTAssertEqual(events.filter { $0.kind == .added }.map(\.id), [a2])
        XCTAssertEqual(events.count, 3)
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

    func testFitMotionsMeasuresEachPairOnceWithTheSharedFit() throws {
        let t = pairTracker()
        t.setFeature(instance: 0, unit(2), source: .render)
        t.setFeature(instance: 1, unit(2), source: .live)
        var seen: [Match] = []
        XCTAssertEqual(t.fitMotions(stop: { true }), 0)
        XCTAssertEqual(t.fitMotions(willFit: { seen.append($0) }), 1)
        XCTAssertEqual(seen.map(\.leftID), [0]); XCTAssertEqual(seen.map(\.arrivedID), [1])
        XCTAssertTrue(t.pairsNeedingMotion().isEmpty)
        XCTAssertEqual(t.fitMotions(), 0)
        // The cached move is exactly what the shared fit gives on the pair's points.
        let input = try XCTUnwrap(t.motionInputs(seen[0]))
        let expected = ObjectTracker.measureMotion(source: input.source, target: input.target)
        XCTAssertEqual(t.events()[0].motion?.translation, expected?.translation)
    }

    /// An asynchronous fit lands after the arrived instance grew by more than 20 %: with the counts captured when its
    /// points were copied the move is stale at once; read at landing, the growth would be missed.
    func testCountsCapturedAtCopyTimeDriveInvalidation() throws {
        let t = pairTracker()
        t.setFeature(instance: 0, unit(2), source: .render)
        t.setFeature(instance: 1, unit(2), source: .live)
        let pair = try XCTUnwrap(t.pairsNeedingMotion().first)
        let input = try XCTUnwrap(t.motionInputs(pair))
        XCTAssertEqual(input.leftVoxels, t.instances[0].voxelCount)
        XCTAssertEqual(input.arrivedVoxels, t.instances[1].voxelCount)
        // While the fit runs, the arrived block grows.
        t.ingest(frameWith(id: 3, arrivedX0: 18, arrivedY0: 7, arrivedSize: 10))
        XCTAssertGreaterThan(Float(t.instances[1].voxelCount), 1.2 * Float(input.arrivedVoxels))
        let m = Motion(translation: SIMD3(0.1, 0, 0), horizontalDistance: 0.1, yawDeg: nil, fitRMS: 0.001, inlierFraction: 1)
        t.setMotion(leftID: 0, arrivedID: 1, m, leftVoxels: input.leftVoxels, arrivedVoxels: input.arrivedVoxels)
        XCTAssertEqual(t.pairsNeedingMotion().count, 1)
        XCTAssertNil(t.events()[0].motion)
        // The landing-time counts (the old behaviour) would have kept the stale move.
        t.setMotion(leftID: 0, arrivedID: 1, m)
        XCTAssertTrue(t.pairsNeedingMotion().isEmpty)
        XCTAssertEqual(t.events()[0].motion?.horizontalDistance, 0.1)
    }
}
