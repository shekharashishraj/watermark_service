import simd
import XCTest
@testable import OSCDCore

/// The object layer end to end on a ray-cast table scene (spec §7.1): ingest → match → planar ICP → events.
final class ObjectTrackerSceneTests: XCTestCase {
    typealias Scene = SyntheticChangeScene

    let roi = AABB(min: SIMD3(repeating: -0.4), max: SIMD3(repeating: 0.4))

    // Object centres (world, +y up, table at y = 0).
    let aBefore = SIMD3<Float>(0, 0.033, 0), aAfter = SIMD3<Float>(0.20, 0.033, 0)
    let bCentre = SIMD3<Float>(0.15, 0.03, 0.1)
    let cCentre = SIMD3<Float>(-0.15, 0.03, -0.1)
    let dCentre = SIMD3<Float>(-0.15, 0.03, 0.15)
    let cubeHalf = SIMD3<Float>(repeating: 0.03)

    /// Every object, and the table, has its own block of SAM2 channels (cosine 0 between different objects).
    static func feature(block: Int) -> [Float] {
        var f = [Float](repeating: 0, count: 256)
        for c in (block * 32)..<(block * 32 + 32) { f[c] = 1 }
        return f
    }

    func sphereA(at c: SIMD3<Float>) -> Scene.Object {
        Scene.Object(shape: .sphere(centre: c, radius: 0.033), feature: Self.feature(block: 0), colour: SIMD3(0.80, 0.15, 0.15))
    }
    func cubeB(yaw: Float) -> Scene.Object {
        Scene.Object(shape: .box(centre: bCentre, half: cubeHalf, yawDeg: yaw), feature: Self.feature(block: 1), colour: SIMD3(0.15, 0.65, 0.20))
    }
    var cubeC: Scene.Object {
        Scene.Object(shape: .box(centre: cCentre, half: cubeHalf, yawDeg: 0), feature: Self.feature(block: 2), colour: SIMD3(0.15, 0.25, 0.85))
    }
    var cubeD: Scene.Object {
        Scene.Object(shape: .box(centre: dCentre, half: cubeHalf, yawDeg: 0), feature: Self.feature(block: 3), colour: SIMD3(0.90, 0.80, 0.10))
    }

    var changedScene: Scene {
        Scene(before: [sphereA(at: aBefore), cubeB(yaw: 0), cubeC],
              after: [sphereA(at: aAfter), cubeB(yaw: 40), cubeD])
    }

    func tracker(supportHeight: Float? = nil) -> ObjectTracker {
        var c = TrackerConfig()
        c.roi = roi
        c.supportHeight = supportHeight
        return ObjectTracker(config: c)
    }

    /// Table patches away from every object whose live depth reads 12–50 mm too far (B's sheet false positives).
    let sheetNoise: [Scene.SheetPatch] = [
        .init(centre: SIMD2(0.05, -0.22), radius: 0.06, offset: 0.03),
        .init(centre: SIMD2(-0.28, 0.0), radius: 0.05, offset: 0.05),
        .init(centre: SIMD2(0.25, -0.2), radius: 0.05, offset: 0.012),
        .init(centre: SIMD2(0.05, 0.28), radius: 0.06, offset: 0.02),
    ]

    /// Ingests the 10 orbit frames, then measures every matched pair with the planar ICP.
    @discardableResult
    func run(_ scene: Scene, into t: ObjectTracker) -> [IngestStats] {
        let stats = Scene.orbitPoses(count: 10, radius: 0.5, height: 0.45).enumerated().map { i, pose in
            t.ingest(scene.frame(id: i, pose: pose, roi: roi))
        }
        for m in t.pairsNeedingMotion() {
            guard let l = t.instances.first(where: { $0.id == m.leftID }),
                  let a = t.instances.first(where: { $0.id == m.arrivedID }) else { continue }
            t.setMotion(leftID: m.leftID, arrivedID: m.arrivedID, PlanarICP.fit(source: l.points, target: a.points))
        }
        return stats
    }

    static func describe(_ e: ObjectEvent) -> String {
        func p(_ v: SIMD3<Float>?) -> String { v.map { String(format: "(%.3f, %.3f, %.3f)", $0.x, $0.y, $0.z) } ?? "-" }
        var s = "\(e.kind.rawValue) id \(e.id) conf \(String(format: "%.2f", e.confidence)) before \(p(e.before?.centroid)) "
            + "[\(e.before?.voxels ?? 0) vox] after \(p(e.after?.centroid)) [\(e.after?.voxels ?? 0) vox]"
        if let m = e.motion {
            s += String(format: " dist %.4f yaw %@ rms %.4f", m.horizontalDistance, m.yawDeg.map { String(format: "%.1f", $0) } ?? "nil", m.fitRMS)
        }
        return s
    }

    func near(_ p: Placement?, _ c: SIMD3<Float>, within r: Float) -> Bool {
        guard let p else { return false }
        return simd_distance(p.centroid, c) < r
    }

    // MARK: - Tests

    func testOrbitPosesLookAtTheOriginFromAbove() {
        let K = Scene.intrinsics
        let poses = Scene.orbitPoses(count: 10, radius: 0.5, height: 0.45)
        XCTAssertEqual(poses.count, 10)
        for pose in poses {
            let c = pose.centre
            XCTAssertEqual(simd_length(SIMD2(c.x, c.z)), 0.5, accuracy: 1e-4)
            XCTAssertEqual(c.y, 0.45, accuracy: 1e-4)
            let o = pose.apply(.zero)
            XCTAssertGreaterThan(o.z, 0)
            let uv = K.project(o)
            XCTAssertEqual(uv.x, K.cx, accuracy: 0.5)
            XCTAssertEqual(uv.y, K.cy, accuracy: 0.5)
            // World up is image up: a point above the origin lands above the centre (smaller v).
            XCTAssertLessThan(K.project(pose.apply(SIMD3(0, 0.05, 0))).y, K.cy - 1)
        }
    }

    func testRayCastDepthsMatchTheScene() {
        // One frame straight down the camera axis onto the origin: the table, then a sphere sitting on it.
        let pose = Scene.orbitPoses(count: 10, radius: 0.5, height: 0.45)[0]
        let K = Scene.intrinsics
        let centre = Int(K.cy.rounded()) * K.width + Int(K.cx.rounded())
        let tableOnly = Scene(before: [], after: []).frame(id: 0, pose: pose, roi: roi)
        let range = simd_length(pose.centre)                     // camera to origin, along the optical axis
        XCTAssertEqual(tableOnly.dRen[centre], range, accuracy: 0.01)
        XCTAssertFalse(tableOnly.changed.contains(true))

        let withSphere = Scene(before: [], after: [sphereA(at: aBefore)]).frame(id: 0, pose: pose, roi: roi)
        XCTAssertLessThan(withSphere.dLive[centre], range - 0.03)
        XCTAssertTrue(withSphere.changed[centre])
        XCTAssertEqual(withSphere.embLive.count, 256 * 64 * 64)
    }

    func testThreeObjectsGiveThreeEvents() {
        assertThreeEvents(changedScene, tracker())
    }

    /// With B's sheet-level noise in the after scene and the support height known, the same three events and nothing else.
    func testThreeObjectsGiveThreeEventsDespiteSheetNoise() {
        var scene = changedScene
        scene.sheetNoise = sheetNoise
        assertThreeEvents(scene, tracker(supportHeight: scene.tableY))
    }

    func testSheetNoiseAloneGivesNoEvents() {
        let objects = [sphereA(at: aBefore), cubeB(yaw: 0), cubeC]
        var scene = Scene(before: objects, after: objects)
        scene.sheetNoise = sheetNoise
        let t = tracker(supportHeight: scene.tableY)
        run(scene, into: t)
        XCTAssertEqual(t.events().filter { !$0.hint }.count, 0, t.events().map(Self.describe).joined(separator: "\n"))

        // Not vacuous: without the support height the same noise becomes confirmed events.
        let ungated = tracker()
        run(scene, into: ungated)
        XCTAssertGreaterThan(ungated.events().filter { !$0.hint }.count, 0)
    }

    func assertThreeEvents(_ scene: Scene, _ t: ObjectTracker, file: StaticString = #filePath, line: UInt = #line) {
        run(scene, into: t)
        let events = t.events().filter { !$0.hint }
        let log = events.map(Self.describe).joined(separator: "\n")
        print("Synthetic scene events:\n" + log)

        // Events in B's region (turned in place) are left out of the counts, but may not link to A, C or D.
        let others = [aBefore, aAfter, cCentre, dCentre]
        let atB = events.filter { near($0.before, bCentre, within: 0.06) || near($0.after, bCentre, within: 0.06) }
        for e in atB where e.kind == .moved {
            for o in others {
                XCTAssertFalse(near(e.before, o, within: 0.06) || near(e.after, o, within: 0.06),
                               "B's region linked to another object: \(Self.describe(e))")
            }
        }
        let counted = events.filter { e in !atB.contains { $0.id == e.id && $0.kind == e.kind } }
        XCTAssertEqual(counted.count, 3, "events away from B:\n" + log)

        // A: the sphere moved 20 cm; a sphere's rotation cannot be told.
        let moved = counted.filter { $0.kind == .moved }
        XCTAssertEqual(moved.count, 1, log)
        if let m = moved.first {
            XCTAssertTrue(near(m.before, aBefore, within: 0.03), Self.describe(m))
            XCTAssertTrue(near(m.after, aAfter, within: 0.03), Self.describe(m))
            let leftA = t.instances.first { $0.sign < 0 && simd_distance($0.centroid, aBefore) < 0.03 }
            XCTAssertEqual(m.id, leftA?.id)
            XCTAssertNotNil(m.motion, "no ICP motion for A")
            XCTAssertEqual(m.motion?.horizontalDistance ?? 0, 0.20, accuracy: 0.01)
            XCTAssertNil(m.motion?.yawDeg)
        }

        // C removed, D added.
        let removed = counted.filter { $0.kind == .removed }
        XCTAssertEqual(removed.count, 1, log)
        XCTAssertTrue(near(removed.first?.before, cCentre, within: 0.03), log)
        let added = counted.filter { $0.kind == .added }
        XCTAssertEqual(added.count, 1, log)
        XCTAssertTrue(near(added.first?.after, dCentre, within: 0.03), log)
    }

    func testUnchangedSceneGivesNoEvents() {
        let objects = [sphereA(at: aBefore), cubeB(yaw: 0), cubeC]
        let t = tracker()
        run(Scene(before: objects, after: objects), into: t)
        XCTAssertEqual(t.events().filter { !$0.hint }.count, 0)
    }

    func testIngestStaysUnderBudgetOnSyntheticFrames() throws {
        #if DEBUG
        throw XCTSkip("Timing budget is measured in Release only (swift test -c release).")
        #else
        let scene = changedScene
        let frames = Scene.orbitPoses(count: 10, radius: 0.5, height: 0.45).enumerated().map { i, pose in
            scene.frame(id: i, pose: pose, roi: roi)
        }
        let t = tracker()
        let ms = frames.map { t.ingest($0).milliseconds }.sorted()
        let median = (ms[4] + ms[5]) / 2
        print(String(format: "Synthetic ingest ms: median %.2f, per frame %@", median, ms.map { String(format: "%.2f", $0) }.joined(separator: " ")))
        XCTAssertLessThan(median, 60)
        #endif
    }
}
