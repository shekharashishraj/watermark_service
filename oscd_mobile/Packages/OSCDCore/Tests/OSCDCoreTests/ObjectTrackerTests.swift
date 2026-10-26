import simd
import XCTest
@testable import OSCDCore

final class ObjectTrackerTests: XCTestCase {
    /// Tests use blocks of 4–36 pixels, below the 40-pixel default floor.
    func config() -> TrackerConfig {
        var c = TrackerConfig()
        c.minPixels = 4
        return c
    }

    /// A camera at world x = `cameraX`, looking down +z with no rotation.
    func pose(cameraX: Float = 0) -> Pose { Pose(R: matrix_identity_float3x3, t: SIMD3(-cameraX, 0, 0)) }

    func inBlock(_ x: Int, _ y: Int, x0: Int, y0: Int, size: Int) -> Bool {
        x >= x0 && x < x0 + size && y >= y0 && y < y0 + size
    }

    /// A table at 0.50 m with a size×size block at (x0, y0) whose live and rendered depths are given.
    func blockFrame(id: Int, x0: Int = 13, y0: Int = 9, size: Int = 6, cameraX: Float = 0,
                    live: Float = 0.45, rendered: Float = 0.50, feature: [Float] = []) -> FrameContext {
        let b = { (x: Int, y: Int) in self.inBlock(x, y, x0: x0, y0: y0, size: size) }
        return frame(id: id, pose: pose(cameraX: cameraX),
                     dLive: { b($0, $1) ? live : 0.50 }, dRen: { b($0, $1) ? rendered : 0.50 },
                     changed: { b($0, $1) }, feature: feature)
    }

    func testArrivedPixelsBackProjectWithLiveDepth() {
        let t = ObjectTracker(config: config())
        var feature = [Float](repeating: 0, count: 256)
        feature[7] = 2
        let stats = t.ingest(blockFrame(id: 0, feature: feature))
        XCTAssertEqual(stats.components, 1)
        XCTAssertEqual(stats.created, 1)
        XCTAssertEqual(t.instances.count, 1)
        let i = t.instances[0]
        XCTAssertEqual(i.id, 0)
        XCTAssertEqual(i.sign, 1)
        XCTAssertEqual(i.centroid.z, 0.45, accuracy: 0.01)
        XCTAssertGreaterThanOrEqual(i.voxelCount, 1)
        XCTAssertEqual(i.featureSource, .live)
        XCTAssertEqual(i.feature[7], 2, accuracy: 1e-5)
        XCTAssertEqual(i.histogram.reduce(0, +), 1, accuracy: 1e-4)
        XCTAssertEqual(i.points.count, 36)
        XCTAssertEqual(i.frames, [0])
    }

    func testLeftPixelsBackProjectWithRenderedDepth() {
        let t = ObjectTracker(config: config())
        t.ingest(blockFrame(id: 0, live: 0.50, rendered: 0.45))
        XCTAssertEqual(t.instances.count, 1)
        XCTAssertEqual(t.instances[0].sign, -1)
        XCTAssertEqual(t.instances[0].centroid.z, 0.45, accuracy: 0.01)
        XCTAssertEqual(t.instances[0].featureSource, .render)
    }

    func testSameSignOverlappingComponentsMerge() {
        // f = 30 at 0.45 m: one pixel spans 1.5 cm, so moving the camera 1.5 cm along +x and the block one
        // pixel to the left back-projects to the same world points.
        let t = ObjectTracker(config: config())
        t.ingest(blockFrame(id: 0))
        let stats = t.ingest(blockFrame(id: 1, x0: 12, cameraX: 0.015))
        XCTAssertEqual(t.instances.count, 1)
        XCTAssertEqual(t.instances[0].frames, [0, 1])
        XCTAssertEqual(stats.merged, 1)
        XCTAssertEqual(stats.created, 0)
        XCTAssertEqual(t.instances[0].voxelCount, 36)
    }

    func testComponentBridgingTwoInstancesMergesIntoLowestID() {
        let t = ObjectTracker(config: config())
        t.ingest(blockFrame(id: 0, x0: 8, y0: 9, size: 4))
        t.ingest(blockFrame(id: 1, x0: 20, y0: 9, size: 4))
        XCTAssertEqual(t.instances.map(\.id), [0, 1])
        // One 16×4 strip covering both blocks.
        let strip = { (x: Int, y: Int) in x >= 8 && x < 24 && y >= 9 && y < 13 }
        let stats = t.ingest(frame(id: 2, pose: pose(), dLive: { strip($0, $1) ? 0.45 : 0.50 }, dRen: { _, _ in 0.50 },
                                   changed: strip))
        XCTAssertEqual(stats.merged, 1)
        XCTAssertEqual(stats.created, 0)
        XCTAssertEqual(t.instances.map(\.id), [0])
        XCTAssertEqual(t.instances[0].frames, [0, 1, 2])
        XCTAssertEqual(t.instances[0].voxelCount, 64)
        XCTAssertEqual(t.instances[0].points.count, 16 + 16 + 64)
    }

    func testDifferentSignsNeverMerge() {
        let t = ObjectTracker(config: config())
        t.ingest(blockFrame(id: 0, live: 0.45, rendered: 0.50))
        t.ingest(blockFrame(id: 1, live: 0.50, rendered: 0.45))
        XCTAssertEqual(t.instances.count, 2)
        XCTAssertEqual(Set(t.instances.map(\.sign)), [1, -1])
    }

    func testConfirmationNeedsThreeFramesAndFifteenVoxels() {
        let c = config()
        let t = ObjectTracker(config: c)
        t.ingest(blockFrame(id: 0))
        t.ingest(blockFrame(id: 1))
        XCTAssertFalse(t.instances[0].isConfirmed(c))
        t.ingest(blockFrame(id: 2))
        XCTAssertTrue(t.instances[0].isConfirmed(c))

        let small = ObjectTracker(config: c)
        for id in 0..<5 { small.ingest(blockFrame(id: id, x0: 15, y0: 11, size: 2)) }
        XCTAssertEqual(small.instances.count, 1)
        XCTAssertEqual(small.instances[0].frames.count, 5)
        XCTAssertLessThanOrEqual(small.instances[0].voxelCount, 4)
        XCTAssertFalse(small.instances[0].isConfirmed(c))
    }

    func testFlatInstanceIsAbsorbedByOverlappingSigned() {
        let t = ObjectTracker(config: config())
        t.ingest(blockFrame(id: 0, live: 0.45, rendered: 0.50))
        // Depth agrees on the block (sign 0) but it is marked changed: a flat component on the same voxels.
        t.ingest(blockFrame(id: 1, live: 0.45, rendered: 0.45))
        XCTAssertEqual(t.instances.count, 1)
        XCTAssertEqual(t.instances[0].id, 0)
        XCTAssertEqual(t.instances[0].sign, 1)
        XCTAssertEqual(t.instances[0].frames, [0, 1])
    }

    func testInstanceTouchingRoiEdgeIsDropped() {
        // The block's voxels end at z = 0.46 m.
        var c = config()
        c.roi = AABB(min: SIMD3(-1, -1, 0), max: SIMD3(1, 1, 0.47))
        let near = ObjectTracker(config: c)
        let stats = near.ingest(blockFrame(id: 0))
        XCTAssertEqual(near.instances.count, 0)
        XCTAssertEqual(stats.dropped, 1)

        c.roi = AABB(min: SIMD3(-1, -1, 0), max: SIMD3(1, 1, 0.56))
        let far = ObjectTracker(config: c)
        far.ingest(blockFrame(id: 0))
        XCTAssertEqual(far.instances.count, 1)
    }

    /// The edge drop applies to the four vertical faces only: an object resting on the table touches the ROI's floor band.
    func testInstanceTouchingRoiFloorIsKeptButSideIsDropped() throws {
        let probe = ObjectTracker(config: config())
        probe.ingest(blockFrame(id: 0))
        let box = try XCTUnwrap(probe.instances.first).bbox

        var c = config()
        c.roi = AABB(min: SIMD3(-1, box.min.y - 0.001, 0), max: SIMD3(1, 1, 1))
        let floor = ObjectTracker(config: c)
        floor.ingest(blockFrame(id: 0))
        XCTAssertEqual(floor.instances.count, 1, "touching the ROI's min-y face must not drop it")

        c.roi = AABB(min: SIMD3(box.min.x - 0.001, -1, 0), max: SIMD3(1, 1, 1))
        let side = ObjectTracker(config: c)
        let stats = side.ingest(blockFrame(id: 0))
        XCTAssertEqual(side.instances.count, 0, "within the margin of the min-x face it is still dropped")
        XCTAssertEqual(stats.dropped, 1)
    }

    /// A changed patch whose points lie at table height (|d| = 3 cm) adds nothing once the support height is known.
    func testChangeAtTableHeightIsIgnoredWithSupportHeight() throws {
        let ungated = ObjectTracker(config: config())
        ungated.ingest(blockFrame(id: 0, live: 0.47, rendered: 0.50))
        let top = try XCTUnwrap(ungated.instances.first).bbox.max.y       // every point's y is below this

        var c = config()
        c.supportHeight = top
        let gated = ObjectTracker(config: c)
        let stats = gated.ingest(blockFrame(id: 0, live: 0.47, rendered: 0.50))
        XCTAssertEqual(gated.instances.count, 0)
        XCTAssertEqual(stats.components, 0)

        // The same for a left patch (its points come from the rendered depth).
        let left = ObjectTracker(config: c)
        left.ingest(blockFrame(id: 0, live: 0.50, rendered: 0.47))
        XCTAssertEqual(left.instances.count, 0)

        // A support plane well below the patch keeps it.
        c.supportHeight = top - 1
        let low = ObjectTracker(config: c)
        low.ingest(blockFrame(id: 0, live: 0.47, rendered: 0.50))
        XCTAssertEqual(low.instances.count, 1)
    }

    /// The model's table sits 1.5 cm nearer than LiDAR's (dRen 0.485 where dLive reads 0.500, camera 0.5 m above the
    /// table looking straight down). Left points are placed with dRen + the frame's offset, so a removed object (dRen
    /// 0.45) stays above the LiDAR support while the model's own sheet (left and flat patches) falls on it.
    func testLeftPointsUseTheFrameOffsetAgainstTheSupport() throws {
        // World → camera for a camera at (0, 0.5, 0) looking down −y: camera z = −world y, so world y = 0.5 − depth.
        let down = Pose(R: simd_float3x3(rows: [SIMD3(1, 0, 0), SIMD3(0, 0, 1), SIMD3(0, -1, 0)]), t: SIMD3(0, 0, 0.5))
        let object = { (x: Int, y: Int) in self.inBlock(x, y, x0: 4, y0: 9, size: 6) }
        let leftSheet = { (x: Int, y: Int) in self.inBlock(x, y, x0: 14, y0: 9, size: 6) }
        let flatSheet = { (x: Int, y: Int) in self.inBlock(x, y, x0: 23, y0: 9, size: 6) }
        let ctx = frame(id: 0, pose: down, dLive: { _, _ in 0.500 },
                        dRen: { object($0, $1) ? 0.45 : (leftSheet($0, $1) ? 0.478 : 0.485) },
                        changed: { object($0, $1) || leftSheet($0, $1) || flatSheet($0, $1) })
        var c = config()
        c.supportHeight = 0                                  // LiDAR's table
        let t = ObjectTracker(config: c)
        t.ingest(ctx)
        XCTAssertEqual(t.instances.count, 1, "only the removed object: \(t.instances.map { ($0.sign, $0.centroid.y) })")
        let i = try XCTUnwrap(t.instances.first)
        XCTAssertEqual(i.sign, -1)
        XCTAssertEqual(i.centroid.y, 0.035, accuracy: 0.01)  // 0.5 − (0.45 + 0.015)
    }

    func testUnconfirmedInstanceIsDroppedAfterMisses() {
        let t = ObjectTracker(config: config())
        t.ingest(blockFrame(id: 0))
        XCTAssertEqual(t.instances.count, 1)
        var dropped = 0
        for id in 1...10 {
            dropped += t.ingest(frame(id: id, pose: pose(), dLive: { _, _ in 0.50 }, dRen: { _, _ in 0.50 },
                                      changed: { _, _ in false })).dropped
        }
        XCTAssertEqual(t.instances.count, 0)
        XCTAssertEqual(dropped, 1)
    }

    func testFrameWithoutDepthAddsNothing() {
        let t = ObjectTracker(config: config())
        let stats = t.ingest(frame(id: 0, pose: pose(), dLive: { _, _ in .nan }, dRen: { _, _ in 0.50 },
                                   changed: { _, _ in true }))
        XCTAssertEqual(stats.components, 0)
        XCTAssertEqual(stats.created, 0)
        XCTAssertEqual(t.instances.count, 0)
    }

    func testBestCropFollowsLargestView() {
        let t = ObjectTracker(config: config())
        t.ingest(blockFrame(id: 0, x0: 14, y0: 10, size: 4))
        XCTAssertEqual(t.instances[0].bestFrame, 0)
        XCTAssertEqual(t.instances[0].bestPixelCount, 16)
        t.ingest(blockFrame(id: 1, x0: 13, y0: 9, size: 6))
        XCTAssertEqual(t.instances.count, 1)
        let i = t.instances[0]
        XCTAssertEqual(i.bestFrame, 1)
        XCTAssertEqual(i.bestPixelCount, 36)
        XCTAssertEqual(i.bestRect, PixelRect(minX: 13, minY: 9, maxX: 18, maxY: 14))
        let crop = try? XCTUnwrap(i.bestCrop)
        XCTAssertGreaterThanOrEqual(crop?.width ?? 0, 6)
        XCTAssertEqual(crop?.channels, 3)
    }

    func testSetFeatureAndLabel() {
        let t = ObjectTracker(config: config())
        t.ingest(blockFrame(id: 0))
        t.setFeature(instance: 0, [Float](repeating: 1, count: 256), source: .beforePhoto)
        t.setLabel(instance: 0, "Ball", confidence: 0.8)
        XCTAssertEqual(t.instances[0].featureSource, .beforePhoto)
        XCTAssertEqual(t.instances[0].feature.first, 1)
        XCTAssertEqual(t.instances[0].label, "Ball")
        XCTAssertEqual(t.instances[0].labelConfidence, 0.8)
    }
}
