import XCTest
import simd
@testable import OSCDCore

final class BeforePhotoFeatureTests: XCTestCase {
    private var dir: URL!

    override func setUp() {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("before-photo-\(UUID().uuidString)")
    }
    override func tearDown() { try? FileManager.default.removeItem(at: dir) }

    /// Two 32×24 keyframes looking down ARKit −z: id 0 at the origin, id 1 0.3 m to the right.
    private func makeStore() throws -> SessionStore {
        let m = SessionManifest(deviceModel: "test", captureWidth: 128, captureHeight: 96, depthWidth: 8, depthHeight: 6,
                                intrinsics: Intrinsics(f: 40, cx: 15.5, cy: 11.5, width: 32, height: 24))
        let store = try SessionStore(create: dir, manifest: m)
        for (id, x) in [(0, Float(0)), (1, Float(0.3))] {
            let pose: [Float] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1]
            try store.append(.keyframes, record: FrameRecord(id: id, timestamp: Double(id), arkitCameraToWorld: pose),
                             rgb: [UInt8](repeating: UInt8(100 + id), count: 32 * 24 * 3), depth: [Float](repeating: 1.5, count: 48),
                             confidence: [UInt8](repeating: 2, count: 48))
        }
        return store
    }

    /// 1 m in front of the cameras; x = 0.55 is outside keyframe 0's view (u ≈ 37) and inside keyframe 1's (u ≈ 25).
    private let rightOnly: [SIMD3<Float>] = [SIMD3(0.50, 0.0, -1), SIMD3(0.55, 0.05, -1), SIMD3(0.60, -0.05, -1)]

    func testBestKeyframeIsTheOneThatSeesThePoints() throws {
        let store = try makeStore()
        let best = try XCTUnwrap(BeforePhotoFeature.bestKeyframe(for: rightOnly, store: store))
        XCTAssertEqual(best.record.id, 1)
        XCTAssertEqual(best.pixels.count, rightOnly.count)
        // 0.55 − 0.3 = 0.25 m right of keyframe 1 at 1 m: u = 40·0.25 + 15.5 = 25.5 → 26, v = 11.5 − 2 → 10 (y up is image up).
        XCTAssertTrue(best.pixels.contains(10 * 32 + 26))
        XCTAssertTrue(best.pixels.allSatisfy { $0 >= 0 && $0 < 32 * 24 })
    }

    func testTieGoesToTheLowerKeyframeIndex() throws {
        let store = try makeStore()
        let both = [SIMD3<Float>(0.15, 0, -1), SIMD3<Float>(0.16, 0.01, -1)]     // seen by both
        XCTAssertEqual(BeforePhotoFeature.bestKeyframe(for: both, store: store)?.record.id, 0)
    }

    func testNoVisibleKeyframeGivesNil() throws {
        let store = try makeStore()
        let behind = [SIMD3<Float>(0, 0, 1), SIMD3<Float>(0.2, 0, 2)]
        XCTAssertNil(BeforePhotoFeature.bestKeyframe(for: behind, store: store))
        var cache: [Int: [Float]] = [:]
        var calls = 0
        let r = try BeforePhotoFeature.pooled(for: behind, store: store, cache: &cache) { _ in calls += 1; return [] }
        XCTAssertNil(r)
        XCTAssertEqual(calls, 0)
        XCTAssertNil(BeforePhotoFeature.bestKeyframe(for: [], store: store))
    }

    func testPooledUsesCachedEmbeddingOncePerKeyframe() throws {
        let store = try makeStore()
        // Embedding value = channel + 0.001·cell, so the pooled value depends on which cells the pixels hit.
        let cells = 64 * 64
        let embedding: [Float] = (0..<(256 * cells)).map { (i: Int) -> Float in
            let channel = Float(i / cells), cell = Float(i % cells)
            return channel + 0.001 * cell
        }
        var cache: [Int: [Float]] = [:]
        var calls = 0
        var seenSize: (Int, Int)?
        let encode: (FloatImage) throws -> [Float] = { img in calls += 1; seenSize = (img.width, img.height); return embedding }

        let a = try XCTUnwrap(BeforePhotoFeature.pooled(for: rightOnly, store: store, cache: &cache, encode: encode))
        let other = [SIMD3<Float>(0.52, 0.02, -1), SIMD3<Float>(0.58, -0.02, -1)]
        let b = try XCTUnwrap(BeforePhotoFeature.pooled(for: other, store: store, cache: &cache, encode: encode))

        XCTAssertEqual(calls, 1)
        XCTAssertEqual(seenSize?.0, 32); XCTAssertEqual(seenSize?.1, 24)
        XCTAssertEqual(a.record.id, 1); XCTAssertEqual(b.record.id, 1)
        XCTAssertEqual(Set(cache.keys), [1])

        let best = try XCTUnwrap(BeforePhotoFeature.bestKeyframe(for: rightOnly, store: store))
        let expected = SAM2Pool.mean(embedding: embedding, pixels: best.pixels, width: 32, height: 24)
        XCTAssertEqual(a.feature.count, 256)
        XCTAssertEqual(a.feature, expected)

        let xs = best.pixels.map { $0 % 32 }, ys = best.pixels.map { $0 / 32 }
        XCTAssertEqual(a.rect, PixelRect(minX: xs.min()!, minY: ys.min()!, maxX: xs.max()!, maxY: ys.max()!))
    }

    func testEncodeErrorPropagatesAndLeavesTheCacheEmpty() throws {
        struct Boom: Error {}
        let store = try makeStore()
        var cache: [Int: [Float]] = [:]
        XCTAssertThrowsError(try BeforePhotoFeature.pooled(for: rightOnly, store: store, cache: &cache) { _ in throw Boom() })
        XCTAssertTrue(cache.isEmpty)
    }
}
