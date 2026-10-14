import XCTest
@testable import OSCDCore

final class CuesATests: XCTestCase {
    func constant(_ v: Float, _ w: Int = 32, _ h: Int = 24) -> FloatImage { FloatImage(width: w, height: h, channels: 3, fill: v) }

    func testSSIMIdenticalIsOneAndConstantsMatchFormula() {
        XCTAssertTrue(SSIM.meanMap(constant(0.3), constant(0.3)).allSatisfy { abs($0 - 1) < 1e-5 })
        // Interior pixel, constants 0.2 vs 0.6: (2·0.12 + C1) / (0.04 + 0.36 + C1).
        let m = SSIM.meanMap(constant(0.2), constant(0.6))
        XCTAssertEqual(m[12 * 32 + 16], Float((0.24 + 1e-4) / (0.40 + 1e-4)), accuracy: 1e-4)
    }

    func testIdenticalImagesGiveZeroCue() {
        let p = CuesA.pixel(live: constant(0.4), render: constant(0.4))
        XCTAssertTrue(p.normalized.allSatisfy { $0 == 0 })
    }

    /// Review Focus 3: an all-invalid mask returns zeros everywhere, with no NaN.
    func testAllInvalidMaskGivesZeros() {
        let n = 32 * 24
        let emb = [Float](repeating: 0.1, count: 256 * 64 * 64)
        let c = CuesA.compute(live: constant(0.1), render: constant(0.9), embRender: emb, embLive: emb.map { $0 + 0.5 },
                              valid: [Bool](repeating: false, count: n))
        XCTAssertTrue(c.combined.allSatisfy { $0 == 0 })
        XCTAssertFalse(c.combined.contains { $0.isNaN })
    }

    func testParityCuesA() throws {
        _ = try Fixtures.dir("cues")
        for k in 0..<3 {
            let d = Fixtures.root.appendingPathComponent(String(format: "cues/frame_%03d", k))
            guard FileManager.default.fileExists(atPath: d.path) else { continue }
            let liveA = try Fixtures.npy(d, "live"), renA = try Fixtures.npy(d, "render")
            let H = liveA.shape[1], W = liveA.shape[2]
            let live = FloatImage(width: W, height: H, channels: 3, data: liveA.floats())
            let render = FloatImage(width: W, height: H, channels: 3, data: renA.floats())
            let ssim = SSIM.meanMap(render, live), wantSSIM = try Fixtures.npy(d, "ssim_map").floats()
            XCTAssertLessThan(zip(ssim, wantSSIM).map { abs($0 - $1) }.max()!, 1e-3, "frame \(k) ssim")
            let p = CuesA.pixel(live: live, render: render)
            XCTAssertLessThan(zip(p.raw, try Fixtures.npy(d, "delta_raw").floats()).map { abs($0 - $1) }.max()!, 1e-3, "frame \(k) delta")
            let emb = try Fixtures.npy(d, "emb").floats(), half = emb.count / 2
            let c = CuesA.compute(live: live, render: render, embRender: Array(emb[..<half]), embLive: Array(emb[half...]))
            XCTAssertLessThan(zip(c.combined, try Fixtures.npy(d, "C").floats()).map { abs($0 - $1) }.max()!, 2e-3, "frame \(k) C")
            // An all-true mask must reproduce O-SCD exactly.
            let masked = CuesA.compute(live: live, render: render, embRender: Array(emb[..<half]), embLive: Array(emb[half...]),
                                       valid: [Bool](repeating: true, count: W * H))
            XCTAssertEqual(masked.combined, c.combined)
        }
    }
}
