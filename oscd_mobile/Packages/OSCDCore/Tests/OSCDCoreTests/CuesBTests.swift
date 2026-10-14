import XCTest
@testable import OSCDCore

final class CuesBTests: XCTestCase {
    func image(_ rgb: SIMD3<Float>, _ w: Int, _ h: Int) -> FloatImage {
        FloatImage(width: w, height: h, channels: 3, data: [Float](repeating: rgb.x, count: w * h) + [Float](repeating: rgb.y, count: w * h) + [Float](repeating: rgb.z, count: w * h))
    }

    func testLabReferenceValues() {
        let red = CuesB.srgbToLab(1, 0, 0)
        XCTAssertEqual(red.x, 53.24, accuracy: 0.05); XCTAssertEqual(red.y, 80.09, accuracy: 0.05); XCTAssertEqual(red.z, 67.20, accuracy: 0.05)
        let white = CuesB.srgbToLab(1, 1, 1)
        XCTAssertEqual(white.x, 100, accuracy: 0.05); XCTAssertEqual(white.y, 0, accuracy: 0.05)
    }

    func testDepthCueRampsAndSigns() {
        let r = CuesB.depthCue(live: [0.50, 0.47, 0.515, .nan], rendered: [0.5, 0.5, 0.5, 0.5], valid: [true, true, true, true])
        XCTAssertEqual(r.cue[0], 0)
        XCTAssertEqual(r.cue[1], 1, accuracy: 1e-5); XCTAssertEqual(r.sign[1], 1)     // 3 cm closer: added
        XCTAssertEqual(r.cue[2], 0.25, accuracy: 1e-4); XCTAssertEqual(r.sign[2], -1)  // 1.5 cm farther: removed
        XCTAssertEqual(r.cue[3], 0)                                                     // Review Focus 4: NaN depth is ignored
    }

    /// Review Focus 4: low-confidence or zero LiDAR depth becomes NaN, never a 0 m surface.
    func testUpsampleDepthDropsLowConfidence() {
        let d = CuesB.upsampleDepth([0.5, 0.0, 0.7, 0.8], confidence: [2, 2, 1, 2], width: 2, height: 2, toWidth: 4, toHeight: 4)
        XCTAssertEqual(d[0], 0.5)
        XCTAssertTrue(d[2].isNaN)           // depth 0
        XCTAssertTrue(d[2 * 4 + 0].isNaN)   // confidence 1
        XCTAssertEqual(d[3 * 4 + 3], 0.8)
    }

    func testColorCueFiresOnRecolorButNotOnLightnessOrExposure() {
        let w = 16, h = 16, n = w * h, valid = [Bool](repeating: true, count: n), nanDepth = [Float](repeating: .nan, count: n)
        let recolor = CuesB.colorCue(live: image(SIMD3(0, 0, 1), w, h), render: image(SIMD3(1, 0, 0), w, h),
                                     liveDepth: nanDepth, renderDepth: nanDepth, valid: valid)
        XCTAssertGreaterThan(recolor.min()!, 0.99)
        let lightness = CuesB.colorCue(live: image(SIMD3(0.6, 0.6, 0.6), w, h), render: image(SIMD3(0.9, 0.9, 0.9), w, h),
                                       liveDepth: nanDepth, renderDepth: nanDepth, valid: valid)
        XCTAssertLessThan(lightness.max()!, 0.05)
        // Exposure ×1.3 on 32×32 (≥ 500 agreeing-depth pixels): the gain alignment removes it.
        let W = 32, N = W * W, depth = [Float](repeating: 0.6, count: N)
        let exposure = CuesB.colorCue(live: image(SIMD3(0.52, 0.39, 0.26), W, W), render: image(SIMD3(0.4, 0.3, 0.2), W, W),
                                      liveDepth: depth, renderDepth: depth, valid: [Bool](repeating: true, count: N))
        XCTAssertLessThan(exposure.max()!, 0.05)
    }

    /// Review Focus 3: nothing valid → zeros.
    func testCombineClampsAndZeroesInvalid() {
        let c = CuesB.combine(cA: [1.5, 0.2, 1.0], depth: [1, 0, 1], color: [1, 0, 0], valid: [true, true, false])
        XCTAssertEqual(c, [2, 0.2, 0])
        let none = CuesB.colorCue(live: image(SIMD3(1, 0, 0), 4, 4), render: image(SIMD3(0, 0, 1), 4, 4),
                                  liveDepth: [Float](repeating: .nan, count: 16), renderDepth: [Float](repeating: .nan, count: 16),
                                  valid: [Bool](repeating: false, count: 16))
        XCTAssertEqual(none, [Float](repeating: 0, count: 16))
    }
}
