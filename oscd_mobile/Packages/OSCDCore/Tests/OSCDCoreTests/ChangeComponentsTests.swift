import XCTest
@testable import OSCDCore

final class ChangeComponentsTests: XCTestCase {
    /// A w×h frame with a rectangular block marked changed, every block pixel carrying `sign` and `d`.
    struct Field {
        let w: Int, h: Int
        var changed: [Bool], sign: [Int8], d: [Float]
        init(_ w: Int, _ h: Int) {
            self.w = w; self.h = h
            changed = [Bool](repeating: false, count: w * h)
            sign = [Int8](repeating: 0, count: w * h)
            d = [Float](repeating: .nan, count: w * h)
        }
        mutating func block(x0: Int, y0: Int, x1: Int, y1: Int, sign s: Int8, d dv: Float) {
            for y in y0...y1 { for x in x0...x1 {
                changed[y * w + x] = true; sign[y * w + x] = s; d[y * w + x] = dv
            } }
        }
        func extract(minPixels: Int = 1) -> [Component2D] {
            ChangeComponents.extract(changed: changed, sign: sign, d: d, width: w, height: h, minPixels: minPixels)
        }
    }

    func image(_ rgb: SIMD3<Float>, _ w: Int, _ h: Int) -> FloatImage {
        FloatImage(width: w, height: h, channels: 3,
                   data: [Float](repeating: rgb.x, count: w * h) + [Float](repeating: rgb.y, count: w * h)
                       + [Float](repeating: rgb.z, count: w * h))
    }

    func testSignUsesMedianOffsetAndTolerance() {
        let dRen = [Float](repeating: 0.400, count: 10)
        var dLive = [Float](repeating: 0.410, count: 10)
        dLive[3] = 0.380; dLive[7] = 0.440
        let r = ChangeSign.compute(dLive: dLive, dRen: dRen, valid: [Bool](repeating: true, count: 10))
        XCTAssertEqual(r.sign[3], 1)
        XCTAssertEqual(r.sign[7], -1)
        for i in [0, 1, 2, 4, 5, 6, 8, 9] { XCTAssertEqual(r.sign[i], 0) }
        XCTAssertEqual(r.d[3], -0.030, accuracy: 1e-5)
    }

    func testSignIsZeroWithoutDepth() {
        let r = ChangeSign.compute(dLive: [Float](repeating: .nan, count: 6), dRen: [Float](repeating: 0.4, count: 6),
                                   valid: [Bool](repeating: true, count: 6))
        XCTAssertEqual(r.sign, [Int8](repeating: 0, count: 6))
        XCTAssertTrue(r.d.allSatisfy { $0.isNaN })
    }

    func testSeparateBlobsKeepTheirSign() {
        var f = Field(20, 10)
        f.block(x0: 2, y0: 2, x1: 5, y1: 5, sign: 1, d: -0.05)
        f.block(x0: 12, y0: 2, x1: 15, y1: 5, sign: -1, d: 0.05)
        let c = f.extract().sorted { $0.rect.minX < $1.rect.minX }
        XCTAssertEqual(c.count, 2)
        XCTAssertEqual(c.map(\.sign), [1, -1])
        XCTAssertEqual(c.map(\.pixels.count), [16, 16])
        XCTAssertEqual(c[0].rect, PixelRect(minX: 2, minY: 2, maxX: 5, maxY: 5))
        XCTAssertEqual(c[1].rect, PixelRect(minX: 12, minY: 2, maxX: 15, maxY: 5))
    }

    func testTouchingMixedBlobIsSplitBySign() {
        var f = Field(10, 6)
        f.block(x0: 1, y0: 1, x1: 4, y1: 4, sign: 1, d: -0.05)
        f.block(x0: 5, y0: 1, x1: 8, y1: 4, sign: -1, d: 0.05)
        let c = f.extract()
        XCTAssertEqual(c.count, 2)
        XCTAssertEqual(Set(c.map(\.sign)), [1, -1])
        XCTAssertEqual(c.map(\.pixels.count), [16, 16])
    }

    func testMostlyFlatBlobIsFlat() {
        var f = Field(10, 10)
        f.block(x0: 1, y0: 1, x1: 6, y1: 6, sign: 0, d: 0)
        for i in 0..<4 { f.sign[(1 * f.w) + 1 + i] = 1; f.d[(1 * f.w) + 1 + i] = -0.05 }
        let c = f.extract()
        XCTAssertEqual(c.count, 1)
        XCTAssertEqual(c[0].sign, 0)
        XCTAssertEqual(c[0].pixels.count, 36)
    }

    func testSmallComponentsAreDropped() {
        var f = Field(10, 10)
        for x in 2..<7 { f.changed[3 * f.w + x] = true; f.sign[3 * f.w + x] = 1; f.d[3 * f.w + x] = -0.05 }
        XCTAssertTrue(f.extract(minPixels: 40).isEmpty)
        XCTAssertEqual(f.extract(minPixels: 5).count, 1)
    }

    func testPoolMapsPixelsToCells() {
        let w = 480, h = 360, cells = 64 * 64
        var emb = [Float](repeating: 0, count: 256 * cells)
        emb[63 * 64 + 63] = 1
        let corner = SAM2Pool.mean(embedding: emb, pixels: [359 * w + 479], width: w, height: h)
        XCTAssertEqual(corner.count, 256)
        XCTAssertEqual(corner[0], 1)
        XCTAssertEqual(corner[1], 0)
        XCTAssertEqual(SAM2Pool.mean(embedding: emb, pixels: [0], width: w, height: h)[0], 0)
    }

    func testHistogramSumsToOneAndIntersects() {
        let w = 8, h = 8, all = Array(0..<(w * h))
        let red = LabHistogram.compute(image: image(SIMD3(1, 0, 0), w, h), pixels: all)
        let blue = LabHistogram.compute(image: image(SIMD3(0, 0, 1), w, h), pixels: all)
        XCTAssertEqual(red.count, LabHistogram.bins)
        XCTAssertEqual(red.reduce(0, +), 1, accuracy: 1e-5)
        XCTAssertEqual(LabHistogram.intersection(red, red), 1, accuracy: 1e-5)
        XCTAssertEqual(LabHistogram.intersection(red, blue), 0)
    }

    func testRenderGainMatchesColorCueStep() {
        let w = 30, h = 20, n = w * h
        var render = image(SIMD3(0, 0, 0), w, h)
        for i in 0..<n {
            let v = 0.2 + 0.5 * Float(i % 7) / 7
            render.data[i] = v; render.data[n + i] = v * 0.9; render.data[2 * n + i] = v * 0.8
        }
        var live = render
        let k = SIMD3<Float>(1.2, 1.0, 0.8)
        for c in 0..<3 { for i in 0..<n { live.data[c * n + i] *= k[c] } }
        let depth = [Float](repeating: 0.5, count: n), valid = [Bool](repeating: true, count: n)
        let g = CuesB.renderGain(live: live, render: render, liveDepth: depth, renderDepth: depth, valid: valid)
        XCTAssertEqual(g.x, 1.2, accuracy: 0.01); XCTAssertEqual(g.y, 1.0, accuracy: 0.01); XCTAssertEqual(g.z, 0.8, accuracy: 0.01)
        let cue = CuesB.colorCue(live: live, render: render, liveDepth: depth, renderDepth: depth, valid: valid)
        XCTAssertTrue(cue.allSatisfy { $0 == 0 })
    }
}
