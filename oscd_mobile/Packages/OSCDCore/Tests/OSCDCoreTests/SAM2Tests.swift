import XCTest
@testable import OSCDCore

final class SAM2Tests: XCTestCase {
    func testParitySAM2EncoderFeedsTheSameCue() throws {
        let d = try Fixtures.dir("cues/frame_000")
        let pkg = try Fixtures.modelPackage("SAM2_1Tiny_2x1024.mlpackage")
        let encoder = try SAM2Encoder(compiledModelURL: try XFeatModel.compile(pkg))
        let liveA = try Fixtures.npy(d, "live"), renA = try Fixtures.npy(d, "render")
        let H = liveA.shape[1], W = liveA.shape[2]
        let live = FloatImage(width: W, height: H, channels: 3, data: liveA.floats())
        let render = FloatImage(width: W, height: H, channels: 3, data: renA.floats())
        let emb = try encoder.encode(render: render, live: live)
        let ours = CuesA.compute(live: live, render: render, embRender: emb.render, embLive: emb.live)
        let want = try Fixtures.npy(d, "C").floats()
        let mae = zip(ours.combined, want).map { abs($0 - $1) }.reduce(0, +) / Float(want.count)
        XCTAssertLessThan(mae, 0.02)
        let ref = try Fixtures.npy(d, "emb").floats(), half = ref.count / 2
        let refCue = CuesA.compute(live: live, render: render, embRender: Array(ref[..<half]), embLive: Array(ref[half...]))
        XCTAssertGreaterThanOrEqual(Self.pearson(ours.feature, refCue.feature), 0.98)
    }

    static func pearson(_ a: [Float], _ b: [Float]) -> Double {
        let n = Double(a.count)
        let ma = a.reduce(0) { $0 + Double($1) } / n, mb = b.reduce(0) { $0 + Double($1) } / n
        var sab = 0.0, saa = 0.0, sbb = 0.0
        for (x, y) in zip(a, b) { let dx = Double(x) - ma, dy = Double(y) - mb; sab += dx * dy; saa += dx * dx; sbb += dy * dy }
        return sab / (saa * sbb).squareRoot()
    }
}
