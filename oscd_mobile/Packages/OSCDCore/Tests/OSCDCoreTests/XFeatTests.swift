import XCTest
import simd
@testable import OSCDCore

final class XFeatTests: XCTestCase {
    /// A single strong peak must come back as one keypoint at the right place with a unit descriptor.
    func testSyntheticPeak() {
        let h = 4, w = 4, H = 32, W = 32
        var heat = [Float](repeating: 0.01, count: H * W)
        heat[10 * W + 13] = 0.9
        let rel = [Float](repeating: 0.5, count: h * w)
        var feats = [Float](repeating: 0, count: 64 * h * w)
        for i in 0..<(h * w) { feats[3 * h * w + i] = 1 }       // channel 3 everywhere
        let kp = XFeatPostprocess.run(feats: feats, heatmap: heat, reliability: rel, h: h, w: w,
                                      topK: 10, imageWidth: 32, imageHeight: 32)
        XCTAssertEqual(kp.count, 1)
        XCTAssertEqual(kp.kpts[0], SIMD2(13, 10))
        XCTAssertEqual(kp.feats[3], 1, accuracy: 1e-5)
    }

    /// Exact O-SCD Detector post-processing on the Sol network outputs.
    func testParityPostprocessMatchesOSCDDetector() throws {
        let d = try Fixtures.dir("xfeat")
        let nf = try Fixtures.npy(d, "net_feats"), hm = try Fixtures.npy(d, "net_heatmap"), rl = try Fixtures.npy(d, "net_reliability")
        let img = try Fixtures.npy(d, "image")
        let got = XFeatPostprocess.run(feats: nf.floats(), heatmap: hm.floats(), reliability: rl.floats(),
                                       h: nf.shape[1], w: nf.shape[2], topK: 6144,
                                       imageWidth: img.shape[1], imageHeight: img.shape[0])
        let want = DescribedKeypoints.load(kpts: try Fixtures.npy(d, "kpts"), feats: try Fixtures.npy(d, "feats"))
        let (posAgree, cosMean) = Self.compare(want: want, got: got, top: 1000)
        XCTAssertGreaterThanOrEqual(posAgree, 0.99)
        XCTAssertGreaterThanOrEqual(cosMean, 0.999)
    }

    /// Core ML network + post-processing end to end, against the O-SCD Detector output.
    func testParityCoreMLDetector() throws {
        let d = try Fixtures.dir("xfeat")
        let pkg = try Fixtures.modelPackage("XFeat_480x352.mlpackage")
        let model = try XFeatModel(compiledModelURL: try XFeatModel.compile(pkg))
        let img = try Fixtures.npy(d, "image")
        let image = FloatImage(hwc: img.floats(), width: img.shape[1], height: img.shape[0], channels: 3)
        let got = try model.detect(image)
        let want = DescribedKeypoints.load(kpts: try Fixtures.npy(d, "kpts"), feats: try Fixtures.npy(d, "feats"))
        let (posAgree, cosMean) = Self.compare(want: want, got: got, top: 1000)
        XCTAssertGreaterThanOrEqual(posAgree, 0.95)
        XCTAssertGreaterThanOrEqual(cosMean, 0.99)
    }

    /// Fraction of O-SCD's first `top` valid keypoints we also found within 1 px, and the mean
    /// descriptor cosine over those pairs.
    static func compare(want: DescribedKeypoints, got: DescribedKeypoints, top: Int) -> (Double, Double) {
        var index: [SIMD2<Int32>: Int] = [:]
        for i in 0..<got.count { index[SIMD2(Int32(got.kpts[i].x.rounded()), Int32(got.kpts[i].y.rounded()))] = i }
        var hit = 0, total = 0, cosSum = 0.0
        for i in 0..<want.count where want.isValid(i) {
            if total == top { break }
            total += 1
            let k = SIMD2(Int32(want.kpts[i].x.rounded()), Int32(want.kpts[i].y.rounded()))
            var found: Int?
            outer: for dy in -1...1 { for dx in -1...1 {
                if let j = index[k &+ SIMD2(Int32(dx), Int32(dy))], simd_distance(got.kpts[j], want.kpts[i]) <= 1 { found = j; break outer }
            } }
            guard let j = found else { continue }
            hit += 1
            var c: Float = 0
            for t in 0..<64 { c += want.feats[i * 64 + t] * got.feats[j * 64 + t] }
            cosSum += Double(c)
        }
        return (Double(hit) / Double(max(total, 1)), hit > 0 ? cosSum / Double(hit) : 0)
    }
}
