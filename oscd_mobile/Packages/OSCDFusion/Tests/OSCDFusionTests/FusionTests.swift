import XCTest
import simd
import OSCDCore
@testable import OSCDFusion

final class FusionTests: XCTestCase {
    // MARK: SSF loss (O-SCD oscd.py:221-226) — values and gradient

    func testSSFLossMatchesFormulaAndFiniteDifferences() {
        let W = 6, H = 5, n = W * H
        var s = SeededSampler(seed: 3)
        // render values kept ≥ 0.02 away from the clamp edges so central differences are valid
        var render = (0..<(n * 3)).map { _ in Float(0.05 + 0.9 * s.uniform()) }
        render[0] = 1.3; render[4] = -0.2                  // clamped channels: zero gradient
        let cue = (0..<n).map { _ in Float(2 * s.uniform()) }
        let valid = (0..<n).map { $0 % 7 != 0 }
        let (loss, grad) = SSF.lossAndGrad(render: render, cue: cue, valid: valid, width: W, height: H)

        func reference(_ r: [Float]) -> Double {
            var sumCm = 0.0, sumM = 0.0, nv = 0.0
            for p in 0..<n where valid[p] {
                var M = 0.0
                for c in 0..<3 { M += Double(min(1, max(0, r[p * 3 + c]))) }
                let m = 1 / (1 + exp(-M / 3))
                sumCm += Double(cue[p]) * (1 - m); sumM += m; nv += 1
            }
            let mean = sumM / nv
            return sumCm / nv + log(mean * mean + 1)
        }
        XCTAssertEqual(Double(loss), reference(render), accuracy: 1e-5)
        XCTAssertEqual(grad[0], 0); XCTAssertEqual(grad[4], 0)
        for i in [1, 2, 3, 8, 20, 47, 59, 88] {
            var up = render, dn = render
            up[i] += 1e-3; dn[i] -= 1e-3
            let fd = (reference(up) - reference(dn)) / 2e-3
            XCTAssertEqual(Double(grad[i]), fd, accuracy: 1e-4, "index \(i)")
        }
        for p in 0..<n where !valid[p] { for c in 0..<3 { XCTAssertEqual(grad[p * 3 + c], 0) } }
    }

    // MARK: torch.optim.Adam (eps 1e-15) — reference values printed by PyTorch 2.5

    func testTorchAdamMatchesPyTorch() {
        var p: [Float] = [1, -2, 0.5], m = [Float](repeating: 0, count: 3), v = m
        let grads: [[Float]] = [[0.3, -1, 0], [0.1, 2, 0], [-0.4, 0.5, 0.25]]
        let want: [[Float]] = [[0.99000001, -1.99000001, 0.5], [0.981289387, -1.993661046, 0.5],
                               [0.982129097, -1.997817516, 0.493611872]]
        for (t, g) in grads.enumerated() {
            TorchAdam.step(param: &p, grad: g, expAvg: &m, expAvgSq: &v, step: t + 1, lr: 0.01)
            for k in 0..<3 { XCTAssertEqual(p[k], want[t][k], accuracy: 2e-6, "step \(t + 1) element \(k)") }
        }
    }

    // MARK: O-SCD densification on a tiny cloud

    func testCloneSmallAndSplitLargeAtThreshold() throws {
        let ply = try TestPLY.write([
            .init(position: SIMD3(0, 0, 2), scale: 0.005, opacity: 0.9),   // small → clone
            .init(position: SIMD3(0.5, 0, 2), scale: 0.2, opacity: 0.9),   // large → split
            .init(position: SIMD3(-0.5, 0, 2), scale: 0.2, opacity: 0.9),   // below the gradient threshold
        ])
        let f = try ChangeFusion(plyPath: ply.path, referenceCentres: [SIMD3(0, 0, 0)])
        // extent 1.0 → percent_dense·extent = 0.01: 0.005 clones, 0.2 splits
        try f.debugDensify(grads: [2e-3, 2e-3, 1e-4], threshold: 1e-3, extent: 1.0)
        XCTAssertEqual(f.count, 5)                                  // 3 + 1 clone + 2 children − 1 parent
        let scales = f.debugLogScales()
        XCTAssertEqual(scales.filter { abs($0 - log(Float(0.005))) < 1e-5 }.count, 2 * 3)
        XCTAssertEqual(scales.filter { abs($0 - log(Float(0.2 / 1.6))) < 1e-5 }.count, 2 * 3)
        XCTAssertEqual(scales.filter { abs($0 - log(Float(0.2))) < 1e-5 }.count, 1 * 3)
    }

    func testPruneLowOpacityAndResetOpacity() throws {
        let ply = try TestPLY.write([.init(position: SIMD3(0, 0, 2), scale: 0.01, opacity: 0.9),
                                     .init(position: SIMD3(0.1, 0, 2), scale: 0.01, opacity: 0.2)])
        let f = try ChangeFusion(plyPath: ply.path, referenceCentres: [SIMD3(0, 0, 0)])
        try f.debugPrune(minOpacity: 0.4)
        XCTAssertEqual(f.count, 1)
        try f.debugResetOpacity()
        XCTAssertEqual(f.debugOpacities()[0], 0.01, accuracy: 1e-6)          // min(σ(op), 0.01)
    }

    func testChangeValuesStartAtHalfGrey() throws {
        // f_dc = 0 → colour 0.5 where a splat covers the pixel (O-SCD load_ply_change).
        let f = try ChangeFusion(plyPath: try TestPLY.plane().path, referenceCentres: [SIMD3(0, 0, 0)])
        let map = try f.changeMap(pose: .identity, intrinsics: TestPLY.camera)
        XCTAssertEqual(map[(TestPLY.camera.height / 2) * TestPLY.camera.width + TestPLY.camera.width / 2], 0.5, accuracy: 0.03)
    }

    // MARK: Expected depth of the reference (Method B, spec §8.1)

    func testExpectedDepthOfThePlaneIsItsDistanceAndNaNWhereUncovered() throws {
        let K = TestPLY.camera
        let d = try ReferenceDepth(plyPath: try TestPLY.plane().path)
        let centre = (K.height / 2) * K.width + K.width / 2
        XCTAssertEqual(try d.expectedDepth(pose: .identity, intrinsics: K)[centre], 2, accuracy: 0.01)
        // Moved 0.5 m towards the plane: world→camera t = (0, 0, −0.5).
        let closer = try d.expectedDepth(pose: Pose(R: matrix_identity_float3x3, t: SIMD3(0, 0, -0.5)), intrinsics: K)
        XCTAssertEqual(closer[centre], 1.5, accuracy: 0.01)
        // Shifted 2 m sideways the plane is out of view: no coverage, so no depth.
        let away = try d.expectedDepth(pose: Pose(R: matrix_identity_float3x3, t: SIMD3(-2, 0, 0)), intrinsics: K)
        XCTAssertTrue(away[centre].isNaN)
    }

    // MARK: End to end — the change model learns where several views agree

    func testFusionLightsUpTheRegionAllViewsMarkAsChanged() throws {
        let K = TestPLY.camera
        let f = try ChangeFusion(plyPath: try TestPLY.plane().path, referenceCentres: [SIMD3(0, 0, 0), SIMD3(0.1, 0, 0)])
        // a 0.3 m square at the centre of the plane (z = 2) is "changed" in every view
        func cue(for pose: Pose) -> [Float] {
            var c = [Float](repeating: 0, count: K.width * K.height)
            for y in 0..<K.height { for x in 0..<K.width {
                let ray = K.unproject(SIMD2(Float(x), Float(y)), depth: 1)
                let camToWorld = pose.inverse
                let o = camToWorld.apply(.zero), d = camToWorld.apply(ray) - o
                let p = o + d * ((2 - o.z) / d.z)
                if abs(p.x) < 0.15 && abs(p.y) < 0.15 { c[y * K.width + x] = 2 }
            } }
            return c
        }
        let poses = [Pose.identity, Pose(R: matrix_identity_float3x3, t: SIMD3(-0.1, 0, 0)), Pose(R: matrix_identity_float3x3, t: SIMD3(0.1, 0.05, 0))]
        for pose in poses {
            try f.addFrame(pose: pose, intrinsics: K, cue: cue(for: pose), valid: [Bool](repeating: true, count: K.width * K.height))
            for _ in 0..<5 { try f.stepFrame() }
        }
        let map = try f.changeMap(pose: .identity, intrinsics: K), want = cue(for: .identity)
        var tp = 0, fp = 0, fn = 0
        for i in 0..<map.count {
            let pred = map[i] > 0.5, truth = want[i] > 0
            if pred && truth { tp += 1 } else if pred { fp += 1 } else if truth { fn += 1 }
        }
        XCTAssertGreaterThan(Double(tp) / Double(tp + fp + fn), 0.6, "IoU of the fused change mask")
    }
}

/// Minimal INRIA-format PLY writer for test clouds (SH degree 0, isotropic scales).
enum TestPLY {
    struct Splat { var position: SIMD3<Float>; var scale: Float; var opacity: Float }
    static let camera = Intrinsics(f: 120, cx: 79.5, cy: 59.5, width: 160, height: 120)

    /// A 1.6 m × 1.2 m plane of overlapping splats at z = 2, filling the test camera's view.
    static func plane() throws -> URL {
        var splats: [Splat] = []
        for y in stride(from: -0.6, through: 0.6, by: 0.02) { for x in stride(from: -0.8, through: 0.8, by: 0.02) {
            splats.append(Splat(position: SIMD3(Float(x), Float(y), 2), scale: 0.012, opacity: 0.95))
        } }
        return try write(splats)
    }

    static func write(_ splats: [Splat]) throws -> URL {
        let props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                     "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        var header = "ply\nformat binary_little_endian 1.0\nelement vertex \(splats.count)\n"
        for p in props { header += "property float \(p)\n" }
        header += "end_header\n"
        var data = Data(header.utf8)
        for s in splats {
            let logit = log(s.opacity / (1 - s.opacity)), ls = log(s.scale)
            let row: [Float] = [s.position.x, s.position.y, s.position.z, 0, 0, 0, 0.3, 0.3, 0.3, logit, ls, ls, ls, 1, 0, 0, 0]
            row.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("oscd-fusion-\(UUID().uuidString).ply")
        try data.write(to: url)
        return url
    }
}
