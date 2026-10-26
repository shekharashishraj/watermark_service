import XCTest
import simd
@testable import OSCDCore

final class StoreTests: XCTestCase {
    func manifest() -> SessionManifest {
        SessionManifest(deviceModel: "test", captureWidth: 8, captureHeight: 4, depthWidth: 2, depthHeight: 1,
                        intrinsics: Intrinsics(f: 3, cx: 1.5, cy: 0.5, width: 4, height: 2))
    }

    func temp() -> URL { FileManager.default.temporaryDirectory.appendingPathComponent("oscd-\(UUID().uuidString)") }

    func testRoundTrip() throws {
        let root = temp()
        let store = try SessionStore(create: root, manifest: manifest())
        try store.setROI(AABB(min: SIMD3(-1, 0, -1), max: SIMD3(1, 0.5, 1)))
        let rec = FrameRecord(id: 0, timestamp: 1.5, arkitCameraToWorld: matrix_identity_float4x4.columnMajor)
        try store.append(.keyframes, record: rec, rgb: [UInt8](repeating: 255, count: 4 * 2 * 3), depth: [0.5, 0.6], confidence: [2, 1])
        let back = try SessionStore(open: root)
        XCTAssertEqual(back.manifest.keyframes, [rec])
        XCTAssertEqual(back.manifest.roi?.max.y, 0.5)
        XCTAssertEqual(try back.rgb(.keyframes, id: 0).data.max(), 1)
        XCTAssertEqual(try back.depth(.keyframes, id: 0).confidence, [2, 1])
    }

    /// Review Focus 5: files written for a frame the manifest never recorded are ignored, and a
    /// listed frame with a missing file throws a clear error instead of crashing.
    func testPartialWriteIsIgnoredAndMissingFileThrows() throws {
        let root = temp()
        let store = try SessionStore(create: root, manifest: manifest())
        let rec = FrameRecord(id: 0, timestamp: 0, arkitCameraToWorld: matrix_identity_float4x4.columnMajor)
        try store.append(.keyframes, record: rec, rgb: [UInt8](repeating: 0, count: 24), depth: [1, 1], confidence: [2, 2])
        try NPYArray(uint8: [UInt8](repeating: 0, count: 24), shape: [2, 4, 3])
            .write(to: root.appendingPathComponent("keyframes/000001_rgb.npy"))          // orphan, as after a crash
        let back = try SessionStore(open: root)
        XCTAssertEqual(back.manifest.keyframes.map(\.id), [0])
        try FileManager.default.removeItem(at: root.appendingPathComponent("keyframes/000000_depth.npy"))
        XCTAssertThrowsError(try back.depth(.keyframes, id: 0))
        XCTAssertThrowsError(try SessionStore(open: temp()))
    }

    /// Feeds frames as (time, x, sharpness[, tracking normal]); the candidate is the frame's index.
    private func feed(_ s: inout KeyframeSelector<Int>, _ frames: [(Double, Float, Double)], normal: [Bool]? = nil,
                      rotationDeg: [Float]? = nil, scored: inout [Int]) -> [KeyframeSelector<Int>.Chosen] {
        var out: [KeyframeSelector<Int>.Chosen] = []
        for (i, f) in frames.enumerated() {
            let R = simd_float3x3(simd_quatf(angle: (rotationDeg?[i] ?? 0) * .pi / 180, axis: SIMD3(0, 1, 0)))
            let pose = Pose(R: R, t: -(R * SIMD3(f.1, 0, 0)))            // centre at (x, 0, 0)
            if let c = s.consider(pose: pose, time: f.0, trackingNormal: normal?[i] ?? true,
                                  sharpness: { scored.append(i); return f.2 }, candidate: { i }) { out.append(c) }
        }
        if let c = s.finish() { out.append(c) }
        return out
    }

    func testKeyframeWindowKeepsTheSharpestFrame() {
        var s = KeyframeSelector<Int>(), scored: [Int] = []
        let chosen = feed(&s, [(0, 0, 5), (0.05, 0.001, 9), (0.10, 0.002, 7),
                               (0.30, 0.004, 1),                          // closes the window; 3 mm from the keyframe
                               (0.60, 0.03, 2)], scored: &scored)         // 3 cm on: a new window
        XCTAssertEqual(chosen.map(\.candidate), [1, 4])
        XCTAssertEqual(chosen.map(\.candidates), [3, 1])
        XCTAssertEqual(chosen[0].sharpness, 9)
        XCTAssertEqual(chosen.map(\.firstSharpness), [5, 2])
        XCTAssertEqual(scored, [0, 1, 2, 4])                              // frame 3 was never scored
    }

    func testKeyframeWindowHoldsAtMostFourFrames() {
        var s = KeyframeSelector<Int>(), scored: [Int] = []
        let frames = (0..<5).map { (Double($0) * 0.01, Float($0) * 0.001, Double($0)) }
        let chosen = feed(&s, frames, scored: &scored)
        XCTAssertEqual(chosen.map(\.candidate), [3])
        XCTAssertEqual(chosen.map(\.candidates), [4])
    }

    func testKeyframeGateRejectsFastAndUntrackedFrames() {
        var s = KeyframeSelector<Int>(), scored: [Int] = []
        let chosen = feed(&s, [(0, 0, 1), (0.1, 0.1, 1), (0.2, 0.11, 1)], normal: [false, true, true], scored: &scored)
        XCTAssertEqual(chosen.map(\.candidate), [2])                     // frame 1 moved at 1 m/s; frame 2 at 0.55 m/s over 0.2 s
        XCTAssertEqual(scored, [2])
    }

    func testKeyframeGateAdmitsWalkingPace() {
        var s = KeyframeSelector<Int>(), scored: [Int] = []
        let chosen = feed(&s, [(0, 0, 1), (0.1, 0.05, 2)], scored: &scored)   // 0.5 m/s, 5 cm on
        XCTAssertEqual(scored, [0, 1])
        XCTAssertEqual(chosen.map(\.candidate), [1])
    }

    func testKeyframeGateAdmitsASlowTurnOnTheSpot() {
        var s = KeyframeSelector<Int>(), scored: [Int] = []
        let chosen = feed(&s, [(0, 0, 1), (0.3, 0, 1), (1.3, 0, 1)], rotationDeg: [0, 0, 12], scored: &scored)   // 12° at 12°/s (trigger is 2°)
        XCTAssertEqual(chosen.map(\.candidate), [0, 2])
    }

    /// A turn at 45°/s is blurred: no keyframe until it slows, even though the 10° trigger fired. The
    /// speed is averaged over the last 0.25 s of frames, so a single jittery 60 fps interval does not decide.
    func testKeyframeGateWaitsForTheTurnToSlow() {
        var s = KeyframeSelector<Int>(), scored: [Int] = []
        // 60 fps: 0.75°/frame = 45°/s for 24 frames (18°), then 0.1°/frame = 6°/s for 36 more.
        var frames: [(Double, Float, Double)] = [], rot: [Float] = []
        var angle: Float = 0
        for i in 0..<60 {
            frames.append((Double(i) / 60, 0, 1)); rot.append(angle)
            angle += i < 24 ? 0.75 : 0.1
        }
        let chosen = feed(&s, frames, rotationDeg: rot, scored: &scored)
        XCTAssertGreaterThanOrEqual(chosen.count, 2)
        XCTAssertEqual(chosen[0].candidate, 0)
        XCTAssertTrue(chosen.dropFirst().allSatisfy { $0.candidate >= 24 })   // the rest only once the turn slowed
        XCTAssertTrue(scored.allSatisfy { $0 == 0 || $0 >= 24 })          // no fast-turn frame was even scored
    }

    /// One jittery interval (a 1° hop in one frame = 60°/s) inside an otherwise slow pan does not refuse the frame.
    func testKeyframeGateIgnoresASingleJitteryFrame() {
        var s = KeyframeSelector<Int>(), scored: [Int] = []
        var frames: [(Double, Float, Double)] = [], rot: [Float] = []
        var angle: Float = 0
        for i in 0..<70 {
            frames.append((Double(i) / 60, 0, 1)); rot.append(angle)
            angle += i == 52 ? 1.0 : 0.2                                  // 12°/s; one 60°/s hop at frame 52
        }
        _ = feed(&s, frames, rotationDeg: rot, scored: &scored)
        // The 2° trigger fires at frame 10; windows of 8 then run back to back, so frames 50…57 are scored too.
        XCTAssertTrue(scored.contains(53) && scored.contains(54), "\(scored)")
    }

    func testMovingTooFastIsReported() {
        var s = KeyframeSelector<Int>()
        let R = matrix_identity_float3x3
        _ = s.consider(pose: Pose(R: R, t: .zero), time: 0, trackingNormal: true, sharpness: { 1 }, candidate: { 0 })
        XCTAssertFalse(s.movingTooFast)
        _ = s.consider(pose: Pose(R: R, t: SIMD3(-0.1, 0, 0)), time: 0.1, trackingNormal: true, sharpness: { 1 }, candidate: { 1 })
        XCTAssertTrue(s.movingTooFast)                                     // 1 m/s
    }

    /// Blur flattens detail finer than the blur, so the gradient energy of a texture drops. (A single clean
    /// edge would not show it: blur turns one big step into several small ones with the same total.)
    func testSharpnessDropsWhenATextureIsBlurred() {
        let w = 64, h = 64, r = 3
        var rng = SeededSampler(seed: 4)
        let sharp = (0..<(w * h)).map { _ in UInt8(255 * rng.uniform()) }
        var blurred = sharp
        for y in r..<(h - r) {
            for x in r..<(w - r) {
                var sum = 0
                for dy in -r...r { for dx in -r...r { sum += Int(sharp[(y + dy) * w + x + dx]) } }
                blurred[y * w + x] = UInt8(sum / ((2 * r + 1) * (2 * r + 1)))
            }
        }
        XCTAssertEqual(Sharpness.score(luma: [UInt8](repeating: 90, count: w * h), width: w, height: h), 0)
        XCTAssertGreaterThan(Sharpness.score(luma: sharp, width: w, height: h),
                             3 * Sharpness.score(luma: blurred, width: w, height: h))
    }

    func testCameraImageConversion() {
        // 4×4 luma, 2×2 interleaved CbCr, factor 2 → 2×2 RGB.
        let gray = CameraImage.rgbArea(y: [UInt8](repeating: 128, count: 16), yStride: 4, cbcr: [UInt8](repeating: 128, count: 8), cbcrStride: 4,
                                       width: 4, height: 4, factor: 2)
        XCTAssertEqual(gray, [UInt8](repeating: 128, count: 12))
        var cbcr = [UInt8](repeating: 0, count: 8)
        for i in stride(from: 0, to: 8, by: 2) { cbcr[i] = 85; cbcr[i + 1] = 255 }        // pure red in BT.601 full range
        let red = CameraImage.rgbArea(y: [UInt8](repeating: 76, count: 16), yStride: 4, cbcr: cbcr, cbcrStride: 4, width: 4, height: 4, factor: 2)
        XCTAssertGreaterThanOrEqual(red[0], 250); XCTAssertLessThanOrEqual(red[1], 3); XCTAssertLessThanOrEqual(red[2], 3)
    }

    /// Review fix: a capture size that is not an exact integer multiple of 480×360 must be rejected,
    /// not crash the recorder when the 480×360 RGB array is written.
    func testAreaFactorAcceptsOnlyExactMultiples() {
        XCTAssertEqual(CameraImage.areaFactor(captureWidth: 1920, captureHeight: 1440, workingWidth: 480, workingHeight: 360), 4)
        XCTAssertNil(CameraImage.areaFactor(captureWidth: 1920, captureHeight: 1080, workingWidth: 480, workingHeight: 360))
        XCTAssertNil(CameraImage.areaFactor(captureWidth: 1000, captureHeight: 750, workingWidth: 480, workingHeight: 360))
        XCTAssertNil(CameraImage.areaFactor(captureWidth: 240, captureHeight: 180, workingWidth: 480, workingHeight: 360))
    }
}
