import CoreML
import Foundation
import OSCDCore
import simd

struct BenchRow: Identifiable {
    let id = UUID()
    let stage: String
    let medianMs: Double
    let runs: Int
    let note: String
}

/// Times every Phase 1 stage on the device and writes Documents/benchmarks/bench_<date>.csv.
final class BenchmarkRunner: ObservableObject {
    @Published var rows: [BenchRow] = []
    @Published var running = false
    @Published var message = ""

    func run() {
        running = true
        rows = []
        message = "Running…"
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let rows = try Self.measureAll { msg in DispatchQueue.main.async { self.message = msg } }
                DispatchQueue.main.async { self.rows = rows; self.running = false; self.message = "Saved to Documents/benchmarks" }
            } catch {
                DispatchQueue.main.async { self.running = false; self.message = "Failed: \(error.localizedDescription)" }
            }
        }
    }

    static func median(_ runs: Int, warmup: Int = 2, _ body: () throws -> Void) rethrows -> Double {
        for _ in 0..<warmup { try body() }
        var times: [Double] = []
        for _ in 0..<runs {
            let start = DispatchTime.now().uptimeNanoseconds
            try body()
            times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
        }
        return times.sorted()[times.count / 2]
    }

    static func measureAll(progress: (String) -> Void) throws -> [BenchRow] {
        guard let xfeatURL = Bundle.main.url(forResource: "XFeat_480x352", withExtension: "mlmodelc"),
              let samURL = Bundle.main.url(forResource: "SAM2_1Tiny_2x1024", withExtension: "mlmodelc") else {
            throw NSError(domain: "bench", code: 1, userInfo: [NSLocalizedDescriptionKey:
                "Models are missing from the app. Run the two converters, then `xcodegen generate` again."])
        }
        var rows: [BenchRow] = []
        let session = try latestSession()
        let image: FloatImage
        if let session, let first = session.manifest.keyframes.first {
            image = try session.rgb(.keyframes, id: first.id)
        } else {
            image = FloatImage(width: 480, height: 360, channels: 3, fill: 0.5)
        }

        progress("XFeat")
        let xfeat = try XFeatModel(compiledModelURL: xfeatURL)
        var kp = DescribedKeypoints(kpts: [], feats: [])
        let xfeatMs = try median(20) { kp = try xfeat.detect(image) }
        rows.append(BenchRow(stage: "xfeat_detect", medianMs: xfeatMs, runs: 20, note: "\(kp.count) keypoints"))

        progress("SAM2")
        let sam = try SAM2Encoder(compiledModelURL: samURL)
        rows.append(BenchRow(stage: "sam2_encode_pair", medianMs: try median(10) { _ = try sam.encode(render: image, live: image) }, runs: 10, note: "batch of 2 at 1024²"))

        progress("Retrieval")
        let frame = kp
        let refs = [DescribedKeypoints](repeating: frame, count: 80)
        rows.append(BenchRow(stage: "retrieval_80_keyframes", medianMs: median(3, warmup: 1) {
            var counts = [Int](repeating: 0, count: refs.count)
            counts.withUnsafeMutableBufferPointer { c in
                DispatchQueue.concurrentPerform(iterations: refs.count) { c[$0] = Matcher.countMutual(refs[$0], frame) }
            }
        }, runs: 3, note: "\(frame.count) keypoints per image"))

        progress("P4P-RANSAC and mini-BA")
        let (uv, xyz, start) = syntheticCorrespondences(2000)
        let centre = SIMD2<Double>(239.5, 179.5)
        rows.append(BenchRow(stage: "p4p_ransac_2000", medianMs: median(5, warmup: 1) {
            var s = SeededSampler(seed: 0)
            _ = P4PRANSAC.run(uv: uv, xyz: xyz, confs: [Double](repeating: 1, count: uv.count), f: 360, centre: centre,
                              initR6: start.r6, initT: start.tDouble, maxError: 1.5, sampler: &s)
        }, runs: 5, note: "2000 hypotheses × 20 LM"))
        rows.append(BenchRow(stage: "mini_ba_2000", medianMs: median(5, warmup: 1) {
            _ = MiniBA.solve(r6: start.r6, t: start.tDouble, f: 360, centre: centre, xyz: xyz, uv: uv, config: .poseRefine)
        }, runs: 5, note: "20 iterations"))

        progress("Pixel cue")
        rows.append(BenchRow(stage: "pixel_cue_ssim", medianMs: median(5, warmup: 1) { _ = CuesA.pixel(live: image, render: image) }, runs: 5, note: "480×360"))

        if let session, session.manifest.keyframes.count >= 20 {
            progress("Anchors on the latest session")
            var views: [(pose: Pose, keypoints: DescribedKeypoints)] = []
            let t0 = DispatchTime.now().uptimeNanoseconds
            for r in session.manifest.keyframes {
                let img = try session.rgb(.keyframes, id: r.id)
                views.append((Pose.fromARKitCameraToWorld(simd_float4x4(columnMajor: r.arkitCameraToWorld)), try xfeat.detect(img)))
            }
            let t1 = DispatchTime.now().uptimeNanoseconds
            let anchors = AnchorBuilder.build(keyframes: views, intrinsics: session.manifest.intrinsics)
            let t2 = DispatchTime.now().uptimeNanoseconds
            rows.append(BenchRow(stage: "xfeat_all_keyframes", medianMs: Double(t1 - t0) / 1e6, runs: 1, note: "\(views.count) keyframes"))
            rows.append(BenchRow(stage: "anchor_build", medianMs: Double(t2 - t1) / 1e6, runs: 1,
                                 note: String(format: "yield %.3f, %d anchors", anchors.yield, anchors.anchorCount)))
        }
        try writeCSV(rows)
        return rows
    }

    static func latestSession() throws -> SessionStore? {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("sessions")
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: dir.path), let last = names.sorted().last else { return nil }
        return try SessionStore(open: dir.appendingPathComponent(last))
    }

    static func syntheticCorrespondences(_ n: Int) -> ([SIMD2<Double>], [SIMD3<Double>], Pose) {
        var s = SeededSampler(seed: 7)
        let truth = Pose(R: simd_float3x3(simd_quatf(angle: 0.4, axis: SIMD3(0, 1, 0))), t: SIMD3(0.05, 0, 0.8))
        let K = Intrinsics(f: 360, cx: 239.5, cy: 179.5, width: 480, height: 360)
        var uv: [SIMD2<Double>] = [], xyz: [SIMD3<Double>] = []
        for i in 0..<n {
            let X = SIMD3(Float(s.uniform()) - 0.5, Float(s.uniform()) * 0.3, Float(s.uniform()) - 0.5)
            var p = K.project(truth.apply(X))
            if i % 3 == 0 { p = SIMD2(Float(s.uniform()) * 480, Float(s.uniform()) * 360) }
            uv.append(SIMD2<Double>(p))
            xyz.append(SIMD3<Double>(X))
        }
        let start = Pose(R: truth.R * simd_float3x3(simd_quatf(angle: 0.08, axis: SIMD3(1, 0, 0))), t: truth.t + SIMD3(0.03, 0, 0))
        return (uv, xyz, start)
    }

    static func writeCSV(_ rows: [BenchRow]) throws {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("benchmarks")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let device = CaptureController.deviceModel()
        var csv = "stage,median_ms,runs,note,device\n"
        for r in rows { csv += "\(r.stage),\(String(format: "%.2f", r.medianMs)),\(r.runs),\"\(r.note)\",\(device)\n" }
        let name = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        try csv.write(to: dir.appendingPathComponent("bench_\(name).csv"), atomically: true, encoding: .utf8)
    }
}
