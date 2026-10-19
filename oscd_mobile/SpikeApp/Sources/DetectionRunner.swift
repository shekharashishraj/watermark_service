import Combine
import Foundation
import OSCDCore
import UIKit

/// A finished (or saved) Detect run: the summary and one image per frame where the camera was placed.
struct DetectionResult: Identifiable {
    var id: String { summary.afterID + "/" + summary.beforeID }
    let summary: DetectionSummary
    let beforeLabel: String
    let afterLabel: String
    let frames: [URL]               // landscape JPEGs, changed pixels red, unjudged pixels dimmed

    /// A saved run, read back from the after-scan's folder.
    static func load(_ summary: DetectionSummary, beforeLabel: String, afterLabel: String, afterURL: URL) -> DetectionResult {
        let dir = DetectionSummary.folder(in: afterURL, beforeID: summary.beforeID)
        let frames = ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "jpg" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        return DetectionResult(summary: summary, beforeLabel: beforeLabel, afterLabel: afterLabel, frames: frames)
    }
}

/// Runs Method A over a recorded after-scan on one dedicated thread (splatx sessions are single-thread).
final class DetectionRunner: ObservableObject {
    enum StepState: Equatable { case pending, active(String), done(String), failed(String) }
    struct Step: Identifiable { let id: Int; let title: String; var state: StepState }

    @Published private(set) var steps: [Step] = []
    @Published private(set) var latest: UIImage?
    @Published private(set) var progress = 0.0
    @Published private(set) var result: DetectionResult?
    @Published private(set) var error: String?
    @Published private(set) var running = false

    private let lock = NSLock()
    private var cancelled = false

    func start(before: ScanInfo, after: ScanInfo, refine: Bool) {
        var titles = ["Load models and \(before.label)", "Check \(after.label), frame by frame"]
        if refine { titles.append(MethodA.Stage.refine) }
        titles += ["Final change masks", "Save results"]
        steps = titles.enumerated().map { Step(id: $0.offset, title: $0.element, state: .pending) }
        latest = nil; progress = 0; result = nil; error = nil; running = true
        lock.lock(); cancelled = false; lock.unlock()
        let thread = Thread { [weak self] in self?.run(before: before, after: after, refine: refine) }
        thread.name = "oscd.detect"
        thread.stackSize = 16 << 20
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    func cancel() { lock.lock(); cancelled = true; lock.unlock() }

    private var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return cancelled }

    private func set(_ step: Int, _ state: StepState) {
        DispatchQueue.main.async { if step < self.steps.count { self.steps[step].state = state } }
    }

    private func run(before: ScanInfo, after: ScanInfo, refine: Bool) {
        let started = DispatchTime.now().uptimeNanoseconds
        func seconds(since t: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - t) / 1e9 }
        var step = 0
        do {
            set(step, .active("Starting…"))
            let method = try MethodA(beforeScan: before.url, beforeLabel: before.label) { [weak self] s in self?.set(0, .active(s)) }
            var timings = method.setupTimings
            set(step, .done(String(format: "%d anchors · %.1f s", method.anchors.anchorCount, seconds(since: started))))

            step = 1
            let store = try SessionStore(open: after.url)
            let m = store.manifest
            var kept: [(id: Int, rgb: [UInt8], valid: [Bool], pose: Pose)] = []
            let fusionTimes = LockedTimings()
            let checkStart = DispatchTime.now().uptimeNanoseconds
            for (i, r) in m.keyframes.enumerated() {
                if isCancelled { throw PreviewError.message("Cancelled") }
                let rgb = try store.array(.keyframes, id: r.id, name: "rgb").uint8()
                let (depth, conf) = try store.depth(.keyframes, id: r.id)
                let frame = CameraFrame(rgb: rgb, depth: depth, confidence: conf, depthWidth: m.depthWidth,
                                        depthHeight: m.depthHeight, intrinsics: m.intrinsics)
                let t0 = DispatchTime.now().uptimeNanoseconds
                let W = m.intrinsics.width, H = m.intrinsics.height
                let out = try method.process(frame) { [weak self] out, fused in
                    // Fusion thread: this frame's fusion and mask times, and its mask for the progress view.
                    fusionTimes.add(MethodA.Stage.fusion, fused.fusionMs)
                    fusionTimes.add(MethodA.Stage.mask, fused.maskMs)
                    let image = PreviewController.image(PreviewComposite.overlay(rgb8: rgb, changed: fused.changed, valid: out.valid), W, H)
                    DispatchQueue.main.async { self?.latest = image }
                }
                for (stage, ms) in out.timings where ms > 0 { timings.add(stage, ms) }
                if let pose = out.pose {
                    // With fusion overlapped, this is how long a placed frame holds up the next one.
                    timings.add("Per placed frame, end to end", Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
                    kept.append((r.id, rgb, out.valid, pose))
                }
                let done = i + 1
                let perFrame = seconds(since: checkStart) / Double(done)
                DispatchQueue.main.async { self.progress = Double(done) / Double(m.keyframes.count) }
                set(step, .active(String(format: "%d / %d · camera placed in %d · %.1f s per frame",
                                         done, m.keyframes.count, kept.count, perFrame)))
            }
            try method.finishFusion()
            timings.merge(fusionTimes.value)
            set(step, .done(String(format: "camera placed in %d of %d frames · %.0f s", kept.count, m.keyframes.count,
                                   seconds(since: checkStart))))

            if refine {
                step += 1
                set(step, .active("Up to 3000 steps over all frames…"))
                let t = DispatchTime.now().uptimeNanoseconds
                try method.refine()
                timings.add(MethodA.Stage.refine, Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6)
                set(step, .done(String(format: "%.0f s · %d Gaussians", seconds(since: t), method.gaussians)))
            }

            // Every frame's mask from the finished change model, so they all agree.
            step += 1
            let dir = DetectionSummary.folder(in: after.url, beforeID: before.id)
            try? FileManager.default.removeItem(at: dir)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let t = DispatchTime.now().uptimeNanoseconds
            var judged = 0, changed = 0
            var frames: [URL] = []
            for (k, f) in kept.enumerated() {
                if isCancelled { throw PreviewError.message("Cancelled") }
                let mask = try method.changeMask(pose: f.pose)
                judged += f.valid.filter { $0 }.count
                changed += zip(mask, f.valid).filter { $0 && $1 }.count
                let overlay = PreviewComposite.overlay(rgb8: f.rgb, changed: mask, valid: f.valid)
                let url = dir.appendingPathComponent(String(format: "%06d.jpg", f.id))
                if let cg = PreviewController.cgImage(overlay, m.intrinsics.width, m.intrinsics.height),
                   let data = UIImage(cgImage: cg).jpegData(compressionQuality: 0.85) {
                    try data.write(to: url)
                    frames.append(url)
                }
                set(step, .active("\(k + 1) / \(kept.count)"))
            }
            timings.add(MethodA.Stage.finalMasks, Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6)
            set(step, .done(String(format: "%d frames · %.1f s", kept.count, seconds(since: t))))

            step += 1
            let summary = DetectionSummary(beforeID: before.id, afterID: after.id, method: "A", refined: refine,
                                           frames: m.keyframes.count, framesWithPose: kept.count,
                                           judgedPixels: judged, changedPixels: changed,
                                           totalSeconds: seconds(since: started), timings: timings)
            try summary.save(in: after.url)
            set(step, .done(String(format: "%.0f s in total", summary.totalSeconds)))
            let result = DetectionResult(summary: summary, beforeLabel: before.label, afterLabel: after.label, frames: frames)
            DispatchQueue.main.async { self.result = result; self.running = false }
        } catch {
            let message = error.localizedDescription
            set(step, .failed(message))
            DispatchQueue.main.async { self.error = message; self.running = false }
        }
    }
}

/// Stage times written from the fusion thread and read once the run's fusion has finished.
private final class LockedTimings {
    private let lock = NSLock()
    private var t = StageTimings()
    func add(_ stage: String, _ ms: Double) { lock.lock(); t.add(stage, ms); lock.unlock() }
    var value: StageTimings { lock.lock(); defer { lock.unlock() }; return t }
}
