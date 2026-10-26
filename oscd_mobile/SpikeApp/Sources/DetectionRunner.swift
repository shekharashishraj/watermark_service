import Combine
import Foundation
import OSCDCore
import simd
import UIKit

/// A finished (or saved) Detect run: the summary and one image per frame where the camera was placed,
/// for Method A and, when it ran, Method B over the same frames.
struct DetectionResult: Identifiable {
    struct Run {
        let summary: DetectionSummary
        let frames: [URL]           // landscape JPEGs, changed pixels red, unjudged pixels dimmed
    }

    var id: String { a.summary.afterID + "/" + a.summary.beforeID }
    let a: Run
    let b: Run?
    let beforeLabel: String
    let afterLabel: String
    /// The objects found over the final masks (method B's when B ran, else A's); nil for runs saved before it existed.
    let objects: ObjectReport?
    /// The before-scan, when it is still on the phone.
    let before: ScanInfo?
    var summary: DetectionSummary { a.summary }

    /// A saved run, read back from the after-scan's folder.
    static func load(_ summary: DetectionSummary, before: ScanInfo?, beforeLabel: String, afterLabel: String,
                     afterURL: URL) -> DetectionResult {
        func frames(_ method: String) -> [URL] {
            let dir = DetectionSummary.folder(in: afterURL, beforeID: summary.beforeID, method: method)
            return ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
                .filter { $0.pathExtension == "jpg" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        }
        let b = DetectionSummary.load(in: afterURL, beforeID: summary.beforeID, method: "B").map { Run(summary: $0, frames: frames("B")) }
        let objects = ObjectReport.load(in: afterURL, beforeID: summary.beforeID, method: b != nil ? "B" : "A")
        return DetectionResult(a: Run(summary: summary, frames: frames("A")), b: b, beforeLabel: beforeLabel, afterLabel: afterLabel,
                               objects: objects, before: before)
    }
}

/// Runs Method A (and optionally B) over a recorded after-scan on one dedicated thread (splatx sessions
/// are single-thread; fusion runs on its own threads).
final class DetectionRunner: ObservableObject {
    /// The object post-pass's step and its stage time in `summary.json`.
    static let findObjects = "Find objects"
    static let cacheFeatures = "Cache SAM2 features (fp16)"

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

    func start(before: ScanInfo, after: ScanInfo, refine: Bool, withB: Bool = false) {
        var titles = ["Load models and \(before.label)", "Check \(after.label), frame by frame" + (withB ? " (A and B)" : "")]
        if refine { titles.append(MethodA.Stage.refine) }
        titles += ["Final change masks", Self.findObjects, "Save results"]
        steps = titles.enumerated().map { Step(id: $0.offset, title: $0.element, state: .pending) }
        latest = nil; progress = 0; result = nil; error = nil; running = true
        lock.lock(); cancelled = false; lock.unlock()
        let thread = Thread { [weak self] in self?.run(before: before, after: after, refine: refine, withB: withB) }
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

    private func run(before: ScanInfo, after: ScanInfo, refine: Bool, withB: Bool) {
        let started = DispatchTime.now().uptimeNanoseconds
        func seconds(since t: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - t) / 1e9 }
        var step = 0
        do {
            set(step, .active("Starting…"))
            let method = try MethodA(beforeScan: before.url, beforeLabel: before.label, withB: withB, withDepth: true) { [weak self] s in
                self?.set(0, .active(s))
            }
            var timings = method.setupTimings
            set(step, .done(String(format: "%d anchors · %.1f s", method.anchors.anchorCount, seconds(since: started))))

            step = 1
            let store = try SessionStore(open: after.url)
            let m = store.manifest
            let W = m.intrinsics.width, H = m.intrinsics.height
            var kept: [Kept] = []
            let fusionTimes = LockedTimings()
            // Each placed frame's SAM2 features, for the object post-pass; deleted however the run ends.
            let cache = try EmbeddingCache(runID: "\(after.id)_\(before.id)")
            defer { cache.removeAll() }
            let checkStart = DispatchTime.now().uptimeNanoseconds
            for (i, r) in m.keyframes.enumerated() {
                if isCancelled { throw PreviewError.message("Cancelled") }
                let rgb = try store.array(.keyframes, id: r.id, name: "rgb").uint8()
                let (depth, conf) = try store.depth(.keyframes, id: r.id)
                let frame = CameraFrame(rgb: rgb, depth: depth, confidence: conf, depthWidth: m.depthWidth,
                                        depthHeight: m.depthHeight, intrinsics: m.intrinsics)
                let t0 = DispatchTime.now().uptimeNanoseconds
                let id = r.id
                let out = try method.process(frame, fused: { [weak self] out, fused, context in
                    // Fusion thread: this frame's fusion and mask times, its features for the object post-pass,
                    // and its mask for the progress view.
                    fusionTimes.add(MethodA.Stage.fusion, fused.fusionMs)
                    fusionTimes.add(MethodA.Stage.mask, fused.maskMs)
                    let tc = DispatchTime.now().uptimeNanoseconds
                    do {
                        try cache.write(frameID: id, embLive: context.embLive, embRender: context.embRender)
                    } catch {
                        AppLog.write("detect: frame \(id)'s SAM2 features were not cached: \(error.localizedDescription)")
                    }
                    fusionTimes.add(Self.cacheFeatures, Double(DispatchTime.now().uptimeNanoseconds - tc) / 1e6)
                    let image = PreviewController.image(PreviewComposite.overlay(rgb8: rgb, changed: fused.changed, valid: out.valid), W, H)
                    DispatchQueue.main.async { self?.latest = image }
                }, fusedB: { _, fused, _ in
                    fusionTimes.add("B: " + MethodA.Stage.fusion, fused.fusionMs)
                    fusionTimes.add("B: " + MethodA.Stage.mask, fused.maskMs)
                })
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

            let methods = method.hasB ? ["A", "B"] : ["A"]
            if refine {
                step += 1
                set(step, .active("Up to 3000 steps over all frames…"))
                let t = DispatchTime.now().uptimeNanoseconds
                for name in methods {
                    let tm = DispatchTime.now().uptimeNanoseconds
                    try method.refine(method: name)
                    timings.add((name == "A" ? "" : "B: ") + MethodA.Stage.refine, Double(DispatchTime.now().uptimeNanoseconds - tm) / 1e6)
                }
                set(step, .done(String(format: "%.0f s · %d Gaussians", seconds(since: t), method.gaussians())))
            }

            // Every frame's mask from the finished change model(s), so they all agree.
            step += 1
            try? FileManager.default.removeItem(at: DetectionSummary.folder(in: after.url, beforeID: before.id))
            let t = DispatchTime.now().uptimeNanoseconds
            var runs: [String: (judged: Int, changed: Int, frames: [URL])] = [:]
            for name in methods {
                let dir = DetectionSummary.folder(in: after.url, beforeID: before.id, method: name)
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                var judged = 0, changed = 0
                var frames: [URL] = []
                for (k, f) in kept.enumerated() {
                    if isCancelled { throw PreviewError.message("Cancelled") }
                    let mask = try method.changeMask(pose: f.pose, method: name)
                    judged += f.valid.filter { $0 }.count
                    changed += zip(mask, f.valid).filter { $0 && $1 }.count
                    let overlay = PreviewComposite.overlay(rgb8: f.rgb, changed: mask, valid: f.valid)
                    let url = dir.appendingPathComponent(String(format: "%06d.jpg", f.id))
                    if let cg = PreviewController.cgImage(overlay, W, H), let data = UIImage(cgImage: cg).jpegData(compressionQuality: 0.85) {
                        try data.write(to: url)
                        frames.append(url)
                    }
                    set(step, .active("\(name): \(k + 1) / \(kept.count)"))
                }
                runs[name] = (judged, changed, frames)
            }
            timings.add(MethodA.Stage.finalMasks, Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6)
            set(step, .done(String(format: "%d frames · %.1f s", kept.count, seconds(since: t))))

            step += 1
            let objects = try findObjects(method: method, after: store, afterURL: after.url, kept: kept, cache: cache,
                                          beforeID: before.id, afterID: after.id, step: step)
            cache.removeAll()
            timings.add(Self.findObjects, objects.timingsMs[Self.findObjects] ?? 0)

            step += 1
            func summary(_ name: String) -> DetectionSummary {
                DetectionSummary(beforeID: before.id, afterID: after.id, method: name, refined: refine,
                                 frames: m.keyframes.count, framesWithPose: kept.count,
                                 judgedPixels: runs[name]!.judged, changedPixels: runs[name]!.changed,
                                 totalSeconds: seconds(since: started), timings: timings)
            }
            let a = summary("A")
            let b = method.hasB ? summary("B") : nil
            try b?.save(in: after.url)
            try a.save(in: after.url)
            try objects.save(in: after.url)
            set(step, .done(String(format: "%.0f s in total", a.totalSeconds)))
            let result = DetectionResult(a: .init(summary: a, frames: runs["A"]!.frames),
                                         b: b.map { .init(summary: $0, frames: runs["B"]!.frames) },
                                         beforeLabel: before.label, afterLabel: after.label, objects: objects, before: before)
            DispatchQueue.main.async { self.result = result; self.running = false }
        } catch {
            let message = error.localizedDescription
            set(step, .failed(message))
            DispatchQueue.main.async { self.error = message; self.running = false }
        }
    }

    private typealias Kept = (id: Int, rgb: [UInt8], valid: [Bool], pose: Pose)

    /// Spec §4.2: the object tracker over the kept frames in order, each with its final mask (B's when B ran),
    /// so the object list agrees with the masks saved beside it. D_live comes from the after-scan's depth, D_ren
    /// and the reference render are rendered again at the frame's pose, and the SAM2 features come from the cache
    /// the fusion callbacks wrote. Then one PlanarICP fit per matched pair. Runs on the detect thread (it owns
    /// `method`'s renderers).
    private func findObjects(method: MethodA, after store: SessionStore, afterURL: URL, kept: [Kept], cache: EmbeddingCache,
                             beforeID: String, afterID: String, step: Int) throws -> ObjectReport {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let m = store.manifest
        let W = m.intrinsics.width, H = m.intrinsics.height
        let name = method.hasB ? "B" : "A"
        var config = TrackerConfig()
        config.roi = method.roi
        config.confirmVoxels = ObjectSettings.minVoxels
        // The table's height from the before-scan's LiDAR: changes lying on it are the sheet, not objects.
        let ts = DispatchTime.now().uptimeNanoseconds
        config.supportHeight = try SupportPlane.height(of: method.beforeStore, roi: method.roi)
        AppLog.write(config.supportHeight.map {
            String(format: "semantic: support y = %.3f m (%.0f ms)", $0, Double(DispatchTime.now().uptimeNanoseconds - ts) / 1e6)
        } ?? "semantic: support y = unknown (no LiDAR points in the ROI)")
        // The before-scan's own LiDAR, splatted into every frame as the reference depth (spec §4.4, C1).
        let tb = DispatchTime.now().uptimeNanoseconds
        let beforePoints = try BeforeDepth.points(of: method.beforeStore, roi: method.roi)
        AppLog.write(String(format: "semantic: before-scan LiDAR · %d points in the ROI · %.0f ms%@", beforePoints.count,
                            Double(DispatchTime.now().uptimeNanoseconds - tb) / 1e6,
                            beforePoints.isEmpty ? " · none: no object will be reported" : ""))
        var splatMs: [Double] = []
        let tracker = ObjectTracker(config: config)
        var ingestMs: [Double] = []
        var uncached = 0
        let details = UserDefaults.standard.bool(forKey: "detectObjects")
        // `-dumpFrames YES`: every frame the tracker ingests, written as FrameContextIO folders for `oscd-objects` on the Mac.
        // The folder is cleared first so a rerun never mixes frames of two runs.
        let dumpDir: URL? = UserDefaults.standard.bool(forKey: "dumpFrames")
            ? DetectionSummary.folder(in: afterURL, beforeID: beforeID).appendingPathComponent("frames", isDirectory: true) : nil
        if let dumpDir { try? FileManager.default.removeItem(at: dumpDir) }
        var dumped = 0, dumpMs = 0.0
        if details {
            let r = method.roi
            AppLog.write(String(format: "objects-roi: (%.3f, %.3f, %.3f)–(%.3f, %.3f, %.3f) · %d kept frames",
                                r.min.x, r.min.y, r.min.z, r.max.x, r.max.y, r.max.z, kept.count))
        }
        for (k, f) in kept.enumerated() {
            if isCancelled { throw PreviewError.message("Cancelled") }
            let emb: (embLive: [Float], embRender: [Float])
            do {
                emb = try cache.read(frameID: f.id)
            } catch {
                uncached += 1
                AppLog.write("detect: frame \(f.id) skipped for objects (no cached SAM2 features): \(error.localizedDescription)")
                continue
            }
            let (depth, conf) = try store.depth(.keyframes, id: f.id)
            let dLive = CuesB.upsampleDepth(depth, confidence: conf, width: m.depthWidth, height: m.depthHeight, toWidth: W, toHeight: H)
            let dRen = try method.expectedDepth(pose: f.pose) ?? [Float](repeating: .nan, count: W * H)
            let mask = try method.changeMask(pose: f.pose, method: name)
            let tsp = DispatchTime.now().uptimeNanoseconds
            let dBefore = BeforeDepth.splat(points: beforePoints, pose: f.pose, intrinsics: method.anchors.intrinsics)
            splatMs.append(Double(DispatchTime.now().uptimeNanoseconds - tsp) / 1e6)
            let context = FrameContext(frameID: f.id, pose: f.pose, intrinsics: method.anchors.intrinsics,
                                       changed: zip(mask, f.valid).map { $0 && $1 }, valid: f.valid, dLive: dLive, dRen: dRen,
                                       live: FloatImage(rgb8: f.rgb, width: W, height: H), render: try method.render(pose: f.pose),
                                       embLive: emb.embLive, embRender: emb.embRender, dBefore: dBefore)
            if let dumpDir {
                let td = DispatchTime.now().uptimeNanoseconds
                do {
                    try FrameContextIO.write(context, to: dumpDir, supportHeight: config.supportHeight, roi: config.roi, keyframeID: f.id)
                    dumped += 1
                } catch {
                    AppLog.write("detect: frame \(f.id) was not dumped: \(error.localizedDescription)")
                }
                dumpMs += Double(DispatchTime.now().uptimeNanoseconds - td) / 1e6
            }
            let previous = details ? tracker.instances : []
            let stats = tracker.ingest(context)
            ingestMs.append(stats.milliseconds)
            if details { Self.logFrame(context, stats, previous: previous, tracker) }
            set(step, .active("\(k + 1) / \(kept.count) · \(tracker.instances.count) candidates"))
        }
        // A removed object has no live pixels: its feature comes from the before-scan's own photo that sees it best
        // (spec §4.6), set before matching so `fitMotions` and the report use it.
        let tp = DispatchTime.now().uptimeNanoseconds
        var photoCache: [Int: [Float]] = [:]
        var photoFeatures = 0
        for l in tracker.instances where l.sign < 0 && l.isConfirmed(config) {
            if isCancelled { throw PreviewError.message("Cancelled") }
            set(step, .active("Looking at object \(l.id) in the before-scan's photos…"))
            do {
                guard let p = try BeforePhotoFeature.pooled(for: l.points, store: method.beforeStore, cache: &photoCache,
                                                            encode: { try method.encodeSAM2($0) }) else {
                    AppLog.write("detect: left object \(l.id) is in none of the before-scan's photos; its feature stays the render's")
                    continue
                }
                tracker.setFeature(instance: l.id, p.feature, source: .beforePhoto)
                tracker.setBeforePhoto(instance: l.id, keyframe: p.record.id, rect: p.rect)
                photoFeatures += 1
            } catch {
                AppLog.write("detect: left object \(l.id)'s before-scan photo feature failed: \(error.localizedDescription)")
            }
        }
        let photoMs = Double(DispatchTime.now().uptimeNanoseconds - tp) / 1e6
        AppLog.write(String(format: "detect: before-photo features · %d left objects · %d photos encoded · %.0f ms", photoFeatures,
                            photoCache.count, photoMs))
        // Vision names every confirmed instance (Settings → Name objects). A left object is read from the full-size
        // before-scan photo chosen above; the others from the best crop the tracker kept.
        var namingMs = 0.0
        if ObjectNamer.enabled {
            let tn = DispatchTime.now().uptimeNanoseconds
            let namer = ObjectNamer()
            var named = 0
            for i in tracker.instances where i.isConfirmed(config) {
                if isCancelled { throw PreviewError.message("Cancelled") }
                set(step, .active("Naming object \(i.id)…"))
                let image: CGImage?
                if i.sign < 0, let k = i.bestBeforeKeyframe, let rect = i.bestBeforeRect {
                    image = ObjectNamer.beforeCrop(store: method.beforeStore, keyframeID: k, rect: rect)
                } else {
                    image = i.bestCrop.flatMap(ObjectNamer.cgImage)
                }
                if let image, let n = namer.name(image) {
                    tracker.setLabel(instance: i.id, n.label, confidence: n.confidence)
                    named += 1
                    AppLog.write(String(format: "naming: object %d (sign %+d) → %@ %.2f", i.id, Int(i.sign), n.label, n.confidence))
                } else {
                    AppLog.write("naming: object \(i.id) (sign \(i.sign)) has no label")
                }
            }
            namingMs = Double(DispatchTime.now().uptimeNanoseconds - tn) / 1e6
            AppLog.write(String(format: "detect: naming · %d named · %.0f ms", named, namingMs))
        }
        let fits = tracker.fitMotions(willFit: { set(step, .active("Measuring how object \($0.leftID) moved…")) },
                                      stop: { isCancelled })
        if isCancelled { throw PreviewError.message("Cancelled") }
        let totalMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6 - dumpMs
        if dumpDir != nil {
            AppLog.write(String(format: "detect: dumped %d frames to %@/detections/%@/frames · %.0f ms (not counted in find objects)", dumped,
                                afterID, beforeID, dumpMs))
        }
        let sorted = ingestMs.sorted()
        let medianMs = sorted.isEmpty ? 0 : (sorted[(sorted.count - 1) / 2] + sorted[sorted.count / 2]) / 2
        let splatSorted = splatMs.sorted()
        AppLog.write(String(format: "semantic: before-LiDAR splat per frame · median %.1f ms · max %.1f ms · %d frames",
                            splatSorted.isEmpty ? 0 : splatSorted[splatSorted.count / 2], splatSorted.last ?? 0, splatSorted.count))
        let report = tracker.report(beforeID: beforeID, afterID: afterID, method: name,
                                    timingsMs: [Self.findObjects: totalMs, "Ingest (median per frame)": medianMs,
                                                "Before-photo features": photoMs,
                                                "Naming": namingMs])
        let shown = report.events.filter { !$0.hint }
        set(step, .done(String(format: "%d %@ · %d frames · %.1f s", shown.count, shown.count == 1 ? "object" : "objects",
                               ingestMs.count, totalMs / 1000)))
        AppLog.write(String(format: "detect: find objects (%@) · %d frames ingested, %d without cached features · %.0f ms · median ingest %.1f ms · %d fits · ",
                            name, ingestMs.count, uncached, totalMs, medianMs, fits)
                     + (report.events.isEmpty ? "no events" : report.events.map { $0.summary(matchAccept: config.matchAccept) }.joined(separator: " | ")))
        if UserDefaults.standard.bool(forKey: "detectObjects") { Self.logDetails(report, tracker) }
        return report
    }

    /// `-detectObjects YES`, per ingested frame: the changed pixels by depth sign and |d| band, the components by
    /// sign (pixels), the ingest counts, and every instance that disappeared in this frame (merged or dropped).
    private static func logFrame(_ f: FrameContext, _ stats: IngestStats, previous: [TrackedInstance], _ tracker: ObjectTracker) {
        let c = tracker.config
        let W = f.intrinsics.width, H = f.intrinsics.height
        let ref = f.dBefore ?? f.dRen                      // the tracker's reference depth
        let (sign, d, offset) = ChangeSign.compute(dLive: f.dLive, dRen: ref, valid: f.valid, tolerance: c.signTolerance)
        var bySign = [Int8: Int](), noDepth = 0
        var bands = [Int](repeating: 0, count: 5)          // |d| < 5, 5–10, 10–20, 20–50, ≥ 50 mm
        for i in f.changed.indices where f.changed[i] {
            guard d[i].isFinite else { noDepth += 1; continue }
            bySign[sign[i], default: 0] += 1
            let a = abs(d[i])
            bands[a < 0.005 ? 0 : a < 0.01 ? 1 : a < 0.02 ? 2 : a < 0.05 ? 3 : 4] += 1
        }
        let comps = ChangeComponents.extract(changed: f.changed, sign: sign, d: d, width: W, height: H, minPixels: c.minPixels)
        func list(_ s: Int8) -> String {
            let sizes = comps.filter { $0.sign == s }.map(\.pixels.count).sorted(by: >)
            return sizes.isEmpty ? "–" : sizes.prefix(6).map(String.init).joined(separator: ",") + (sizes.count > 6 ? ",…(\(sizes.count))" : "")
        }
        AppLog.write("objects-frame: \(f.frameID) · changed +\(bySign[1] ?? 0) −\(bySign[-1] ?? 0) 0:\(bySign[0] ?? 0) nodepth \(noDepth) · offset \(String(format: "%.1f", offset * 1000)) mm · |d| mm <5 \(bands[0]) 5–10 \(bands[1]) 10–20 \(bands[2]) 20–50 \(bands[3]) ≥50 \(bands[4]) · components + [\(list(1))] − [\(list(-1))] 0 [\(list(0))] · created \(stats.created) merged \(stats.merged) dropped \(stats.dropped) · \(tracker.instances.count) instances")
        // Each signed component back-projected as the tracker does (left points with the reference depth + the frame's offset): how many of its points lie within the margin of
        // the ROI's vertical faces (what drops a new instance in the frame it is created) and how many lie below
        // the support gate (ignored by the tracker).
        if let roi = c.roi {
            let toWorld = f.pose.inverse
            let o: Float = offset.isFinite ? offset : 0
            for comp in comps where comp.sign != 0 {
                let depth = comp.sign > 0 ? f.dLive : ref.map { $0 + o }
                var n = 0, edge = 0, low = 0
                let floorY = c.supportHeight.map { $0 + c.minHeightAboveSupport }
                var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
                var ys: [Float] = []
                for i in comp.pixels where depth[i].isFinite && depth[i] > 0 {
                    let p = toWorld.apply(f.intrinsics.unproject(SIMD2(Float(i % W), Float(i / W)), depth: depth[i]))
                    n += 1; ys.append(p.y)
                    lo = simd_min(lo, p); hi = simd_max(hi, p)
                    let below = p - roi.min, above = roi.max - p, m = c.roiEdgeMargin
                    if below.x < m || below.z < m || above.x < m || above.z < m { edge += 1 }
                    if let floorY, p.y < floorY { low += 1 }
                }
                guard n > 0 else { continue }
                ys.sort()
                AppLog.write(String(format: "objects-comp: %d · sign %+d · %d px · %d points · %d at the ROI sides · %d below the support gate · median y %.3f · box (%.2f, %.2f, %.2f)–(%.2f, %.2f, %.2f)",
                                    f.frameID, Int(comp.sign), comp.pixels.count, n, edge, low, ys[ys.count / 2], lo.x, lo.y, lo.z, hi.x, hi.y, hi.z))
            }
        }
        let now = Set(tracker.instances.map(\.id))
        for i in previous where !now.contains(i.id) {
            let edge = c.roi.map { roi in
                let lo = i.bbox.min - roi.min, hi = roi.max - i.bbox.max
                return String(format: "side gap %.3f", min(lo.x, lo.z, hi.x, hi.z))
            } ?? ""
            AppLog.write(String(format: "objects-gone: id %d · sign %+d · %@ · %d voxels · %d frames · observed %d misses %d · bbox (%.2f, %.2f, %.2f)–(%.2f, %.2f, %.2f) before this frame · %@",
                                i.id, Int(i.sign), i.isConfirmed(c) ? "confirmed" : "unconfirmed", i.voxelCount, i.frames.count,
                                i.observed, i.misses, i.bbox.min.x, i.bbox.min.y, i.bbox.min.z, i.bbox.max.x, i.bbox.max.y, i.bbox.max.z, edge))
        }
    }

    /// `-detectObjects YES`: the report as one `objects:` JSON line, then every instance and every left × arrived
    /// score, for checking the tracker on a recorded pair.
    private static func logDetails(_ report: ObjectReport, _ tracker: ObjectTracker) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        if let data = try? encoder.encode(report), let json = String(data: data, encoding: .utf8) {
            AppLog.write("objects: " + json)
        } else {
            AppLog.write("objects: the report could not be encoded")
        }
        let c = tracker.config
        for i in tracker.instances {
            AppLog.write(String(format: "objects-instance: id %d · sign %+d · %@ · %d voxels · %d frames (%d–%d) · centroid (%.3f, %.3f, %.3f) · observed %d · misses %d · %d points",
                                i.id, Int(i.sign), i.isConfirmed(c) ? "confirmed" : "unconfirmed", i.voxelCount, i.frames.count,
                                i.firstSeen, i.lastSeen, i.centroid.x, i.centroid.y, i.centroid.z, i.observed, i.misses, i.points.count))
        }
        for l in tracker.instances where l.sign < 0 {
            for a in tracker.instances where a.sign > 0 {
                AppLog.write(String(format: "objects-score: left %d → arrived %d · %.3f%@", l.id, a.id, InstanceMatcher.score(l, a),
                                    l.isConfirmed(c) && a.isConfirmed(c) ? "" : " (not both confirmed)"))
            }
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
