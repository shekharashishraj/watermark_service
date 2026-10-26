import Foundation
import OSCDCore

struct PreviewResult {
    let overlayRGBA: [UInt8]         // live frame with the change mask in red, judged area only
    let renderRGBA: [UInt8]?         // the before model rendered from where the camera is
    let width: Int
    let height: Int
    let headline: String             // plain-language status
    let details: String              // per-stage times, for "Show timings"
    let objects: ObjectReport?       // the object layer's newest report (may lag this frame by a frame or two)
    let pose: Pose?                  // this frame's camera, nil when it was not placed
    let intrinsics: Intrinsics       // the camera model `pose` and the masks use
}

/// Live view: runs Method A on the newest camera frame, on one dedicated thread (splatx sessions are
/// single-thread). Online fusion only, no refine: refine prunes the change model to a few hundred
/// Gaussians that smear at viewpoints that were not fused. Each fused frame also goes to a
/// `SemanticStage` (object events, spec §4.1); its reports ride along on the next delivered result and are
/// saved to Documents/preview/objects.json.
final class PreviewWorker {
    var onStatus: ((String) -> Void)?
    var onResult: ((PreviewResult) -> Void)?

    private let lock = NSLock()
    private var pending: CameraFrame?
    private var processing = false
    private var stopRequested = false
    private var ready = false
    private let wake = DispatchSemaphore(value: 0)

    /// True when the pipeline is set up and would take a new frame now.
    var wantsFrame: Bool {
        lock.lock(); defer { lock.unlock() }
        return ready && !processing && pending == nil && !stopRequested
    }

    func start(before: ScanInfo) {
        let thread = Thread { [weak self] in self?.run(before: before) }
        thread.name = "oscd.preview"
        thread.stackSize = 16 << 20
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    func submit(_ frame: CameraFrame) {
        lock.lock(); pending = frame; lock.unlock()
        wake.signal()
    }

    func stop() {
        lock.lock(); stopRequested = true; lock.unlock()
        wake.signal()
    }

    /// Documents/preview-log.txt — status messages and one line per frame (NSLog does not reach
    /// `devicectl --console`; pull this file instead). Written from the preview, fusion and semantic threads.
    private var logHandle: FileHandle?
    private let logLock = NSLock()

    private func log(_ s: String) {
        let line = "\(String(format: "%.3f", Date().timeIntervalSince1970)) \(s)\n"
        logLock.lock(); logHandle?.write(Data(line.utf8)); logLock.unlock()
    }

    /// The semantic stage's newest report and per-frame ingest times (written on its thread).
    private var latestObjects: ObjectReport?
    private var ingestMs: [Double] = []
    /// Each fused frame's before-LiDAR splat time, by the frame ID its `FrameContext` carries (guarded by `lock`).
    private var splatMs: [Int: Double] = [:]

    private var objects: ObjectReport? { lock.lock(); defer { lock.unlock() }; return latestObjects }

    static var objectsURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("preview").appendingPathComponent(ObjectReport.fileName)
    }

    /// On the semantic thread: keep the report for the next result, save it, log one `objects:` line.
    private func received(_ report: ObjectReport) {
        lock.lock(); latestObjects = report; lock.unlock()
        let encoder = JSONEncoder()                       // as ObjectReport.save (SessionStore.encoder is internal)
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        do {
            let url = Self.objectsURL
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try encoder.encode(report).write(to: url, options: .atomic)
        } catch {
            log("objects: could not save \(ObjectReport.fileName): \(error.localizedDescription)")
        }
        log("objects: \(report.events.count) events" + (report.events.isEmpty ? "" : " · ")
            + report.events.map { $0.summary() }.joined(separator: " | "))
    }

    private func status(_ s: String) {
        log("[status] " + s)
        onStatus?(s)
    }

    private func run(before: ScanInfo) {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let logURL = docs.appendingPathComponent("preview-log.txt")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        logHandle = try? FileHandle(forWritingTo: logURL)
        defer { try? logHandle?.close() }
        do {
            let method = try MethodA(beforeScan: before.url, beforeLabel: before.label, withDepth: true) { status($0) }
            var config = TrackerConfig()
            config.roi = method.roi
            config.confirmVoxels = ObjectSettings.minVoxels
            // The table's height from the before-scan's LiDAR: changes lying on it are the sheet, not objects.
            let ts = DispatchTime.now().uptimeNanoseconds
            config.supportHeight = try SupportPlane.height(of: method.beforeStore, roi: method.roi)
            log(config.supportHeight.map {
                String(format: "semantic: support y = %.3f m (%.0f ms)", $0, Double(DispatchTime.now().uptimeNanoseconds - ts) / 1e6)
            } ?? "semantic: support y = unknown (no LiDAR points in the ROI)")
            // The before-scan's own LiDAR: each frame's reference depth for the object layer (spec §4.4, C1).
            let tb = DispatchTime.now().uptimeNanoseconds
            method.beforeLiDAR = try BeforeDepth.points(of: method.beforeStore, roi: method.roi)
            log(String(format: "semantic: before-scan LiDAR · %d points in the ROI · %.0f ms", method.beforeLiDAR?.count ?? 0,
                       Double(DispatchTime.now().uptimeNanoseconds - tb) / 1e6))
            let stage = SemanticStage(config: config, beforeID: before.id, method: "A")
            stage.onReport = { [weak self] in self?.received($0) }
            stage.onIngest = { [weak self] frameID, stats, instances, reportMs in
                guard let self else { return }
                self.lock.lock(); self.ingestMs.append(stats.milliseconds); let splat = self.splatMs[frameID]; self.splatMs = self.splatMs.filter { $0.key > frameID }; self.lock.unlock()
                self.log(String(format: "semantic: frame %d · before-LiDAR splat %@ · ingest %.1f ms · report %.1f ms · %d components · %d instances",
                                frameID, splat.map { String(format: "%.1f ms", $0) } ?? "–", stats.milliseconds, reportMs,
                                stats.components, instances))
            }
            try? FileManager.default.removeItem(at: Self.objectsURL)
            defer {
                stage.close()
                let (ingested, dropped) = stage.counts
                lock.lock(); let ms = ingestMs.sorted(); lock.unlock()
                let median = ms.isEmpty ? 0 : ms[ms.count / 2]
                log(String(format: "semantic: %d frames ingested, %d dropped · median ingest %.1f ms", ingested, dropped, median))
            }
            status("Ready. Point the phone at the table, about 30 cm away.")
            lock.lock(); ready = true; lock.unlock()
            while true { autoreleasepool {                                   // see ModelViewer: images per frame
                wake.wait()
                lock.lock()
                if stopRequested { lock.unlock(); return }
                guard let frame = pending else { lock.unlock(); return }
                pending = nil; processing = true
                lock.unlock()
                // Returns once the change cue is ready; the frame's fusion and mask arrive later from the
                // fusion thread, while the next frame is already being processed.
                process(frame, method: method, stage: stage) { [weak self] result in
                    self?.log(result.headline + " | " + result.details)
                    self?.onResult?(result)
                }
                lock.lock(); processing = false; lock.unlock()
            }; if stopRequested { break } }
            status("Stopped")
        } catch {
            status("Could not start: \(error.localizedDescription)")
        }
    }

    private let namer = ObjectNamer()
    /// Naming attempts per instance id, so one that Vision cannot name does not hold up the others (preview thread only).
    private var namingTries: [Int: Int] = [:]
    private static let maxNamingTries = 3

    /// Live naming (Settings → Name objects): after `method.process` returned, so both of the frame's SAM2 encodes are
    /// done, name at most one confirmed instance without a label from the crop the tracker kept (live crop for
    /// sign ≥ 0, render crop for sign < 0). Vision runs on this thread; the label lands on the tracker thread.
    private func nameOnePending(_ stage: SemanticStage) {
        guard ObjectNamer.enabled, let i = stage.pendingNaming().first(where: { namingTries[$0.id, default: 0] < Self.maxNamingTries })
        else { return }
        namingTries[i.id, default: 0] += 1
        let t = DispatchTime.now().uptimeNanoseconds
        guard let crop = i.bestCrop, let image = ObjectNamer.cgImage(crop), let n = namer.name(image) else {
            // Vision found nothing it is sure of; the tracker may keep a better crop, so it gets a few more tries.
            log(String(format: "naming: object %d has no label yet (%.0f ms)", i.id, Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6))
            return
        }
        try? stage.sync { $0.setLabel(instance: i.id, n.label, confidence: n.confidence) }
        log(String(format: "naming: object %d → %@ %.2f (%.0f ms)", i.id, n.label, n.confidence,
                   Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6))
    }

    private func process(_ f: CameraFrame, method: MethodA, stage: SemanticStage, deliver: @escaping (PreviewResult) -> Void) {
        let W = f.intrinsics.width, H = f.intrinsics.height, n = W * H
        let K = method.anchors.intrinsics
        let plain = PreviewComposite.overlay(rgb8: f.rgb, changed: [Bool](repeating: false, count: n),
                                             valid: [Bool](repeating: true, count: n))
        do {
            let t0 = DispatchTime.now().uptimeNanoseconds
            let out = try method.process(f, fused: { [weak self] out, fused, context in
                // On the fusion thread, after this frame has been fused; `out` is this frame's result.
                if let ms = out.timings.first(where: { $0.0 == MethodA.Stage.beforeDepth })?.1, let self {
                    self.lock.lock(); self.splatMs[context.frameID] = ms; self.lock.unlock()
                }
                stage.submit(context)
                let judged = out.valid.filter { $0 }.count
                let flagged = zip(fused.changed, out.valid).filter { $0 && $1 }.count
                var rgba = [UInt8](repeating: 255, count: n * 4)
                if let ren = out.render {
                    for i in 0..<n { for c in 0..<3 { rgba[i * 4 + c] = UInt8(min(255, ren.data[c * n + i] * 255)) } }
                }
                let totalMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
                let details = out.timings.map { String(format: "%@ %.0f", $0.0, $0.1) }.joined(separator: " · ")
                    + String(format: " · %@ %.0f · %@ %.0f · frame to mask %.0f ms · %d anchor points, %d Gaussians",
                             MethodA.Stage.fusion, fused.fusionMs, MethodA.Stage.mask, fused.maskMs, totalMs,
                             out.anchorPoints, fused.gaussians)
                deliver(PreviewResult(
                    overlayRGBA: PreviewComposite.overlay(rgb8: f.rgb, changed: fused.changed, valid: out.valid),
                    renderRGBA: rgba, width: W, height: H,
                    headline: String(format: "Camera placed · %.0f%% of the table in view changed · %d views combined",
                                     100 * Double(flagged) / Double(max(judged, 1)), fused.fusedFrames),
                    details: details, objects: self?.objects, pose: out.pose, intrinsics: K))
            })
            nameOnePending(stage)
            if out.pose == nil {
                deliver(PreviewResult(overlayRGBA: plain, renderRGBA: nil, width: W, height: H,
                                      headline: "Can't line this view up with \(method.beforeLabel): \(out.rejection ?? "no match"). Point at the same table from about as far as that scan.",
                                      details: out.timings.map { String(format: "%@ %.0f", $0.0, $0.1) }.joined(separator: " · "),
                                      objects: objects, pose: nil, intrinsics: K))
            }
        } catch {
            deliver(PreviewResult(overlayRGBA: plain, renderRGBA: nil, width: W, height: H,
                                  headline: "This frame failed: \(error.localizedDescription)", details: "",
                                  objects: objects, pose: nil, intrinsics: K))
        }
    }
}
