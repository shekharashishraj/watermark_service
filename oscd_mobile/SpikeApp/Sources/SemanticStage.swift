import Foundation
import OSCDCore

/// The object layer (spec §4.1, §4.12): one `ObjectTracker` on its own thread "oscd.semantic", fed with
/// fused frames. At most one frame waits; a newer frame replaces it (the older one is dropped), so the
/// fusion thread never blocks here. After each ingest the matched pairs without a measured move are fitted
/// with `PlanarICP` on a `.utility` queue; the fit lands back on the tracker thread with `setMotion`, and a
/// new report follows. Only the tracker thread touches the tracker. Call `close()` when done (after the last fused
/// frame was submitted): it ingests a frame still waiting, measures the moves still missing and sends a final report.
/// The thread keeps the stage alive until then.
final class SemanticStage {
    /// Called on the tracker thread after each ingest and after each measured move lands.
    var onReport: ((ObjectReport) -> Void)?
    /// Called on the tracker thread after each ingest: the frame's id, its stats, the instance count and
    /// how long matching and the report took (ms).
    var onIngest: ((_ frameID: Int, _ stats: IngestStats, _ instances: Int, _ reportMs: Double) -> Void)?
    /// Called on the tracker thread at `close()`, before the final report: how many moves were measured then.
    var onFinal: ((_ fits: Int) -> Void)?

    private let tracker: ObjectTracker
    private let beforeID: String
    private let method: String
    private let fitQueue = DispatchQueue(label: "oscd.semantic.fit", qos: .utility)

    private let cond = NSCondition()
    private var pending: FrameContext?            // guarded by cond
    private var jobs: [() -> Void] = []           // guarded by cond
    private var stopping = false                  // guarded by cond
    private var ingested = 0, dropped = 0         // guarded by cond
    private let finished = DispatchSemaphore(value: 0)

    /// Pairs whose fit is queued or running; touched only on the tracker thread.
    private var fitting = Set<PairID>()
    private var lastIngestMs = 0.0               // tracker thread only

    private struct PairID: Hashable { let left: Int; let arrived: Int }
    /// The voxel counts of a pair when its fit's points were copied.
    private struct FitInput { let leftVoxels: Int; let arrivedVoxels: Int }

    init(config: TrackerConfig, beforeID: String, method: String) {
        tracker = ObjectTracker(config: config)
        self.beforeID = beforeID
        self.method = method
        let thread = Thread { self.loop() }
        thread.name = "oscd.semantic"
        thread.stackSize = 16 << 20
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    /// Hands a fused frame to the tracker thread. Never blocks: a frame still waiting is dropped.
    func submit(_ context: FrameContext) {
        cond.lock()
        defer { cond.unlock() }
        guard !stopping else { return }
        if pending != nil { dropped += 1 }
        pending = context
        cond.signal()
    }

    /// Runs body on the tracker thread (naming, before-photo features). Blocks until done.
    func sync<T>(_ body: @escaping (ObjectTracker) throws -> T) throws -> T {
        let done = DispatchSemaphore(value: 0)
        var result: Result<T, Error> = .failure(SemanticStageError.closed)
        let queued = enqueue { [tracker] in
            result = Result { try body(tracker) }
            done.signal()
        }
        guard queued else { throw SemanticStageError.closed }
        done.wait()
        return try result.get()
    }

    /// Confirmed instances without a label, oldest first. Live: at most one is named per frame, on the preview thread,
    /// and only when the newest report has an unnamed object (non-hint event without a label). Runs on the tracker
    /// thread through `sync`, so it waits for an ingest in progress.
    func pendingNaming() -> [TrackedInstance] {
        (try? sync { t in t.instances.filter { $0.label == nil && $0.isConfirmed(t.config) }.sorted { $0.id < $1.id } }) ?? []
    }

    /// Frames ingested and frames dropped because a newer one arrived first.
    var counts: (ingested: Int, dropped: Int) {
        cond.lock(); defer { cond.unlock() }
        return (ingested, dropped)
    }

    /// Stops taking frames and jobs, runs the jobs already queued, ingests a frame still waiting (Stop must not lose
    /// the last fused frame), then on the tracker thread measures every matched pair still without a move (the same
    /// fit, synchronously; asynchronous fits still running are discarded when they finish) and sends one final
    /// report when any frame was ingested. Blocks until then; no callback runs after this returns.
    func close() {
        cond.lock()
        guard !stopping else { cond.unlock(); return }
        stopping = true
        cond.signal()
        cond.unlock()
        finished.wait()
    }

    // MARK: - Tracker thread

    @discardableResult
    private func enqueue(_ job: @escaping () -> Void) -> Bool {
        cond.lock(); defer { cond.unlock() }
        guard !stopping else { return false }
        jobs.append(job)
        cond.signal()
        return true
    }

    /// Queued jobs (landed fits, `sync` bodies) run before the waiting frame.
    private func loop() {
        while true {
            cond.lock()
            while jobs.isEmpty && pending == nil && !stopping { cond.wait() }
            if !jobs.isEmpty {
                let job = jobs.removeFirst()
                cond.unlock()
                autoreleasepool { job() }
            } else if let frame = pending {
                pending = nil
                ingested += 1
                cond.unlock()
                autoreleasepool { ingest(frame) }
            } else {                                // stopping, nothing left
                let any = ingested > 0
                cond.unlock()
                if any { autoreleasepool { finalReport() } }
                break
            }
        }
        onReport = nil; onIngest = nil; onFinal = nil
        finished.signal()
    }

    private func ingest(_ frame: FrameContext) {
        let stats = tracker.ingest(frame)
        lastIngestMs = stats.milliseconds
        let t0 = DispatchTime.now().uptimeNanoseconds
        startFits()
        let report = makeReport()
        let reportMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        onIngest?(frame.frameID, stats, tracker.instances.count, reportMs)
        onReport?(report)
    }

    private func makeReport() -> ObjectReport {
        tracker.report(beforeID: beforeID, afterID: nil, method: method, timingsMs: ["Ingest (last frame)": lastIngestMs])
    }

    /// Queues one PlanarICP fit per matched pair that has no measured move and no fit in flight. The same pairs, inputs
    /// and fit as `ObjectTracker.fitMotions()`, but asynchronous so the tracker thread keeps ingesting.
    private func startFits() {
        for m in tracker.pairsNeedingMotion() {
            let key = PairID(left: m.leftID, arrived: m.arrivedID)
            guard !fitting.contains(key), let input = tracker.motionInputs(m) else { continue }
            fitting.insert(key)
            let (source, target) = (input.source, input.target)
            let counts = FitInput(leftVoxels: input.leftVoxels, arrivedVoxels: input.arrivedVoxels)
            fitQueue.async { [weak self] in
                let motion = ObjectTracker.measureMotion(source: source, target: target)
                self?.enqueue { [weak self] in self?.landFit(key, motion, counts) }
            }
        }
    }

    /// At close: the moves still missing, measured here, then the last report.
    private func finalReport() {
        let fits = tracker.fitMotions()
        fitting.removeAll()
        onFinal?(fits)
        onReport?(makeReport())
    }

    /// The counts are those at copy time, so an instance that grew > 20 % while the fit ran is fitted again.
    private func landFit(_ key: PairID, _ motion: Motion?, _ counts: FitInput) {
        fitting.remove(key)
        tracker.setMotion(leftID: key.left, arrivedID: key.arrived, motion, leftVoxels: counts.leftVoxels, arrivedVoxels: counts.arrivedVoxels)
        onReport?(makeReport())
    }
}

/// Settings that feed the object layer's `TrackerConfig`.
enum ObjectSettings {
    static let minVoxelsKey = "minObjectVoxels"
    static let minVoxelsDefault = 15
    static let minVoxelsRange = 5...60

    /// Smallest object (voxels of 1 cm) that is reported; `TrackerConfig.confirmVoxels`. Launch argument `-minObjectVoxels N` works too.
    static var minVoxels: Int {
        let d = UserDefaults.standard
        guard d.object(forKey: minVoxelsKey) != nil else { return minVoxelsDefault }
        return min(max(d.integer(forKey: minVoxelsKey), minVoxelsRange.lowerBound), minVoxelsRange.upperBound)
    }
}

/// The object layer's per-run inputs from the before-scan (R35): the table height (`SupportPlane`) and the before-scan's
/// LiDAR in the ROI (`BeforeDepth`). Without either the layer is off for the run: it never runs without the support
/// gate, and Methods A/B carry on as usual.
enum ObjectLayerInputs {
    struct Ready {
        let support: SupportPlane.Estimate
        let beforePoints: [SIMD3<Float>]
    }

    /// `.success` with both inputs, or `.failure` with the reason the layer is off, for a step label or a status line
    /// ("no reliable table height in the before-scan …"). `log` gets one line per input.
    static func prepare(store: SessionStore, roi: AABB, log: (String) -> Void) -> Result<Ready, Off> {
        let ts = DispatchTime.now().uptimeNanoseconds
        let band: SupportPlane.Estimate?
        do {
            band = try SupportPlane.densestBand(of: store, roi: roi)
        } catch {
            log("semantic: support y = unknown (the before-scan's LiDAR could not be read: \(error.localizedDescription)) · object layer off")
            return .failure(Off(reason: "the before-scan's LiDAR could not be read"))
        }
        let supportMs = Double(DispatchTime.now().uptimeNanoseconds - ts) / 1e6
        guard let band else {
            log(String(format: "semantic: support y = unknown (no LiDAR points in the ROI, %.0f ms) · object layer off", supportMs))
            return .failure(Off(reason: "no LiDAR points on the before-scan's table"))
        }
        log(String(format: "semantic: support y = %.3f m (share %.1f %% of %d points, %.0f ms)%@", band.height, band.share * 100,
                   band.points, supportMs, band.isReliable ? "" : String(format: " · under %.0f %%: object layer off", SupportPlane.minShare * 100)))
        guard band.isReliable else {
            return .failure(Off(reason: String(format: "no reliable table height in the before-scan (densest 1 cm band: %.0f %% of its points)",
                                               band.share * 100)))
        }
        let tb = DispatchTime.now().uptimeNanoseconds
        let points: [SIMD3<Float>]
        do {
            points = try BeforeDepth.points(of: store, roi: roi)
        } catch {
            log("semantic: before-scan LiDAR could not be read: \(error.localizedDescription) · object layer off")
            return .failure(Off(reason: "the before-scan's LiDAR could not be read"))
        }
        log(String(format: "semantic: before-scan LiDAR · %d points in the ROI · %.0f ms", points.count,
                   Double(DispatchTime.now().uptimeNanoseconds - tb) / 1e6))
        return .success(Ready(support: band, beforePoints: points))
    }

    struct Off: Error {
        let reason: String
        /// "Off: no reliable table height in the before-scan (…)".
        var label: String { "Off: " + reason }
    }
}

enum SemanticStageError: LocalizedError {
    case closed
    var errorDescription: String? { "The object tracker has stopped." }
}
