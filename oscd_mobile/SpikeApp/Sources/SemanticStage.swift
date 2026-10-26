import Foundation
import OSCDCore

/// The object layer (spec §4.1, §4.12): one `ObjectTracker` on its own thread "oscd.semantic", fed with
/// fused frames. At most one frame waits; a newer frame replaces it (the older one is dropped), so the
/// fusion thread never blocks here. After each ingest the matched pairs without a measured move are fitted
/// with `PlanarICP` on a `.utility` queue; the fit lands back on the tracker thread with `setMotion`, and a
/// new report follows. Only the tracker thread touches the tracker. Call `close()` when done: the thread
/// keeps the stage alive until then.
final class SemanticStage {
    /// Called on the tracker thread after each ingest and after each measured move lands.
    var onReport: ((ObjectReport) -> Void)?
    /// Called on the tracker thread after each ingest: the frame's id, its stats, the instance count and
    /// how long matching and the report took (ms).
    var onIngest: ((_ frameID: Int, _ stats: IngestStats, _ instances: Int, _ reportMs: Double) -> Void)?

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

    /// Confirmed instances without a label, oldest first. Live: at most one is named per frame, on the preview thread.
    /// Runs on the tracker thread through `sync`, so it waits for an ingest in progress.
    func pendingNaming() -> [TrackedInstance] {
        (try? sync { t in t.instances.filter { $0.label == nil && $0.isConfirmed(t.config) }.sorted { $0.id < $1.id } }) ?? []
    }

    /// Frames ingested and frames dropped because a newer one arrived first.
    var counts: (ingested: Int, dropped: Int) {
        cond.lock(); defer { cond.unlock() }
        return (ingested, dropped)
    }

    /// Drops a waiting frame, runs the jobs already queued, and stops the thread. Moves still being measured
    /// are discarded when they finish. No callback runs after this returns.
    func close() {
        cond.lock()
        guard !stopping else { cond.unlock(); return }
        stopping = true
        if pending != nil { pending = nil; dropped += 1 }
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
                cond.unlock()
                break
            }
        }
        onReport = nil; onIngest = nil
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
            guard !fitting.contains(key), let (source, target) = tracker.motionInputs(m) else { continue }
            fitting.insert(key)
            fitQueue.async { [weak self] in
                let motion = ObjectTracker.measureMotion(source: source, target: target)
                self?.enqueue { [weak self] in self?.landFit(key, motion) }
            }
        }
    }

    private func landFit(_ key: PairID, _ motion: Motion?) {
        fitting.remove(key)
        tracker.setMotion(leftID: key.left, arrivedID: key.arrived, motion)
        onReport?(makeReport())
    }
}

enum SemanticStageError: LocalizedError {
    case closed
    var errorDescription: String? { "The object tracker has stopped." }
}
