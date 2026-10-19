import Foundation
import OSCDCore
import OSCDFusion

/// O-SCD's change fusion (its own splatx session) on a dedicated thread, so fusing frame N overlaps
/// frame N+1's pose, render and SAM2. Jobs run strictly in submission order, so the change model is
/// exactly what sequential processing gives. At most one frame waits for fusion (back-pressure).
final class FusionStage {
    private let cond = NSCondition()
    private var jobs: [() -> Void] = []
    private var stopping = false
    private let inFlight = DispatchSemaphore(value: 1)
    private var fusion: ChangeFusion!

    init(plyPath: String, kernelSourceDir: String, referenceCentres: [SIMD3<Float>]) throws {
        let thread = Thread { [unowned self] in self.loop() }
        thread.name = "oscd.fusion"
        thread.stackSize = 16 << 20
        thread.qualityOfService = .userInitiated
        thread.start()
        try syncRaw {
            self.fusion = try ChangeFusion(plyPath: plyPath, kernelSourceDir: kernelSourceDir, referenceCentres: referenceCentres)
        }
    }

    deinit { close() }

    /// Queue a fusion job; blocks while a previous submitted job is still running.
    func submit(_ body: @escaping (ChangeFusion) -> Void) {
        inFlight.wait()
        enqueue { [unowned self] in
            body(self.fusion)
            self.inFlight.signal()
        }
    }

    /// Run `body` on the fusion thread after every queued job and return its result.
    func sync<T>(_ body: @escaping (ChangeFusion) throws -> T) throws -> T {
        var result: Result<T, Error>!
        try syncRaw { result = Result { try body(self.fusion) } }
        return try result.get()
    }

    /// Finish queued jobs, release the change model on its own thread, and stop the thread.
    func close() {
        cond.lock()
        guard !stopping else { cond.unlock(); return }
        cond.unlock()
        try? syncRaw { self.fusion = nil }
        cond.lock(); stopping = true; cond.signal(); cond.unlock()
    }

    private func syncRaw(_ body: @escaping () throws -> Void) throws {
        let done = DispatchSemaphore(value: 0)
        var error: Error?
        enqueue {
            do { try body() } catch let e { error = e }
            done.signal()
        }
        done.wait()
        if let error { throw error }
    }

    private func enqueue(_ job: @escaping () -> Void) {
        cond.lock(); jobs.append(job); cond.signal(); cond.unlock()
    }

    private func loop() {
        while true {
            cond.lock()
            while jobs.isEmpty && !stopping { cond.wait() }
            if jobs.isEmpty && stopping { cond.unlock(); return }
            let job = jobs.removeFirst()
            cond.unlock()
            job()
        }
    }
}
