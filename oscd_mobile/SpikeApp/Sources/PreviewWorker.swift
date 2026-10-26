import Foundation
import OSCDCore

struct PreviewResult {
    let overlayRGBA: [UInt8]         // live frame with the change mask in red, judged area only
    let renderRGBA: [UInt8]?         // the before model rendered from where the camera is
    let width: Int
    let height: Int
    let headline: String             // plain-language status
    let details: String              // per-stage times, for "Show timings"
}

/// Live view: runs Method A on the newest camera frame, on one dedicated thread (splatx sessions are
/// single-thread). Online fusion only, no refine: refine prunes the change model to a few hundred
/// Gaussians that smear at viewpoints that were not fused.
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
    /// `devicectl --console`; pull this file instead).
    private var logHandle: FileHandle?

    private func log(_ s: String) {
        let line = "\(String(format: "%.3f", Date().timeIntervalSince1970)) \(s)\n"
        logHandle?.write(Data(line.utf8))
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
            let method = try MethodA(beforeScan: before.url, beforeLabel: before.label) { status($0) }
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
                process(frame, method: method) { [weak self] result in
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

    private func process(_ f: CameraFrame, method: MethodA, deliver: @escaping (PreviewResult) -> Void) {
        let W = f.intrinsics.width, H = f.intrinsics.height, n = W * H
        let plain = PreviewComposite.overlay(rgb8: f.rgb, changed: [Bool](repeating: false, count: n),
                                             valid: [Bool](repeating: true, count: n))
        do {
            let t0 = DispatchTime.now().uptimeNanoseconds
            let out = try method.process(f, fused: { out, fused in
                // On the fusion thread, after this frame has been fused; `out` is this frame's result.
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
                    details: details))
            })
            if out.pose == nil {
                deliver(PreviewResult(overlayRGBA: plain, renderRGBA: nil, width: W, height: H,
                                      headline: "Can't line this view up with \(method.beforeLabel): \(out.rejection ?? "no match"). Point at the same table from about as far as that scan.",
                                      details: out.timings.map { String(format: "%@ %.0f", $0.0, $0.1) }.joined(separator: " · ")))
            }
        } catch {
            deliver(PreviewResult(overlayRGBA: plain, renderRGBA: nil, width: W, height: H,
                                  headline: "This frame failed: \(error.localizedDescription)", details: ""))
        }
    }
}
