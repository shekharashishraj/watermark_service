import Combine
import Foundation
import OSCDCore
import SplatXTrainC

/// Builds a scan's 3D model on the phone (spec §7): ColmapExport, then splatx training with the settings of
/// the Mac model the spike validated (SH 3, 0.8·L1 + 0.2·D-SSIM; steps and Gaussian cap from `Quality`),
/// kernels compiled on the phone. The result replaces the scan's reference.ply; an existing model is kept
/// as reference.previous.ply. splatx allows one training run at a time, on one thread.
final class ReferenceBuilder: ObservableObject {
    enum Stage: Equatable {
        case idle
        case exporting
        case training(iter: Int, total: Int, splats: Int, secondsLeft: Double, thermal: Int)
        case saving
        case done(seconds: Double, trainSeconds: Double, splats: Int)
        case failed(String)
    }

    /// Steps and starting Gaussians. On the 2026-10-25 room scan (105 frames, Mac M4 ≈ phone speed): Quick 24.0 dB
    /// held-out in 23 s, Standard 25.0 dB in 2 min, High 25.4 dB in 4 min; more steps or Gaussians beyond that gave
    /// nothing. Standard is the default; Quick is for a change-detection reference in a hurry.
    enum Quality: String, CaseIterable, Identifiable {
        case quick, standard, high, brush
        var id: String { rawValue }
        var iterations: Int32 {
            switch self {
            case .quick: 2000; case .standard: 8000; case .high: 8000
            case .brush: Int32(UserDefaults.standard.integer(forKey: "brushSteps")).nonZero ?? 7000   // `-brushSteps N` for checks
            }
        }
        var maxPoints: Int64 { switch self { case .quick: 200_000; case .standard: 400_000; case .high: 600_000; case .brush: 0 } }
        /// Brush (adaptive density control, mip-splatting) instead of splatx: the trainer dinovi uses.
        var usesBrush: Bool { self == .brush }
        /// High trains at the full capture resolution (1920×1440) from the keyframes' JPEGs: what the model can
        /// learn is bounded by the training image. Scans recorded before the JPEGs existed fall back to 480×360.
        var fullResolution: Bool { self == .high || self == .brush }
        var factor: Int32 { 1 }
        var label: String {
            switch self {
            case .quick: "Quick (2,000 steps, about 30 s)"
            case .standard: "Standard (8,000 steps, about 2 min)"
            case .high: "High (full resolution 1920×1440, 8,000 steps, 600k Gaussians)"
            case .brush: "Brush (7,000 steps, grows to ~1M Gaussians, 1280 px) — experimental"
            }
        }
        static let defaultsKey = "trainQuality"
        static var current: Quality {
            Quality(rawValue: UserDefaults.standard.string(forKey: defaultsKey) ?? "") ?? .standard
        }
    }
    static var iterations: Int32 { Quality.current.iterations }

    @Published private(set) var stage = Stage.idle
    @Published private(set) var scanLabel = ""
    @Published private(set) var exportSeconds = 0.0
    var running: Bool {
        switch stage { case .exporting, .training, .saving: return true; default: return false }
    }

    private var thermalObserver: NSObjectProtocol?
    private var started = Date()
    private var lastPublish = 0.0

    func start(scan: ScanInfo) {
        guard !running else { return }
        scanLabel = scan.label
        stage = .exporting
        started = Date()
        let thread = Thread { [weak self] in self?.run(scan: scan) }
        thread.name = "oscd.reference-build"
        thread.stackSize = 16 << 20
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    func cancel() { sx_train_request_stop() }

    private func publish(_ s: Stage) { DispatchQueue.main.async { self.stage = s } }

    private func log(_ s: String) { AppLog.write("build: " + s) }

    private func run(scan: ScanInfo) {
        let fm = FileManager.default
        let work = fm.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("reference-build").appendingPathComponent(scan.id)
        do {
            try? fm.removeItem(at: work)
            let t0 = Date()
            let session = try SessionStore(open: scan.url)
            let fullRes = Quality.current.fullResolution && session.hasFullImages
            // The export loads every keyframe's arrays on this thread: drain their autoreleased objects before training.
            let export = try autoreleasepool {
                try ColmapExport.write(session: session, to: work.appendingPathComponent("colmap"), fullResolution: fullRes)
            }
            log("\(scan.id): after export mem \(AppLog.residentMB()) MB")
            let exportSec = Date().timeIntervalSince(t0)
            DispatchQueue.main.async { self.exportSeconds = exportSec }
            log(String(format: "%@: exported %d images, %d points in %.1f s", scan.id, export.images, export.points, exportSec))
            guard export.images >= 20 else { throw PreviewError.message("\(scan.label) has only \(export.images) frames; at least 20 are needed.") }
            guard let kernelURL = Bundle.main.url(forResource: "SplatXKernelSrc", withExtension: nil) else {
                throw PreviewError.message("The app is missing the splatx kernel sources; rebuild it.")
            }

            let quality = Quality.current
            log("\(scan.id): training \(quality.rawValue): \(quality.iterations) steps, up to \(quality.maxPoints) Gaussians, "
                + (fullRes ? "\(session.manifest.captureWidth / Int(quality.factor))×\(session.manifest.captureHeight / Int(quality.factor)) from the full-size JPEGs" : "480×360"))
            publish(.training(iter: 0, total: Int(quality.iterations), splats: export.points, secondsLeft: .nan, thermal: 0))
            startThermalFeed()
            defer { stopThermalFeed() }
            let out = work.appendingPathComponent("reference.ply")
            var stats = SxTrainStats()
            if quality.usesBrush {
                let t0 = Date()
                // `-brushMaxRes N` for checks (default 1280). Memory and progress go to the log once a second.
                var bopts = BrushTrainer.Options(steps: UInt32(quality.iterations))
                if let r = UInt32(exactly: UserDefaults.standard.integer(forKey: "brushMaxRes")), r > 0 { bopts.maxResolution = r }
                if let m = UInt32(exactly: UserDefaults.standard.integer(forKey: "brushMaxSplats")), m > 0 { bopts.maxSplats = m }   // `-brushMaxSplats N`
                log("\(scan.id): Brush starting, \(bopts.maxResolution) px, up to \(bopts.maxSplats) splats, mem \(AppLog.residentMB()) MB")
                var lastLog = Date()
                let ply = try BrushTrainer.train(dataset: work.appendingPathComponent("colmap"), outputDir: work.appendingPathComponent("brush"),
                                                 options: bopts) { [weak self] iter in
                    guard let self else { return }
                    if Date().timeIntervalSince(lastLog) > 1 || iter <= 2 {
                        lastLog = Date()
                        AppLog.write("build: Brush step \(iter)  mem \(AppLog.residentMB()) MB", sync: true)
                    }
                    let now = Date().timeIntervalSince1970
                    guard now - lastPublish > 0.2 || iter >= Int(quality.iterations) else { return }
                    lastPublish = now
                    let elapsed = Date().timeIntervalSince(t0)
                    publish(.training(iter: iter, total: Int(quality.iterations), splats: 0,
                                      secondsLeft: iter > 0 ? elapsed / Double(iter) * Double(Int(quality.iterations) - iter) : .nan, thermal: 0))
                }
                try fm.moveItem(at: ply, to: out)
                stats.train_seconds = Date().timeIntervalSince(t0)
                stats.completed_iters = quality.iterations
                stats.exported_splats = 0
                log(String(format: "%@: Brush trained %d steps in %.1f s", scan.id, quality.iterations, stats.train_seconds))
            } else {
            var err = [CChar](repeating: 0, count: 4096)
            let rc: Int32 = work.appendingPathComponent("colmap").path.withCString { data in
                out.path.withCString { outPath in
                    kernelURL.path.withCString { kernels in
                        withUnsafeMutablePointer(to: &stats) { statsPtr in
                            var a = SxTrainArgs()
                            a.data_dir = data
                            a.factor = fullRes ? quality.factor : 1
                            a.iters = quality.iterations
                            a.max_points = quality.maxPoints
                            a.sh_degree = 3
                            a.out_path = outPath
                            a.eval_image = 0
                            a.kernel_src_dir = kernels
                            a.log_every = 100
                            a.mcmc = 0
                            a.cap_max = 1_000_000
                            a.test_split = 0
                            a.eval_every = 0
                            a.ssim_weight = 0.2
                            a.antialiased = 0
                            a.resident = 1
                            a.thermal_mode = 1                 // balanced, as the Mac CLI default
                            a.progress = { p, user in
                                guard let p, let user else { return }
                                Unmanaged<ReferenceBuilder>.fromOpaque(user).takeUnretainedValue().onProgress(p.pointee)
                            }
                            a.progress_user = Unmanaged.passUnretained(self).toOpaque()
                            a.stats = statsPtr
                            return sx_train_run_args(&a, &err, Int32(err.count))
                        }
                    }
                }
            }
            switch rc {
            case 0: break
            case 2: throw PreviewError.message("Stopped because the phone got too hot. Let it cool and try again.")
            case 3: throw PreviewError.message("Cancelled")
            default: throw PreviewError.message("Training failed: \(String(cString: err))")
            }
            log(String(format: "%@: trained %d iterations in %.1f s, %lld Gaussians", scan.id, stats.completed_iters,
                       stats.train_seconds, stats.exported_splats))
            }

            publish(.saving)
            let dest = scan.url.appendingPathComponent(ScanCatalog.referenceFile)
            if fm.fileExists(atPath: dest.path) {
                let previous = scan.url.appendingPathComponent("reference.previous.ply")
                try? fm.removeItem(at: previous)
                try fm.moveItem(at: dest, to: previous)
            }
            try fm.moveItem(at: out, to: dest)
            try? fm.removeItem(at: work)
            let total = Date().timeIntervalSince(started)
            log(String(format: "%@: done in %.1f s (export %.1f s, training %.1f s)", scan.id, total, exportSec, stats.train_seconds))
            publish(.done(seconds: total, trainSeconds: stats.train_seconds, splats: Int(stats.exported_splats)))
        } catch {
            log("\(scan.id): failed: \(error.localizedDescription)")
            publish(.failed(error.localizedDescription))
        }
    }

    /// Called on the training thread at every step; publishes at most 5 times a second.
    private func onProgress(_ p: SxTrainProgress) {
        let now = Date().timeIntervalSince1970
        guard now - lastPublish > 0.2 || p.iter == p.run_until else { return }
        lastPublish = now
        let left = Double(p.run_until - p.iter) * p.step_ms / 1000
        publish(.training(iter: Int(p.iter), total: Int(p.run_until), splats: Int(p.splat_count), secondsLeft: left,
                          thermal: Int(p.thermal_state)))
    }

    /// splatx does not watch the platform; the app feeds it the thermal state (0 nominal … 3 critical).
    private func startThermalFeed() {
        sx_thermal_set_state(Int32(ProcessInfo.processInfo.thermalState.rawValue))
        thermalObserver = NotificationCenter.default.addObserver(forName: ProcessInfo.thermalStateDidChangeNotification,
                                                                 object: nil, queue: nil) { _ in
            sx_thermal_set_state(Int32(ProcessInfo.processInfo.thermalState.rawValue))
        }
    }

    private func stopThermalFeed() {
        if let thermalObserver { NotificationCenter.default.removeObserver(thermalObserver) }
        thermalObserver = nil
    }
}

private extension Int32 { var nonZero: Int32? { self == 0 ? nil : self } }
