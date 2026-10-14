import CoreML
import Foundation
import OSCDCore
import OSCDFusion
import simd

/// One camera frame handed from ARKit to the preview pipeline.
struct LiveFrame {
    let rgb: [UInt8]                 // 480×360 interleaved RGB (sensor orientation)
    let depth: [Float]
    let confidence: [UInt8]
    let depthWidth: Int
    let depthHeight: Int
    let intrinsics: Intrinsics       // working resolution
}

struct PreviewResult {
    let overlayRGBA: [UInt8]         // live frame with the change mask in red, judged area only
    let renderRGBA: [UInt8]?         // the reference render at the O-SCD pose
    let width: Int
    let height: Int
    let line: String                 // HUD line
}

/// Runs the Method A preview on one dedicated thread (splatx sessions are single-thread):
/// setup (models, anchors from the before-scan, reference renderer), then for each submitted frame
/// XFeat → O-SCD pose → reference render → pixel + SAM2 cue → mask (cue > 1.0, no fusion).
final class PreviewWorker {
    static let disambiguationDepths: (Float, Float) = (0.1, 1.0)     // spec D11
    static let changeThreshold: Float = 1.0

    var onStatus: ((String) -> Void)?
    var onResult: ((PreviewResult) -> Void)?

    private let lock = NSLock()
    private var pending: LiveFrame?
    private var processing = false
    private var stopRequested = false
    private var ready = false
    private let wake = DispatchSemaphore(value: 0)
    private var lastFinish: UInt64 = 0
    private var refineRequested = false
    private var refined = false

    /// True when the pipeline is set up and would take a new frame now.
    var wantsFrame: Bool {
        lock.lock(); defer { lock.unlock() }
        return ready && !processing && pending == nil && !stopRequested
    }

    func start(assets: URL) {
        let thread = Thread { [weak self] in self?.run(assets: assets) }
        thread.name = "oscd.preview"
        thread.stackSize = 16 << 20
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    func submit(_ frame: LiveFrame) {
        lock.lock(); pending = frame; lock.unlock()
        wake.signal()
    }

    /// O-SCD --refine on the frames fused so far (runs on the worker thread between frames).
    func requestRefine() {
        lock.lock(); refineRequested = true; lock.unlock()
        wake.signal()
    }

    func stop() {
        lock.lock(); stopRequested = true; lock.unlock()
        wake.signal()
    }

    private func status(_ s: String) { onStatus?(s) }

    private func run(assets: URL) {
        do {
            status("Loading models…")
            guard let xURL = Bundle.main.url(forResource: "XFeat_480x352", withExtension: "mlmodelc"),
                  let sURL = Bundle.main.url(forResource: "SAM2_1Tiny_2x1024", withExtension: "mlmodelc"),
                  let kernelURL = Bundle.main.url(forResource: "SplatXKernelSrc", withExtension: nil) else {
                throw PreviewError.message("The app is missing a model or the splatx kernel sources; rebuild it.")
            }
            let xfeat = try XFeatModel(compiledModelURL: xURL)
            let sam = try SAM2Encoder(compiledModelURL: sURL)
            let plyURL = assets.appendingPathComponent("reference.ply")
            guard FileManager.default.fileExists(atPath: plyURL.path) else {
                throw PreviewError.message("No reference model at Documents/preview/reference.ply. Copy it from the Mac first.")
            }
            let store = try SessionStore(open: assets.appendingPathComponent("before"))
            guard let roi = store.manifest.roi else { throw PreviewError.message("The before-scan has no table region.") }

            var views: [(pose: Pose, keypoints: DescribedKeypoints)] = []
            for (i, r) in store.manifest.keyframes.enumerated() {
                if i % 5 == 0 { status("Reading the before-scan… \(i)/\(store.manifest.keyframes.count)") }
                let pose = Pose.fromARKitCameraToWorld(simd_float4x4(columnMajor: r.arkitCameraToWorld))
                views.append((pose, try xfeat.detect(try store.rgb(.keyframes, id: r.id))))
            }
            status("Building O-SCD anchors…")
            let anchors = AnchorBuilder.build(keyframes: views, intrinsics: store.manifest.intrinsics,
                                              disambiguationDepths: Self.disambiguationDepths)
            let estimator = PoseEstimator(anchors: anchors)
            status("Opening the reference model (compiling splatx kernels on the phone)…")
            let renderer = try SplatRenderer(plyPath: plyURL.path, kernelSourceDir: kernelURL.path)
            status("Opening the O-SCD change model…")
            let fusion = try ChangeFusion(plyPath: plyURL.path, kernelSourceDir: kernelURL.path,
                                          referenceCentres: views.map { $0.pose.centre })
            status(String(format: "Ready: %d anchors from %d keyframes. Point the phone at the table.", anchors.anchorCount, views.count))
            lock.lock(); ready = true; lock.unlock()

            while true {
                wake.wait()
                lock.lock()
                if stopRequested { lock.unlock(); break }
                if refineRequested {
                    refineRequested = false
                    lock.unlock()
                    status("Refining the change model (O-SCD --refine)…")
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    try fusion.refine()
                    refined = true
                    status(String(format: "Refined in %.0f s: %d steps, %d Gaussians. Keep pointing at the table.",
                                  ms(t0, DispatchTime.now().uptimeNanoseconds) / 1000, fusion.totalSteps, fusion.count))
                    continue
                }
                guard let frame = pending else { lock.unlock(); continue }
                pending = nil; processing = true
                lock.unlock()
                let result = process(frame, xfeat: xfeat, sam: sam, estimator: estimator, renderer: renderer,
                                     fusion: fusion, referenceK: anchors.intrinsics, roi: roi)
                onResult?(result)
                lock.lock(); processing = false; lock.unlock()
            }
            status("Stopped")
        } catch {
            status("Preview failed: \(error.localizedDescription)")
        }
    }

    private func ms(_ a: UInt64, _ b: UInt64) -> Double { Double(b - a) / 1e6 }

    private func process(_ f: LiveFrame, xfeat: XFeatModel, sam: SAM2Encoder, estimator: PoseEstimator,
                         renderer: SplatRenderer, fusion: ChangeFusion, referenceK: Intrinsics, roi: AABB) -> PreviewResult {
        let W = f.intrinsics.width, H = f.intrinsics.height, n = W * H
        let now = { DispatchTime.now().uptimeNanoseconds }
        let t0 = now()
        let rate = lastFinish > 0 ? 1e9 / Double(t0 - lastFinish) : 0
        defer { lastFinish = now() }
        let plain = PreviewComposite.overlay(rgb8: f.rgb, changed: [Bool](repeating: false, count: n),
                                             valid: [Bool](repeating: true, count: n))
        do {
            let live = FloatImage(rgb8: f.rgb, width: W, height: H)
            let kp = try xfeat.detect(live)
            let t1 = now()
            let est = estimator.estimate(kp)
            let t2 = now()
            guard est.accepted, let pose = est.pose else {
                return PreviewResult(overlayRGBA: plain, renderRGBA: nil, width: W, height: H,
                                     line: String(format: "pose rejected (%@): %d kpts, %d corr | xfeat %.0f · pose %.0f ms",
                                                  est.failReason ?? "?", kp.count, est.nCorrespondences, ms(t0, t1), ms(t1, t2)))
            }
            let ren = try renderer.render(pose: pose, intrinsics: referenceK)
            let t3 = now()
            let valid = PreviewValidity.mask(depth: f.depth, confidence: f.confidence, depthWidth: f.depthWidth,
                                             depthHeight: f.depthHeight, intrinsics: f.intrinsics, pose: pose, roi: roi, render: ren)
            // After refine, O-SCD's model is final: show it at new poses without adding frames.
            var t4 = now(), t5 = t4
            if !refined {
                let emb = try sam.encode(render: ren, live: live)
                t4 = now()
                let cues = CuesA.compute(live: live, render: ren, embRender: emb.render, embLive: emb.live, valid: valid)
                try fusion.addFrame(pose: pose, intrinsics: referenceK, cue: cues.combined, valid: valid)
                try fusion.stepFrame()
                t5 = now()
            }
            // O-SCD's mask: the change model rendered at this pose, value > 0.5 (oscd.py:258-263).
            let changed = try fusion.changeMap(pose: pose, intrinsics: referenceK).map { $0 > 0.5 }
            let t6 = now()
            let judged = valid.filter { $0 }.count
            let flagged = zip(changed, valid).filter { $0 && $1 }.count
            var renderRGBA = [UInt8](repeating: 255, count: n * 4)
            for i in 0..<n { for c in 0..<3 { renderRGBA[i * 4 + c] = UInt8(min(255, ren.data[c * n + i] * 255)) } }
            return PreviewResult(
                overlayRGBA: PreviewComposite.overlay(rgb8: f.rgb, changed: changed, valid: valid), renderRGBA: renderRGBA,
                width: W, height: H,
                line: String(format: "pose ✓ %d pts | changed %.0f%% of %.0f%% judged | %@ %d frames, %d Gaussians | xfeat %.0f · pose %.0f · render %.0f · sam2 %.0f · fusion %.0f · mask %.0f ms | %.2f Hz",
                             est.nBAInlierComponents / 2, 100 * Double(flagged) / Double(max(judged, 1)), 100 * Double(judged) / Double(n),
                             refined ? "refined," : "fused", fusion.totalSteps / 16, fusion.count,
                             ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t4, t5), ms(t5, t6), rate))
        } catch {
            return PreviewResult(overlayRGBA: plain, renderRGBA: nil, width: W, height: H, line: "frame failed: \(error.localizedDescription)")
        }
    }
}
