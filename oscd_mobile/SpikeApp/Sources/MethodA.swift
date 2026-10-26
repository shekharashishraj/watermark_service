import CoreML
import Foundation
import OSCDCore
import OSCDFusion
import simd

/// One camera frame (live or recorded) at the working resolution.
struct CameraFrame {
    let rgb: [UInt8]                 // 480×360 interleaved RGB (sensor orientation)
    let depth: [Float]
    let confidence: [UInt8]
    let depthWidth: Int
    let depthHeight: Int
    let intrinsics: Intrinsics
}

/// Method A, exact O-SCD, against one before-scan and its 3D model. Shared by Detect (recorded
/// after-scan) and the live view. splatx sessions are single-thread: create and use this object on
/// one thread only.
final class MethodA {
    static let disambiguationDepths: (Float, Float) = (0.1, 1.0)     // spec D11

    /// Stage names as the results screen shows them.
    enum Stage {
        static let loadModels = "Load XFeat and SAM2"
        static let beforeKeypoints = "XFeat on the before-scan"
        static let anchors = "Build 3D anchors"
        static let openModel = "Open the before 3D model"
        static let openFusion = "Open the change model"
        static let setup = [loadModels, beforeKeypoints, anchors, openModel, openFusion]

        static let xfeat = "XFeat keypoints"
        static let retrieval = "Find the closest before frames"
        static let matching = "Match keypoints to 3D anchors"
        static let pnp = "PnP-RANSAC"
        static let bundleAdjust = "Bundle adjustment"
        static let render = "Render the before model"
        static let judged = "Judged area (LiDAR, table)"
        static let sam2 = "SAM2 features (render)"
        static let sam2Live = "SAM2 features (live, alongside pose)"
        static let cue = "Change cue (pixel + SAM2)"
        static let fusion = "Fusion (16 steps)"
        static let mask = "Change mask"
        static let fusionWait = "Wait for the previous frame's fusion"
        static let depthRender = "B: render the before model's depth"
        static let expectedDepth = "Render the before model's depth"
        static let cuesB = "B: depth + colour cues"
        static let perFrame = [xfeat, retrieval, matching, pnp, bundleAdjust, render, judged, sam2, sam2Live, cue, fusionWait, fusion, mask]

        static let refine = "Refine (O-SCD final pass)"
        static let finalMasks = "Final masks"
    }

    struct Output {
        var pose: Pose?
        var rejection: String?           // plain-language reason when the camera could not be placed
        var anchorPoints = 0
        var matches = 0                  // 2D–3D correspondences with the before-scan's anchors
        var valid: [Bool] = []
        var changed: [Bool] = []
        var render: FloatImage?
        var timings: [(String, Double)] = []
    }

    /// What fusing one frame produced (delivered on the fusion thread).
    struct Fused {
        var changed: [Bool]               // change model at the frame's pose after fusing it, > 0.5
        var fusionMs: Double
        var maskMs: Double
        var fusedFrames: Int
        var gaussians: Int
    }

    let anchors: AnchorSet
    let roi: AABB
    let beforeLabel: String
    private(set) var setupTimings = StageTimings()
    private var fusedFrames = 0                 // touched only on the fusion threads
    private var fusedFramesB = 0
    private let xfeat: XFeatModel
    private let sam: SAM2Encoder
    private let estimator: PoseEstimator
    private let renderer: SplatRenderer
    private let fusion: FusionStage
    /// Method B (spec §8): same frames and poses as A, cue C_B = clamp(C_A + C_depth + C_color, 0, 2),
    /// fused into its own change model with the same seed. Nil when only A runs.
    private let fusionB: FusionStage?
    var hasB: Bool { fusionB != nil }
    /// The before model's expected depth (B's depth cue, and D_ren for the object layer). Opened when
    /// `withB || withDepth`; used on this object's thread only.
    private let referenceDepth: ReferenceDepth?
    /// The opened before-scan.
    let beforeStore: SessionStore

    /// `withDepth` opens the before model's depth renderer without running B, so the fused callbacks'
    /// `FrameContext` carries D_ren (otherwise D_ren is all NaN and the object layer finds nothing).
    init(beforeScan: URL, beforeLabel: String, withB: Bool = false, withDepth: Bool = false,
         status: (String) -> Void) throws {
        self.beforeLabel = beforeLabel
        var clock = DispatchTime.now().uptimeNanoseconds
        var setup = StageTimings()
        func lap(_ stage: String) {
            let now = DispatchTime.now().uptimeNanoseconds
            setup.add(stage, Double(now - clock) / 1e6)
            clock = now
        }

        status("Loading the models…")
        guard let xURL = Bundle.main.url(forResource: "XFeat_480x352", withExtension: "mlmodelc"),
              let sURL = Bundle.main.url(forResource: "SAM2_1Tiny_1x1024", withExtension: "mlmodelc"),
              let kernelURL = Bundle.main.url(forResource: "SplatXKernelSrc", withExtension: nil) else {
            throw PreviewError.message("The app is missing a model. Rebuild it with the models in oscd_mobile/Models.")
        }
        let plyURL = beforeScan.appendingPathComponent(ScanCatalog.referenceFile)
        guard FileManager.default.fileExists(atPath: plyURL.path) else {
            throw PreviewError.message("\(beforeLabel) has no 3D model yet. Build it on the Mac with tools/build_reference.sh.")
        }
        xfeat = try XFeatModel(compiledModelURL: xURL)
        sam = try SAM2Encoder(compiledModelURL: sURL)
        lap(Stage.loadModels)

        let store = try SessionStore(open: beforeScan)
        beforeStore = store
        guard let roi = try SceneRegion.region(of: store) else { throw PreviewError.message("\(beforeLabel) has no frames with depth.") }
        self.roi = roi
        var views: [(pose: Pose, keypoints: DescribedKeypoints)] = []
        for (i, r) in store.manifest.keyframes.enumerated() {
            if i % 5 == 0 { status("Finding keypoints in \(beforeLabel)… \(i)/\(store.manifest.keyframes.count)") }
            let pose = Pose.fromARKitCameraToWorld(simd_float4x4(columnMajor: r.arkitCameraToWorld))
            views.append((pose, try xfeat.detect(try store.rgb(.keyframes, id: r.id))))
        }
        lap(Stage.beforeKeypoints)
        status("Building 3D anchors from \(beforeLabel)…")
        anchors = AnchorBuilder.build(keyframes: views, intrinsics: store.manifest.intrinsics,
                                      disambiguationDepths: Self.disambiguationDepths)
        estimator = PoseEstimator(anchors: anchors)
        lap(Stage.anchors)
        status("Opening \(beforeLabel)'s 3D model…")
        renderer = try SplatRenderer(plyPath: plyURL.path, kernelSourceDir: kernelURL.path)
        lap(Stage.openModel)
        status("Preparing the change model…")
        fusion = try FusionStage(plyPath: plyURL.path, kernelSourceDir: kernelURL.path,
                                 referenceCentres: views.map { $0.pose.centre })
        if withB {
            fusionB = try FusionStage(plyPath: plyURL.path, kernelSourceDir: kernelURL.path,
                                      referenceCentres: views.map { $0.pose.centre })
        } else {
            fusionB = nil
        }
        referenceDepth = withB || withDepth ? try ReferenceDepth(plyPath: plyURL.path, kernelSourceDir: kernelURL.path) : nil
        lap(Stage.openFusion)
        setupTimings = setup
    }

    /// XFeat → O-SCD pose → render → judged area → SAM2 + pixel cue, then hands the cue to the fusion
    /// thread: 16 SSF steps and the mask (change model rendered at this pose, value > 0.5,
    /// oscd.py:258-263), delivered to `fused` on that thread. Returns once the cue is computed, so the
    /// caller can start the next frame while this one fuses (frames fuse in order; at most one waits).
    /// Each callback also gets the frame's `FrameContext` for the object layer, built on the fusion thread:
    /// `changed` = that method's fused mask ∧ valid, `frameID` = the method's fused-frame count, D_ren all
    /// NaN when no depth renderer is open.
    func process(_ f: CameraFrame,
                 fused: @escaping (Output, Fused, FrameContext) -> Void = { _, _, _ in },
                 fusedB: @escaping (Output, Fused, FrameContext) -> Void = { _, _, _ in }) throws -> Output {
        var out = Output()
        var clock = DispatchTime.now().uptimeNanoseconds
        func lap(_ stage: String) {
            let now = DispatchTime.now().uptimeNanoseconds
            out.timings.append((stage, Double(now - clock) / 1e6))
            clock = now
        }
        let live = FloatImage(rgb8: f.rgb, width: f.intrinsics.width, height: f.intrinsics.height)
        // The live image's SAM2 embedding does not depend on the pose: encode it on the Neural
        // Engine while XFeat and pose estimation run, so only the render's embedding waits.
        let liveEmb = LiveEmbedding()
        let liveDone = DispatchGroup()
        liveDone.enter()
        DispatchQueue.global(qos: .userInitiated).async { [sam] in
            let s = DispatchTime.now().uptimeNanoseconds
            liveEmb.result = Result { try sam.encode(live) }
            liveEmb.ms = Double(DispatchTime.now().uptimeNanoseconds - s) / 1e6
            liveDone.leave()
        }
        defer { liveDone.wait() }

        let kp = try xfeat.detect(live)
        lap(Stage.xfeat)
        let est = estimator.estimate(kp)
        out.timings += [(Stage.retrieval, est.timings.retrievalMs), (Stage.matching, est.timings.matchingMs),
                        (Stage.pnp, est.timings.pnpMs), (Stage.bundleAdjust, est.timings.bundleAdjustMs)]
        clock = DispatchTime.now().uptimeNanoseconds
        out.anchorPoints = est.nBAInlierComponents / 2
        out.matches = est.nCorrespondences
        guard est.accepted, let pose = est.pose else {
            // O-SCD's gate: more than min_num_inliers (100) residual components after mini-BA, i.e. more
            // than 50 matched points that agree on one camera position.
            out.rejection = "\(est.nBAInlierComponents / 2) of \(est.nCorrespondences) matches with \(beforeLabel) agree on one camera position; more than 50 are needed"
            return out
        }
        out.pose = pose
        let ren = try renderer.render(pose: pose, intrinsics: anchors.intrinsics)
        out.render = ren
        lap(Stage.render)
        out.valid = PreviewValidity.mask(depth: f.depth, confidence: f.confidence, depthWidth: f.depthWidth,
                                         depthHeight: f.depthHeight, intrinsics: f.intrinsics, pose: pose, roi: roi, render: ren)
        lap(Stage.judged)
        let embRender = try sam.encode(ren)
        liveDone.wait()
        let embLive = try liveEmb.result!.get()
        lap(Stage.sam2)
        out.timings.append((Stage.sam2Live, liveEmb.ms))
        let cues = CuesA.compute(live: live, render: ren, embRender: embRender, embLive: embLive, valid: out.valid)
        lap(Stage.cue)
        let snapshot = out
        let W = f.intrinsics.width, H = f.intrinsics.height
        let dLive = CuesB.upsampleDepth(f.depth, confidence: f.confidence, width: f.depthWidth, height: f.depthHeight,
                                        toWidth: W, toHeight: H)
        let dRen: [Float]
        if let referenceDepth {
            dRen = try referenceDepth.expectedDepth(pose: pose, intrinsics: anchors.intrinsics)
            lap(hasB ? Stage.depthRender : Stage.expectedDepth)
        } else {
            dRen = [Float](repeating: .nan, count: W * H)
        }
        // `changed` and `frameID` are filled in on the fusion thread.
        let context = FrameContext(frameID: 0, pose: pose, intrinsics: anchors.intrinsics, changed: [], valid: out.valid,
                                   dLive: dLive, dRen: dRen, live: live, render: ren, embLive: embLive, embRender: embRender)
        submit(to: fusion, pose: pose, cue: cues.combined, valid: out.valid, snapshot: snapshot, context: context,
               counter: \.fusedFrames, deliver: fused)
        if let fusionB {
            let depthCue = CuesB.depthCue(live: dLive, rendered: dRen, valid: out.valid).cue
            let colorCue = CuesB.colorCue(live: live, render: ren, liveDepth: dLive, renderDepth: dRen, valid: out.valid)
            let cB = CuesB.combine(cA: cues.combined, depth: depthCue, color: colorCue, valid: out.valid)
            lap(Stage.cuesB)
            submit(to: fusionB, pose: pose, cue: cB, valid: out.valid, snapshot: snapshot, context: context,
                   counter: \.fusedFramesB, deliver: fusedB)
        }
        lap(Stage.fusionWait)
        return out
    }

    private func submit(to stage: FusionStage, pose: Pose, cue: [Float], valid: [Bool], snapshot: Output, context: FrameContext,
                        counter: ReferenceWritableKeyPath<MethodA, Int>, deliver: @escaping (Output, Fused, FrameContext) -> Void) {
        let K = anchors.intrinsics
        stage.submit { [weak self] model in
            do {
                let t0 = DispatchTime.now().uptimeNanoseconds
                try model.addFrame(pose: pose, intrinsics: K, cue: cue, valid: valid)
                try model.stepFrame()
                let t1 = DispatchTime.now().uptimeNanoseconds
                let changed = try model.changeMap(pose: pose, intrinsics: K).map { $0 > 0.5 }
                let t2 = DispatchTime.now().uptimeNanoseconds
                let n = (self?[keyPath: counter] ?? 0) + 1
                self?[keyPath: counter] = n
                var context = context
                context.frameID = n
                context.changed = zip(changed, valid).map { $0 && $1 }
                deliver(snapshot, Fused(changed: changed, fusionMs: Double(t1 - t0) / 1e6, maskMs: Double(t2 - t1) / 1e6,
                                        fusedFrames: n, gaussians: model.count), context)
            } catch {
                AppLog.write("fusion: \(error.localizedDescription)")
            }
        }
    }

    private func stage(_ method: String) -> FusionStage? { method == "B" ? fusionB : fusion }

    /// Waits for queued fusion, then renders method A's (or B's) change model at `pose`, thresholded at 0.5
    /// as O-SCD does.
    func changeMask(pose: Pose, method: String = "A") throws -> [Bool] {
        let K = anchors.intrinsics
        guard let stage = stage(method) else { return [] }
        return try stage.sync { try $0.changeMap(pose: pose, intrinsics: K).map { $0 > 0.5 } }
    }

    /// Waits for queued fusion, then runs O-SCD --refine: up to 3000 total steps over every frame fused so far.
    func refine(method: String = "A") throws { try stage(method)?.sync { try $0.refine() } }

    func gaussians(method: String = "A") -> Int { (try? stage(method)?.sync { $0.count }) ?? 0 }

    /// The before model's expected depth at `pose` (metres, NaN where uncovered), at the anchors' intrinsics;
    /// nil when no depth renderer is open (neither `withB` nor `withDepth`).
    func expectedDepth(pose: Pose) throws -> [Float]? {
        try referenceDepth?.expectedDepth(pose: pose, intrinsics: anchors.intrinsics)
    }

    /// The before model rendered at `pose`, at the anchors' intrinsics, as `process` renders it (Detect's object
    /// post-pass re-renders kept frames instead of holding every render in memory).
    func render(pose: Pose) throws -> FloatImage {
        try renderer.render(pose: pose, intrinsics: anchors.intrinsics)
    }

    /// Waits until every submitted frame has been fused (A and B).
    func finishFusion() throws {
        try fusion.sync { _ in }
        try fusionB?.sync { _ in }
    }
}

/// The live image's embedding, filled in on a background queue while the pose is estimated.
private final class LiveEmbedding {
    var result: Result<[Float], Error>?
    var ms = 0.0
}
