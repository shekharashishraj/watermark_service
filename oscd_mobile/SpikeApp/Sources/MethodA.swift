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
        static let perFrame = [xfeat, retrieval, matching, pnp, bundleAdjust, render, judged, sam2, sam2Live, cue, fusion, mask]

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

    let anchors: AnchorSet
    let roi: AABB
    let beforeLabel: String
    private(set) var setupTimings = StageTimings()
    private(set) var fusedFrames = 0
    private let xfeat: XFeatModel
    private let sam: SAM2Encoder
    private let estimator: PoseEstimator
    private let renderer: SplatRenderer
    private let fusion: ChangeFusion

    var gaussians: Int { fusion.count }
    var steps: Int { fusion.totalSteps }

    init(beforeScan: URL, beforeLabel: String, status: (String) -> Void) throws {
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
        guard let roi = store.manifest.roi else { throw PreviewError.message("\(beforeLabel) has no table region.") }
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
        fusion = try ChangeFusion(plyPath: plyURL.path, kernelSourceDir: kernelURL.path,
                                  referenceCentres: views.map { $0.pose.centre })
        lap(Stage.openFusion)
        setupTimings = setup
    }

    /// XFeat → O-SCD pose → render → judged area → SAM2 + pixel cue → fusion (16 steps) → mask
    /// (change model rendered at this pose, value > 0.5, oscd.py:258-263).
    func process(_ f: CameraFrame) throws -> Output {
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
            out.rejection = est.nCorrespondences < 150
                ? "only \(est.nCorrespondences) points match \(beforeLabel)"
                : "\(est.nCorrespondences) points match \(beforeLabel), but they don't agree on where the camera is"
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
        try fusion.addFrame(pose: pose, intrinsics: anchors.intrinsics, cue: cues.combined, valid: out.valid)
        try fusion.stepFrame()
        fusedFrames += 1
        lap(Stage.fusion)
        out.changed = try changeMask(pose: pose)
        lap(Stage.mask)
        return out
    }

    /// The change model rendered at `pose`, thresholded at 0.5 as O-SCD does.
    func changeMask(pose: Pose) throws -> [Bool] {
        try fusion.changeMap(pose: pose, intrinsics: anchors.intrinsics).map { $0 > 0.5 }
    }

    /// O-SCD --refine: up to 3000 total steps over every frame fused so far.
    func refine() throws { try fusion.refine() }
}

/// The live image's embedding, filled in on a background queue while the pose is estimated.
private final class LiveEmbedding {
    var result: Result<[Float], Error>?
    var ms = 0.0
}
