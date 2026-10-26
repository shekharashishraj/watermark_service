import Combine
import OSCDCore
import SwiftUI
import simd

/// Renders a scan's 3D model on its own thread (splatx sessions are single-thread). Only the latest
/// requested view is rendered, so dragging never queues up frames.
///
/// Two ways to look at the model:
/// - **Walk** (default): stand where the phone stood. The view is a keyframe's own camera (pose and
///   intrinsics), so it frames what that photo framed, with a small look-around on top. A splat is
///   sharp near the path it was captured on and smeared away from it; this mode stays on the path.
/// - **Orbit**: the free camera around the scan's focus point (drag to orbit, pinch to zoom).
final class ModelViewer: ObservableObject {
    enum Mode: String, CaseIterable { case walk = "Walk", orbit = "Orbit" }

    /// Portrait render for Orbit, about a 60° vertical field of view.
    static let orbitIntrinsics = Intrinsics(f: 360 / tan(Float.pi / 6), cx: 269.5, cy: 359.5, width: 540, height: 720)

    /// A keyframe's camera plus a look-around (radians, about the camera's own axes).
    struct Walk: Equatable {
        var index: Int
        var yaw: Float = 0
        var pitch: Float = 0
        static let maxYaw: Float = 0.8, maxPitch: Float = 0.5
    }

    /// What the render thread draws: a world-to-camera pose with its intrinsics. `portrait` is true
    /// when the image is in sensor orientation and must be turned upright for the screen.
    struct Request {
        var pose: Pose
        var intrinsics: Intrinsics
        var portrait: Bool
        var label: String
    }

    @Published private(set) var image: UIImage?
    @Published private(set) var status = "Opening the 3D model…"
    @Published private(set) var renderMs = 0.0
    @Published private(set) var mode = Mode.walk
    @Published private(set) var camera: OrbitCamera?
    @Published private(set) var walk: Walk?
    private(set) var home: OrbitCamera?
    private(set) var keyframeCount = 0

    private var keyframes: [simd_float4x4] = []
    private var walkIntrinsics: Intrinsics?
    private let cond = NSCondition()
    private var requested: Request?
    private var stopping = false

    func open(scan: ScanInfo) {
        do {
            let store = try SessionStore(open: scan.url)
            let m = store.manifest
            guard !m.keyframes.isEmpty, let box = try SceneRegion.region(of: store) else {
                throw PreviewError.message("\(scan.label) has no frames with depth.")
            }
            keyframes = m.keyframes.map { simd_float4x4(columnMajor: $0.arkitCameraToWorld) }
            keyframeCount = keyframes.count
            // Walk renders at 1.5× the working size with the scan's own camera.
            let K = m.intrinsics, s: Float = 1.5
            walkIntrinsics = Intrinsics(f: K.f * s, cx: (K.cx + 0.5) * s - 0.5, cy: (K.cy + 0.5) * s - 0.5,
                                        width: Int(Float(K.width) * s), height: Int(Float(K.height) * s))
            let target = SceneRegion.focus(arkitCameraToWorld: keyframes, within: box)
            let kf = keyframes[keyframes.count / 2]
            let start = OrbitCamera(lookingFrom: Pose.fromARKitCameraToWorld(kf), at: target)
            home = start
            camera = start
            // `-viewWalk N` opens Walk at keyframe N; `-viewYawDeg` / `-viewPitchDeg` turn the first Orbit view
            // (checks without touch).
            let d = UserDefaults.standard
            if d.object(forKey: "viewYawDeg") != nil || d.object(forKey: "viewPitchDeg") != nil {
                var first = start
                first.orbit(dYaw: Float(d.double(forKey: "viewYawDeg")) * .pi / 180, dPitch: Float(d.double(forKey: "viewPitchDeg")) * .pi / 180)
                mode = .orbit
                set(first)
            } else {
                let n = d.object(forKey: "viewWalk") != nil ? d.integer(forKey: "viewWalk") : keyframes.count / 2
                setWalk(Walk(index: n))
            }
        } catch {
            status = "Could not open: \(error.localizedDescription)"
            return
        }
        let ply = scan.url.appendingPathComponent(ScanCatalog.referenceFile).path
        let thread = Thread { [weak self] in self?.run(ply: ply) }
        thread.name = "oscd.viewer"
        thread.stackSize = 16 << 20
        thread.qualityOfService = .userInteractive
        thread.start()
    }

    func setMode(_ m: Mode) {
        guard m != mode else { return }
        mode = m
        switch m {
        case .walk: setWalk(walk ?? Walk(index: keyframes.count / 2))
        case .orbit: if let cam = camera { set(cam) }
        }
    }

    /// Orbit mode.
    func set(_ cam: OrbitCamera) {
        camera = cam
        request(Request(pose: cam.pose, intrinsics: Self.orbitIntrinsics, portrait: false,
                        label: String(format: "orbit distance %.2f m pitch %.0f° yaw %.0f°", cam.distance, cam.pitch * 180 / .pi, cam.yaw * 180 / .pi)))
    }

    /// Walk mode.
    func setWalk(_ w: Walk) {
        guard !keyframes.isEmpty, let K = walkIntrinsics else { return }
        var w = w
        w.index = min(max(w.index, 0), keyframes.count - 1)
        w.yaw = min(max(w.yaw, -Walk.maxYaw), Walk.maxYaw)
        w.pitch = min(max(w.pitch, -Walk.maxPitch), Walk.maxPitch)
        walk = w
        let base = Pose.fromARKitCameraToWorld(keyframes[w.index])
        // Look-around about the camera's own axes (OpenCV: x right, y down, z forward): yaw about y, pitch about x.
        let Rd = simd_float3x3(simd_quatf(angle: w.yaw, axis: SIMD3(0, 1, 0)) * simd_quatf(angle: w.pitch, axis: SIMD3(1, 0, 0)))
        let pose = Pose(R: Rd * base.R, t: Rd * base.t)
        request(Request(pose: pose, intrinsics: K, portrait: true,
                        label: String(format: "walk keyframe %d/%d yaw %.0f° pitch %.0f°", w.index + 1, keyframes.count, w.yaw * 180 / .pi, w.pitch * 180 / .pi)))
    }

    func step(_ delta: Int) {
        guard let w = walk else { return }
        setWalk(Walk(index: w.index + delta, yaw: w.yaw, pitch: w.pitch))
    }

    private func request(_ r: Request) {
        cond.lock(); requested = r; cond.signal(); cond.unlock()
    }

    func close() {
        cond.lock(); stopping = true; cond.signal(); cond.unlock()
    }

    private func run(ply: String) {
        do {
            guard let kernels = Bundle.main.url(forResource: "SplatXKernelSrc", withExtension: nil) else {
                throw PreviewError.message("The app is missing the splatx kernel sources; rebuild it.")
            }
            let bytes = (try? FileManager.default.attributesOfItem(atPath: ply)[.size] as? Int) ?? 0
            AppLog.write("viewer: opening \(ply.split(separator: "/").suffix(2).joined(separator: "/")) (\(bytes / 1_000_000) MB)", sync: true)
            let renderer = try SplatRenderer(plyPath: ply, kernelSourceDir: kernels.path)
            AppLog.write("viewer: open", sync: true)
            DispatchQueue.main.async { self.status = "" }
            var renders = 0, lastLog = Date()
            defer { AppLog.write("viewer: closed after \(renders) renders", sync: true) }
            // Each view's UIImage/CGImage/Data are autoreleased on this thread, which never exits: without a
            // pool per view they accumulated 1.56 MB (540×720×4) per render until iOS killed the app.
            while true { try autoreleasepool {
                cond.lock()
                while requested == nil && !stopping { cond.wait() }
                if stopping { cond.unlock(); return }
                let r = requested!
                requested = nil
                cond.unlock()
                let t0 = DispatchTime.now().uptimeNanoseconds
                // Each view is logged before it renders (synchronously), so a crash inside the render leaves its camera behind.
                renders += 1
                if renders <= 3 || Date().timeIntervalSince(lastLog) > 1 {
                    lastLog = Date()
                    AppLog.write("viewer: render \(renders)  \(r.label)  mem \(AppLog.residentMB()) MB", sync: true)
                }
                let K = r.intrinsics
                let rgba = try renderer.renderRGBA(pose: r.pose, intrinsics: K)
                let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
                let image = r.portrait ? PreviewController.image(rgba, K.width, K.height)
                                       : PreviewController.cgImage(rgba, K.width, K.height).map { UIImage(cgImage: $0) }
                DispatchQueue.main.async { self.image = image; self.renderMs = ms }
            } }
        } catch {
            DispatchQueue.main.async { self.status = "Could not render: \(error.localizedDescription)" }
        }
    }
}

/// Walk: slider / ◀ ▶ move along the captured path, drag looks around, double-tap looks straight again.
/// Orbit: drag to orbit, pinch to zoom, double-tap to go back to the starting view.
struct ModelViewerScreen: View {
    let scan: ScanInfo
    @Environment(\.dismiss) private var dismiss
    @StateObject private var viewer = ModelViewer()
    @State private var lastDrag = CGSize.zero
    @State private var lastScale: CGFloat = 1
    @State private var slider = 0.0

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if let image = viewer.image {
                Image(uiImage: image).resizable().scaledToFit()
            }
            VStack {
                HStack {
                    Button { dismiss() } label: {
                        Image(systemName: "chevron.left").font(.system(size: 20, weight: .medium)).foregroundColor(.white)
                    }
                    Spacer()
                    Text("\(scan.label) · 3D model").font(.system(size: 17, weight: .semibold)).foregroundColor(.white)
                    Spacer()
                    Image(systemName: "chevron.left").opacity(0)
                }
                .padding(.horizontal, 16).padding(.top, 8)
                Spacer()
                if !viewer.status.isEmpty {
                    Text(viewer.status).font(.callout).foregroundColor(.white)
                }
                if viewer.mode == .walk, let w = viewer.walk, viewer.keyframeCount > 1 {
                    HStack(spacing: 12) {
                        Button { viewer.step(-1) } label: { Image(systemName: "chevron.left.circle.fill").font(.system(size: 28)) }
                        Slider(value: $slider, in: 0...Double(viewer.keyframeCount - 1), step: 1)
                            .onChange(of: slider) { _, v in
                                if Int(v) != w.index { viewer.setWalk(ModelViewer.Walk(index: Int(v), yaw: w.yaw, pitch: w.pitch)) }
                            }
                        Button { viewer.step(1) } label: { Image(systemName: "chevron.right.circle.fill").font(.system(size: 28)) }
                    }
                    .tint(.white).foregroundColor(.white)
                    .padding(.horizontal, 20)
                    Text("Frame \(w.index + 1) of \(viewer.keyframeCount) · drag to look around")
                        .font(.caption).foregroundColor(.white.opacity(0.7))
                }
                Picker("Mode", selection: Binding(get: { viewer.mode }, set: { viewer.setMode($0) })) {
                    ForEach(ModelViewer.Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented).frame(width: 180).padding(.top, 6)
                Text(String(format: viewer.mode == .orbit ? "Drag to orbit · pinch to zoom · double-tap to reset · %.0f ms per view"
                                                          : "%.0f ms per view", viewer.renderMs))
                    .font(.caption).foregroundColor(.white.opacity(0.7))
                    .padding(.bottom, 16)
            }
        }
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { g in
                    let dx = Float(g.translation.width - lastDrag.width), dy = Float(g.translation.height - lastDrag.height)
                    lastDrag = g.translation
                    switch viewer.mode {
                    case .orbit:
                        guard var cam = viewer.camera else { return }
                        cam.orbit(dYaw: -dx * 0.008, dPitch: dy * 0.008)
                        viewer.set(cam)
                    case .walk:
                        guard var w = viewer.walk else { return }
                        // The screen is the sensor image turned 90°: a horizontal finger move turns the camera about
                        // its sensor-vertical axis.
                        w.yaw += dy * 0.004; w.pitch += dx * 0.004
                        viewer.setWalk(w)
                    }
                }
                .onEnded { _ in lastDrag = .zero }
                .simultaneously(with: MagnificationGesture()
                    .onChanged { s in
                        guard viewer.mode == .orbit, var cam = viewer.camera else { return }
                        cam.zoom(Float(s / lastScale))
                        lastScale = s
                        viewer.set(cam)
                    }
                    .onEnded { _ in lastScale = 1 })
        )
        .onTapGesture(count: 2) {
            switch viewer.mode {
            case .orbit: if let home = viewer.home { viewer.set(home) }
            case .walk: if let w = viewer.walk { viewer.setWalk(ModelViewer.Walk(index: w.index)) }
            }
        }
        .toolbar(.hidden, for: .navigationBar)
        .onAppear {
            viewer.open(scan: scan)
            slider = Double(viewer.walk?.index ?? 0)
            // `-viewSpin N`: orbit 2° per view for N views; `-viewWalkSpin N`: step N keyframes (checks without touch).
            let d = UserDefaults.standard
            let spin = d.integer(forKey: "viewSpin"), walkSpin = d.integer(forKey: "viewWalkSpin")
            if spin > 0 {
                viewer.setMode(.orbit)
                var left = spin
                Timer.scheduledTimer(withTimeInterval: 0.02, repeats: true) { t in
                    guard left > 0, var cam = viewer.camera else { t.invalidate(); return }
                    cam.orbit(dYaw: 2 * .pi / 180, dPitch: 0); viewer.set(cam); left -= 1
                }
            } else if walkSpin > 0 {
                var left = walkSpin
                Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { t in
                    guard left > 0, let w = viewer.walk else { t.invalidate(); return }
                    viewer.setWalk(ModelViewer.Walk(index: (w.index + 1) % max(viewer.keyframeCount, 1))); left -= 1
                }
            }
        }
        .onChange(of: viewer.walk?.index ?? 0) { _, i in slider = Double(i) }
        .onDisappear { viewer.close() }
    }
}
