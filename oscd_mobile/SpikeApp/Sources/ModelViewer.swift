import Combine
import OSCDCore
import SwiftUI
import simd

/// Renders a scan's 3D model from an orbiting camera on its own thread (splatx sessions are
/// single-thread). Only the latest requested camera is rendered, so dragging never queues up frames.
final class ModelViewer: ObservableObject {
    /// Portrait render, about a 60° vertical field of view.
    static let intrinsics = Intrinsics(f: 360 / tan(Float.pi / 6), cx: 269.5, cy: 359.5, width: 540, height: 720)

    @Published private(set) var image: UIImage?
    @Published private(set) var status = "Opening the 3D model…"
    @Published private(set) var renderMs = 0.0
    @Published private(set) var camera: OrbitCamera?
    private(set) var home: OrbitCamera?

    private let cond = NSCondition()
    private var requested: OrbitCamera?
    private var stopping = false

    func open(scan: ScanInfo) {
        do {
            let store = try SessionStore(open: scan.url)
            let m = store.manifest
            guard let roi = m.roi, !m.keyframes.isEmpty else { throw PreviewError.message("\(scan.label) has no table region.") }
            // The table region was placed 2 cm below the tapped table point; aim a little above the table.
            let c = (roi.min + roi.max) / 2
            let target = SIMD3(c.x, roi.min.y + 0.05, c.z)
            let kf = m.keyframes[m.keyframes.count / 2]
            let start = OrbitCamera(lookingFrom: Pose.fromARKitCameraToWorld(simd_float4x4(columnMajor: kf.arkitCameraToWorld)), at: target)
            home = start
            // `-viewYawDeg` / `-viewPitchDeg` turn the first view (checks without touch).
            var first = start
            let d = UserDefaults.standard
            first.orbit(dYaw: Float(d.double(forKey: "viewYawDeg")) * .pi / 180, dPitch: Float(d.double(forKey: "viewPitchDeg")) * .pi / 180)
            set(first)
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

    func set(_ cam: OrbitCamera) {
        camera = cam
        cond.lock(); requested = cam; cond.signal(); cond.unlock()
    }

    func close() {
        cond.lock(); stopping = true; cond.signal(); cond.unlock()
    }

    private func run(ply: String) {
        do {
            guard let kernels = Bundle.main.url(forResource: "SplatXKernelSrc", withExtension: nil) else {
                throw PreviewError.message("The app is missing the splatx kernel sources; rebuild it.")
            }
            let renderer = try SplatRenderer(plyPath: ply, kernelSourceDir: kernels.path)
            DispatchQueue.main.async { self.status = "" }
            while true {
                cond.lock()
                while requested == nil && !stopping { cond.wait() }
                if stopping { cond.unlock(); return }
                let cam = requested!
                requested = nil
                cond.unlock()
                let t0 = DispatchTime.now().uptimeNanoseconds
                let K = Self.intrinsics
                let rgba = try renderer.renderRGBA(pose: cam.pose, intrinsics: K)
                let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
                let image = PreviewController.cgImage(rgba, K.width, K.height).map { UIImage(cgImage: $0) }
                DispatchQueue.main.async { self.image = image; self.renderMs = ms }
            }
        } catch {
            DispatchQueue.main.async { self.status = "Could not render: \(error.localizedDescription)" }
        }
    }
}

/// Drag to orbit around the table, pinch to zoom, double-tap to go back to the starting view.
struct ModelViewerScreen: View {
    let scan: ScanInfo
    @Environment(\.dismiss) private var dismiss
    @StateObject private var viewer = ModelViewer()
    @State private var lastDrag = CGSize.zero
    @State private var lastScale: CGFloat = 1

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
                Text(String(format: "Drag to orbit · pinch to zoom · double-tap to reset · %.0f ms per view", viewer.renderMs))
                    .font(.caption).foregroundColor(.white.opacity(0.7))
                    .padding(.bottom, 16)
            }
        }
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { g in
                    guard var cam = viewer.camera else { return }
                    let dx = Float(g.translation.width - lastDrag.width), dy = Float(g.translation.height - lastDrag.height)
                    lastDrag = g.translation
                    cam.orbit(dYaw: -dx * 0.008, dPitch: dy * 0.008)
                    viewer.set(cam)
                }
                .onEnded { _ in lastDrag = .zero }
                .simultaneously(with: MagnificationGesture()
                    .onChanged { s in
                        guard var cam = viewer.camera else { return }
                        cam.zoom(Float(s / lastScale))
                        lastScale = s
                        viewer.set(cam)
                    }
                    .onEnded { _ in lastScale = 1 })
        )
        .onTapGesture(count: 2) { if let home = viewer.home { viewer.set(home) } }
        .toolbar(.hidden, for: .navigationBar)
        .onAppear { viewer.open(scan: scan) }
        .onDisappear { viewer.close() }
    }
}
