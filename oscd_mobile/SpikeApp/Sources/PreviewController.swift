import ARKit
import Combine
import OSCDCore
import RealityKit
import UIKit

/// Live Method A preview: ARKit supplies frames and LiDAR; PreviewWorker does the O-SCD work.
/// Assets come from Documents/preview/: reference.ply (trained on the Mac) and before/ (the before-scan).
final class PreviewController: NSObject, ObservableObject, ARSessionDelegate {
    let arView = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
    @Published var status = "Press Start. The reference and the before-scan must be in Documents/preview."
    @Published var overlay: UIImage?
    @Published var reference: UIImage?
    @Published var hud = ""
    @Published var running = false

    private var worker: PreviewWorker?
    private var sessionRunning = false

    static var assets: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("preview")
    }

    func startSession() {
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = [.sceneDepth]
        config.isAutoFocusEnabled = false
        if let format = ARWorldTrackingConfiguration.supportedVideoFormats.first(where: {
            $0.imageResolution == CGSize(width: 1920, height: 1440) }) {
            config.videoFormat = format
        }
        arView.session.delegate = self
        arView.session.run(config)
        sessionRunning = true
    }

    func stopSession() {
        stopPipeline()
        arView.session.pause()
        sessionRunning = false
    }

    func toggle() { running ? stopPipeline() : startPipeline() }

    private func startPipeline() {
        let w = PreviewWorker()
        w.onStatus = { s in DispatchQueue.main.async { self.status = s } }
        w.onResult = { r in
            let overlay = Self.image(r.overlayRGBA, r.width, r.height)
            let reference = r.renderRGBA.flatMap { Self.image($0, r.width, r.height) }
            DispatchQueue.main.async {
                self.overlay = overlay
                if let reference { self.reference = reference }
                self.hud = r.line
            }
        }
        worker = w
        running = true
        w.start(assets: Self.assets)
    }

    func refine() { worker?.requestRefine() }

    private func stopPipeline() {
        worker?.stop()
        worker = nil
        running = false
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard let worker, worker.wantsFrame, let sceneDepth = frame.sceneDepth else { return }
        let res = frame.camera.imageResolution
        let working = CaptureController.working
        guard let factor = CameraImage.areaFactor(captureWidth: Int(res.width), captureHeight: Int(res.height),
                                                  workingWidth: working.width, workingHeight: working.height) else {
            status = "Camera format \(Int(res.width))×\(Int(res.height)) is not supported."
            return
        }
        let K = frame.camera.intrinsics
        let intrinsics = Intrinsics.fromARKit(fx: K[0][0], fy: K[1][1], cx: K[2][0], cy: K[2][1],
                                              captureWidth: Int(res.width), captureHeight: Int(res.height),
                                              workingWidth: working.width, workingHeight: working.height)
        let (depth, confidence) = CaptureController.copyDepth(sceneDepth)
        worker.submit(LiveFrame(rgb: CaptureController.workingRGB(frame.capturedImage, factor: factor),
                                depth: depth, confidence: confidence,
                                depthWidth: CVPixelBufferGetWidth(sceneDepth.depthMap),
                                depthHeight: CVPixelBufferGetHeight(sceneDepth.depthMap),
                                intrinsics: intrinsics))
    }

    /// Sensor-orientation RGBA → portrait UIImage.
    static func image(_ rgba: [UInt8], _ w: Int, _ h: Int) -> UIImage? {
        let data = Data(rgba) as CFData
        guard let provider = CGDataProvider(data: data),
              let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                               space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                               provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else { return nil }
        return UIImage(cgImage: cg, scale: 1, orientation: .right)
    }
}
