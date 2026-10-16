import ARKit
import Combine
import OSCDCore
import RealityKit
import UIKit

/// Live view: ARKit supplies frames and LiDAR; PreviewWorker runs Method A against the before-scan.
final class PreviewController: NSObject, ObservableObject, ARSessionDelegate {
    let arView = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
    @Published var status = "Press Start to compare what the camera sees with the before-scan."
    @Published var overlay: UIImage?
    @Published var reference: UIImage?
    @Published var headline = ""
    @Published var details = ""
    @Published var running = false

    private var worker: PreviewWorker?

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
    }

    func stopSession() {
        stop()
        arView.session.pause()
    }

    func start(before: ScanInfo) {
        let w = PreviewWorker()
        w.onStatus = { s in DispatchQueue.main.async { self.status = s } }
        w.onResult = { r in
            let overlay = Self.image(r.overlayRGBA, r.width, r.height)
            let reference = r.renderRGBA.flatMap { Self.image($0, r.width, r.height) }
            DispatchQueue.main.async {
                self.overlay = overlay
                if let reference { self.reference = reference }
                self.headline = r.headline
                self.details = r.details
            }
        }
        worker = w
        running = true
        overlay = nil; reference = nil; headline = ""; details = ""
        w.start(before: before)
    }

    func stop() {
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
            status = "This camera format (\(Int(res.width))×\(Int(res.height))) is not supported."
            return
        }
        let K = frame.camera.intrinsics
        let intrinsics = Intrinsics.fromARKit(fx: K[0][0], fy: K[1][1], cx: K[2][0], cy: K[2][1],
                                              captureWidth: Int(res.width), captureHeight: Int(res.height),
                                              workingWidth: working.width, workingHeight: working.height)
        let (depth, confidence) = CaptureController.copyDepth(sceneDepth)
        worker.submit(CameraFrame(rgb: CaptureController.workingRGB(frame.capturedImage, factor: factor),
                                  depth: depth, confidence: confidence,
                                  depthWidth: CVPixelBufferGetWidth(sceneDepth.depthMap),
                                  depthHeight: CVPixelBufferGetHeight(sceneDepth.depthMap),
                                  intrinsics: intrinsics))
    }

    /// Sensor-orientation RGBA → portrait UIImage.
    static func image(_ rgba: [UInt8], _ w: Int, _ h: Int) -> UIImage? {
        cgImage(rgba, w, h).map { UIImage(cgImage: $0, scale: 1, orientation: .right) }
    }

    static func cgImage(_ rgba: [UInt8], _ w: Int, _ h: Int) -> CGImage? {
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                       space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}
