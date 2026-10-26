import ARKit
import Combine
import OSCDCore
import RealityKit
import UIKit

/// Live view: ARKit supplies frames and LiDAR; PreviewWorker runs Method A against the before-scan.
final class PreviewController: NSObject, ObservableObject, ARSessionDelegate {
    var arView: ARView { SharedAR.view }
    @Published var status = "Press Start to compare what the camera sees with the before-scan."
    @Published var overlay: UIImage?
    @Published var reference: UIImage?
    @Published var headline = ""
    @Published var details = ""
    @Published var running = false
    /// The object layer's newest report, and the camera of the newest placed frame (for projecting it).
    @Published var objects: ObjectReport?
    @Published var lastPose: Pose?
    @Published var lastIntrinsics: Intrinsics?

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
        AppLog.write("live: session.run format=\(config.videoFormat.imageResolution) camera=\(AppLog.cameraAuthorization())")
        firstFrameLogged = false
        arView.session.run(config)
    }

    func stopSession() {
        stop()
        AppLog.write("live: session.pause")
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
                if let objects = r.objects { self.objects = objects }
                if let pose = r.pose { self.lastPose = pose; self.lastIntrinsics = r.intrinsics }
            }
        }
        worker = w
        running = true
        overlay = nil; reference = nil; headline = ""; details = ""
        objects = nil; lastPose = nil; lastIntrinsics = nil
        w.start(before: before)
    }

    func stop() {
        worker?.stop()
        worker = nil
        running = false
    }

    private var firstFrameLogged = false

    func session(_ session: ARSession, didFailWithError error: Error) {
        AppLog.write("live: didFailWithError \(error.localizedDescription) \((error as NSError).code)")
    }

    func sessionWasInterrupted(_ session: ARSession) { AppLog.write("live: interrupted") }

    func sessionInterruptionEnded(_ session: ARSession) { AppLog.write("live: interruption ended") }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        AppLog.write("live: tracking \(camera.trackingState)")
    }

    private func logFirstFrame(_ frame: ARFrame) {
        guard !firstFrameLogged else { return }
        firstFrameLogged = true
        // Mean luma of a sparse sample: ~0 means the lens sees nothing (covered or face down).
        let buf = frame.capturedImage
        CVPixelBufferLockBaseAddress(buf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
        let w = CVPixelBufferGetWidthOfPlane(buf, 0), h = CVPixelBufferGetHeightOfPlane(buf, 0)
        let row = CVPixelBufferGetBytesPerRowOfPlane(buf, 0)
        let y = CVPixelBufferGetBaseAddressOfPlane(buf, 0)!.assumingMemoryBound(to: UInt8.self)
        var sum = 0, n = 0
        for v in stride(from: 0, to: h, by: 40) { for u in stride(from: 0, to: w, by: 40) { sum += Int(y[v * row + u]); n += 1 } }
        AppLog.write("live: first frame, mean luma \(sum / max(n, 1)) of 255")
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        logFirstFrame(frame)
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
