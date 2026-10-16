import ARKit
import AVFoundation
import Combine
import OSCDCore
import RealityKit
import UIKit

/// Records reference-scan sessions: keyframes (RGB at 480×360, LiDAR depth + confidence, ARKit pose).
/// ARSession delegate callbacks arrive on the main thread; files are written on a serial queue.
final class CaptureController: NSObject, ObservableObject, ARSessionDelegate {
    static let working = (width: 480, height: 360)

    let arView = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
    @Published var status = "Tap the table to place the scan region"
    @Published var keyframeCount = 0
    @Published var recording = false

    private var selector = KeyframeSelector()
    private var store: SessionStore?
    private var roi: AABB?
    private let writeQueue = DispatchQueue(label: "oscd.capture.write")

    func start() {
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = [.sceneDepth]
        config.planeDetection = [.horizontal]
        config.isAutoFocusEnabled = false
        // 1920×1440 maps exactly onto the 480×360 working size (4×4 blocks).
        if let format = ARWorldTrackingConfiguration.supportedVideoFormats.first(where: {
            $0.imageResolution == CGSize(width: 1920, height: 1440) }) {
            config.videoFormat = format
        }
        arView.session.delegate = self
        AppLog.write("capture: session.run format=\(config.videoFormat.imageResolution) camera=\(AppLog.cameraAuthorization())")
        firstFrameLogged = false
        arView.session.run(config)
        started = true
    }

    private var started = false

    /// The Preview tab needs the camera: pause this session while it is shown.
    func pause() {
        guard started else { return }
        if recording { toggleRecording() }
        AppLog.write("capture: session.pause")
        arView.session.pause()
    }

    func resume() {
        guard started else { return }
        start()
    }

    func placeROI(at point: CGPoint) {
        guard let query = arView.makeRaycastQuery(from: point, allowing: .existingPlaneGeometry, alignment: .horizontal),
              let hit = arView.session.raycast(query).first else {
            status = "No table plane there yet. Move the phone slowly over the table, then tap again."
            return
        }
        let c = hit.worldTransform.columns.3
        let p = SIMD3<Float>(c.x, c.y, c.z)
        roi = AABB(min: p - SIMD3(0.4, 0.02, 0.4), max: p + SIMD3(0.4, 0.35, 0.4))      // ARKit world is y-up
        status = "Table region placed. Press Record and sweep slowly around it."
    }

    func toggleRecording() {
        if recording {
            recording = false
            store = nil
            status = "Saved \(keyframeCount) frames. The scan is in the Scans tab."
            return
        }
        guard roi != nil else { status = "Tap the table to place the scan region first"; return }
        lockCamera()
        selector = KeyframeSelector()
        keyframeCount = 0
        recording = true
        status = "Recording. Sweep slowly around the table, about 30 cm away, phone upright."
    }

    /// Exposure, white balance and focus stay locked for the whole session (iOS 16+).
    private func lockCamera() {
        guard let device = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera else { return }
        do {
            try device.lockForConfiguration()
            if device.isExposureModeSupported(.locked) { device.exposureMode = .locked }
            if device.isWhiteBalanceModeSupported(.locked) { device.whiteBalanceMode = .locked }
            if device.isFocusModeSupported(.locked) { device.focusMode = .locked }
            device.unlockForConfiguration()
        } catch {
            status = "Could not lock the camera: \(error.localizedDescription)"
        }
    }

    private var firstFrameLogged = false

    func session(_ session: ARSession, didFailWithError error: Error) {
        AppLog.write("capture: didFailWithError \(error.localizedDescription) \((error as NSError).code)")
    }

    func sessionWasInterrupted(_ session: ARSession) { AppLog.write("capture: interrupted") }

    func sessionInterruptionEnded(_ session: ARSession) { AppLog.write("capture: interruption ended") }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        AppLog.write("capture: tracking \(camera.trackingState)")
    }

    private func logFirstFrame() {
        guard !firstFrameLogged else { return }
        firstFrameLogged = true
        AppLog.write("capture: first frame")
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        logFirstFrame()
        guard recording, let sceneDepth = frame.sceneDepth, let roi else { return }
        var normal = false
        if case .normal = frame.camera.trackingState { normal = true }
        let pose = Pose.fromARKitCameraToWorld(frame.camera.transform)
        guard selector.consider(pose: pose, time: frame.timestamp, trackingNormal: normal) else { return }
        let res = frame.camera.imageResolution
        guard let factor = CameraImage.areaFactor(captureWidth: Int(res.width), captureHeight: Int(res.height),
                                                  workingWidth: Self.working.width, workingHeight: Self.working.height) else {
            status = "Camera format \(Int(res.width))×\(Int(res.height)) is not supported; it must be 4:3 at a multiple of 480×360."
            return
        }
        if store == nil {
            do {
                store = try makeStore(frame: frame, depth: sceneDepth, roi: roi)
            } catch {
                status = "Could not create the session folder: \(error.localizedDescription)"
                recording = false
                return
            }
        }
        guard let store else { return }
        let rgb = Self.workingRGB(frame.capturedImage, factor: factor)
        let (depth, confidence) = Self.copyDepth(sceneDepth)
        let record = FrameRecord(id: keyframeCount, timestamp: frame.timestamp, arkitCameraToWorld: frame.camera.transform.columnMajor)
        keyframeCount += 1
        writeQueue.async {
            do {
                try store.append(.keyframes, record: record, rgb: rgb, depth: depth, confidence: confidence)
            } catch {
                DispatchQueue.main.async { self.status = "Saving frame \(record.id) failed: \(error.localizedDescription)" }
            }
        }
    }

    private func makeStore(frame: ARFrame, depth: ARDepthData, roi: AABB) throws -> SessionStore {
        let K = frame.camera.intrinsics
        let res = frame.camera.imageResolution
        let intrinsics = Intrinsics.fromARKit(fx: K[0][0], fy: K[1][1], cx: K[2][0], cy: K[2][1],
                                              captureWidth: Int(res.width), captureHeight: Int(res.height),
                                              workingWidth: Self.working.width, workingHeight: Self.working.height)
        var manifest = SessionManifest(deviceModel: Self.deviceModel(), captureWidth: Int(res.width), captureHeight: Int(res.height),
                                       depthWidth: CVPixelBufferGetWidth(depth.depthMap), depthHeight: CVPixelBufferGetHeight(depth.depthMap),
                                       intrinsics: intrinsics)
        manifest.roi = roi
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let name = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        return try SessionStore(create: docs.appendingPathComponent("sessions/\(name)"), manifest: manifest)
    }

    static func deviceModel() -> String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
    }

    static func workingRGB(_ buffer: CVPixelBuffer, factor: Int) -> [UInt8] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidthOfPlane(buffer, 0), h = CVPixelBufferGetHeightOfPlane(buffer, 0)
        let y = CVPixelBufferGetBaseAddressOfPlane(buffer, 0)!.assumingMemoryBound(to: UInt8.self)
        let cbcr = CVPixelBufferGetBaseAddressOfPlane(buffer, 1)!.assumingMemoryBound(to: UInt8.self)
        return CameraImage.rgbArea(y: y, yStride: CVPixelBufferGetBytesPerRowOfPlane(buffer, 0),
                                   cbcr: cbcr, cbcrStride: CVPixelBufferGetBytesPerRowOfPlane(buffer, 1),
                                   width: w, height: h, factor: factor)
    }

    static func copyDepth(_ data: ARDepthData) -> ([Float], [UInt8]) {
        let dm = data.depthMap
        CVPixelBufferLockBaseAddress(dm, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(dm, .readOnly) }
        let w = CVPixelBufferGetWidth(dm), h = CVPixelBufferGetHeight(dm)
        var depth = [Float](repeating: 0, count: w * h), conf = [UInt8](repeating: 0, count: w * h)
        let dp = CVPixelBufferGetBaseAddress(dm)!, dr = CVPixelBufferGetBytesPerRow(dm)
        for y in 0..<h {
            let row = (dp + y * dr).assumingMemoryBound(to: Float32.self)
            for x in 0..<w { depth[y * w + x] = row[x] }
        }
        if let cm = data.confidenceMap {
            CVPixelBufferLockBaseAddress(cm, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(cm, .readOnly) }
            let cp = CVPixelBufferGetBaseAddress(cm)!, cr = CVPixelBufferGetBytesPerRow(cm)
            for y in 0..<h {
                let row = (cp + y * cr).assumingMemoryBound(to: UInt8.self)
                for x in 0..<w { conf[y * w + x] = row[x] }
            }
        }
        return (depth, conf)
    }
}
