import ARKit
import AVFoundation
import Combine
import OSCDCore
import RealityKit
import UIKit

/// Records reference-scan sessions: keyframes (RGB at 480×360, the full camera image as JPEG for training,
/// LiDAR depth + confidence, ARKit pose), each the
/// sharpest frame of its window (`KeyframeSelector`). At Stop the scanned area is worked out from the scan's
/// LiDAR (`SceneRegion`), so a scan can be a tabletop or a room.
/// ARSession delegate callbacks arrive on the main thread; files are written on a serial queue.
final class CaptureController: NSObject, ObservableObject, ARSessionDelegate {
    static let working = (width: 480, height: 360)

    var arView: ARView { SharedAR.view }
    @Published var status = "Press Record, then move slowly over the scene"
    @Published var keyframeCount = 0
    @Published var recording = false
    /// Working out the scanned area after Stop.
    @Published var saving = false

    /// A frame held while its window is open, copied out of the ARFrame so ARKit can reuse its buffers.
    /// The camera planes are copied as they are (a memcpy); colour conversion happens on the write queue.
    struct Candidate {
        var timestamp: Double
        var arkitCameraToWorld: [Float]
        var image: Planes
        var depth: [Float]
        var confidence: [UInt8]
    }

    /// The camera image's 420f planes, tightly packed: luma (width × height), interleaved CbCr (half size).
    struct Planes {
        var luma: [UInt8]
        var chroma: [UInt8]
        var width: Int
        var height: Int

        init(_ buffer: CVPixelBuffer) {
            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
            width = CVPixelBufferGetWidthOfPlane(buffer, 0)
            height = CVPixelBufferGetHeightOfPlane(buffer, 0)
            luma = Self.packed(buffer, plane: 0, rowBytes: width)
            chroma = Self.packed(buffer, plane: 1, rowBytes: CVPixelBufferGetWidthOfPlane(buffer, 1) * 2)
        }

        private static func packed(_ buffer: CVPixelBuffer, plane: Int, rowBytes: Int) -> [UInt8] {
            let h = CVPixelBufferGetHeightOfPlane(buffer, plane), stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, plane)
            let src = CVPixelBufferGetBaseAddressOfPlane(buffer, plane)!
            var out = [UInt8](repeating: 0, count: rowBytes * h)
            out.withUnsafeMutableBytes { dst in
                for y in 0..<h { (dst.baseAddress! + y * rowBytes).copyMemory(from: src + y * stride, byteCount: rowBytes) }
            }
            return out
        }

        func rgb(factor: Int) -> [UInt8] {
            CameraImage.rgbArea(y: luma, yStride: width, cbcr: chroma, cbcrStride: width, width: width, height: height, factor: factor)
        }
    }

    /// What one recording cost and chose, logged at Stop (README "Capture checks").
    struct RecordingStats {
        var frameGapsMs: [Double] = []         // between consecutive camera frames
        var handleMs: [Double] = []            // didUpdate, per frame
        var scoreMs: [Double] = []             // sharpness, per eligible frame
        var copyMs: [Double] = []              // image + depth copy, per new sharpest frame
        var gains: [Double] = []               // keyframe sharpness ÷ its window's first frame's
        var candidates: [Int] = []
        var writeMs: [Double] = []             // JPEG + arrays per keyframe, on the write queue
        var dropped = 0                        // keyframes not saved because the write queue was full
        var lastTimestamp: Double?

        static func quantile(_ v: [Double], _ q: Double) -> Double {
            guard !v.isEmpty else { return .nan }
            let s = v.sorted()
            return s[min(Int((q * Double(s.count - 1)).rounded()), s.count - 1)]
        }

        var summary: String {
            let q = Self.quantile, c = candidates.map(Double.init)
            let blurryFirst = gains.filter { $0 > 1 / 0.7 }.count
            return String(format: "capture stats: %d frames, %d keyframes | frame gap ms median %.1f, max %.1f"
                          + " | didUpdate ms median %.2f, p95 %.2f, max %.2f | sharpness ms median %.2f | copy ms median %.2f (%d copies)"
                          + " | keyframe/first sharpness median %.2f, first < 70%% in %d of %d windows | candidates median %.0f, mean %.1f"
                          + " | write ms median %.0f, max %.0f | dropped %d",
                          handleMs.count, gains.count, q(frameGapsMs, 0.5), frameGapsMs.max() ?? .nan,
                          q(handleMs, 0.5), q(handleMs, 0.95), handleMs.max() ?? .nan, q(scoreMs, 0.5),
                          q(copyMs, 0.5), copyMs.count, q(gains, 0.5), blurryFirst, gains.count,
                          q(c, 0.5), c.isEmpty ? .nan : c.reduce(0, +) / Double(c.count),
                          q(writeMs, 0.5), writeMs.max() ?? .nan, dropped)
        }
    }

    private var selector = KeyframeSelector<Candidate>()
    private var stats = RecordingStats()
    private var inFlight = 0
    private var store: SessionStore?
    /// Taken from the recording's first frame; the folder is created with the first keyframe.
    private var pendingManifest: SessionManifest?
    private let writeQueue = DispatchQueue(label: "oscd.capture.write")

    func start() {
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = [.sceneDepth]
        config.isAutoFocusEnabled = false
        // 1920×1440 maps exactly onto the 480×360 working size (4×4 blocks).
        if let format = ARWorldTrackingConfiguration.supportedVideoFormats.first(where: {
            $0.imageResolution == CGSize(width: 1920, height: 1440) }) {
            config.videoFormat = format
        }
        arView.session.delegate = self
        // Which larger formats this phone offers (training at more than 1920×1440 needs one that keeps LiDAR depth).
        AppLog.write("capture: video formats " + ARWorldTrackingConfiguration.supportedVideoFormats.map {
            "\(Int($0.imageResolution.width))×\(Int($0.imageResolution.height))@\($0.framesPerSecond)"
        }.joined(separator: ", ") + "; 4K: " + (ARWorldTrackingConfiguration.recommendedVideoFormatFor4KResolution.map {
            "\(Int($0.imageResolution.width))×\(Int($0.imageResolution.height))@\($0.framesPerSecond)"
        } ?? "none"))
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

    func toggleRecording() {
        if recording {
            recording = false
            if let chosen = selector.finish() { save(chosen) }
            pendingManifest = nil
            AppLog.write(stats.summary)
            guard let store else {
                status = "No frames were saved. Move more slowly, with the scene in view."
                return
            }
            self.store = nil
            let n = keyframeCount
            saving = true
            status = "Working out the scanned area…"
            writeQueue.async {
                var message = "Saved \(n) frames. The scan is in the Scans tab."
                do {
                    if let box = try SceneRegion.estimate(store, keyframeStride: max(n / 60, 1)) {
                        try store.setROI(box)
                        let size = box.max - box.min
                        AppLog.write(String(format: "capture: scanned area %.2f × %.2f × %.2f m from %d frames", size.x, size.y, size.z, n))
                    }
                } catch {
                    message = "Saved \(n) frames, but working out the scanned area failed: \(error.localizedDescription)"
                }
                DispatchQueue.main.async {
                    self.saving = false
                    self.status = message
                }
            }
            return
        }
        guard !saving else { return }
        lockCamera()
        selector = KeyframeSelector()
        stats = RecordingStats()
        inFlight = 0
        keyframeCount = 0
        recording = true
        tooFastSince = nil
        calmSince = nil
        status = Self.recordingStatus
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
        AppLog.write("capture: first frame, mean luma \(sum / max(n, 1)) of 255")
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        logFirstFrame(frame)
        guard recording, let sceneDepth = frame.sceneDepth else { return }
        let start = CACurrentMediaTime()
        defer { stats.handleMs.append((CACurrentMediaTime() - start) * 1000) }
        if let last = stats.lastTimestamp { stats.frameGapsMs.append((frame.timestamp - last) * 1000) }
        stats.lastTimestamp = frame.timestamp
        let res = frame.camera.imageResolution
        guard CameraImage.areaFactor(captureWidth: Int(res.width), captureHeight: Int(res.height),
                                     workingWidth: Self.working.width, workingHeight: Self.working.height) != nil else {
            status = "Camera format \(Int(res.width))×\(Int(res.height)) is not supported; it must be 4:3 at a multiple of 480×360."
            return
        }
        if store == nil, pendingManifest == nil { pendingManifest = makeManifest(frame: frame, depth: sceneDepth) }
        var normal = false
        if case .normal = frame.camera.trackingState { normal = true }
        let pose = Pose.fromARKitCameraToWorld(frame.camera.transform)
        let chosen = selector.consider(pose: pose, time: frame.timestamp, trackingNormal: normal,
                                       sharpness: {
            let t = CACurrentMediaTime()
            defer { stats.scoreMs.append((CACurrentMediaTime() - t) * 1000) }
            return Self.sharpness(frame.capturedImage)
        }, candidate: {
            let t = CACurrentMediaTime()
            defer { stats.copyMs.append((CACurrentMediaTime() - t) * 1000) }
            let (depth, confidence) = Self.copyDepth(sceneDepth)
            return Candidate(timestamp: frame.timestamp, arkitCameraToWorld: frame.camera.transform.columnMajor,
                             image: Planes(frame.capturedImage), depth: depth, confidence: confidence)
        })
        if let chosen { save(chosen) }
        // Nothing is saved while the phone turns faster than 30°/s (blurred frames). Say so only after 1.5 s of
        // it, and keep saying it until 1 s of calm, so the message does not flicker with every hand movement.
        if selector.movingTooFast {
            tooFastSince = tooFastSince ?? frame.timestamp
            calmSince = nil
            if frame.timestamp - tooFastSince! > 1.5, status != Self.slowDownStatus { status = Self.slowDownStatus }
        } else {
            tooFastSince = nil
            calmSince = calmSince ?? frame.timestamp
            if status == Self.slowDownStatus, frame.timestamp - calmSince! > 1 { status = Self.recordingStatus }
        }
    }

    private var tooFastSince: Double?
    private var calmSince: Double?
    static let recordingStatus = "Recording. Move slowly and cover the scene from several sides."
    static let slowDownStatus = "Turning too fast: nothing is saved until you slow down."

    private func save(_ chosen: KeyframeSelector<Candidate>.Chosen) {
        if store == nil {
            guard let manifest = pendingManifest else { return }
            do {
                store = try makeStore(manifest)
            } catch {
                status = "Could not create the session folder: \(error.localizedDescription)"
                recording = false
                return
            }
        }
        guard let store else { return }
        // At video rate the JPEG encode can fall behind the camera; hold at most 6 keyframes in flight and
        // drop beyond that rather than let memory grow (each candidate is ~5 MB).
        if inFlight >= 6 { stats.dropped += 1; return }
        inFlight += 1
        let c = chosen.candidate
        let factor = c.image.width / Self.working.width
        stats.gains.append(chosen.firstSharpness > 0 ? chosen.sharpness / chosen.firstSharpness : 1)
        stats.candidates.append(chosen.candidates)
        let record = FrameRecord(id: keyframeCount, timestamp: c.timestamp, arkitCameraToWorld: c.arkitCameraToWorld,
                                 stats: ["sharpness": chosen.sharpness, "firstSharpness": chosen.firstSharpness,
                                         "candidates": Double(chosen.candidates)])
        keyframeCount += 1
        writeQueue.async {
            let t0 = CACurrentMediaTime()
            do {
                // Full-size image first: the manifest entry (written by append) only appears once both exist.
                try ColmapExport.writeJPEG(rgb: c.image.rgb(factor: 1), width: c.image.width, height: c.image.height,
                                           to: store.fullImageURL(.keyframes, id: record.id))
                try store.append(.keyframes, record: record, rgb: c.image.rgb(factor: factor), depth: c.depth, confidence: c.confidence)
            } catch {
                DispatchQueue.main.async { self.status = "Saving frame \(record.id) failed: \(error.localizedDescription)" }
            }
            let ms = (CACurrentMediaTime() - t0) * 1000
            DispatchQueue.main.async { self.inFlight -= 1; self.stats.writeMs.append(ms) }
        }
    }

    /// PocketGS's gradient energy on the full-resolution luma plane, every 4th pixel.
    static func sharpness(_ buffer: CVPixelBuffer) -> Double {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let y = CVPixelBufferGetBaseAddressOfPlane(buffer, 0)!.assumingMemoryBound(to: UInt8.self)
        return Sharpness.score(luma: y, width: CVPixelBufferGetWidthOfPlane(buffer, 0), height: CVPixelBufferGetHeightOfPlane(buffer, 0),
                               rowBytes: CVPixelBufferGetBytesPerRowOfPlane(buffer, 0))
    }

    private func makeManifest(frame: ARFrame, depth: ARDepthData) -> SessionManifest {
        let K = frame.camera.intrinsics
        let res = frame.camera.imageResolution
        let intrinsics = Intrinsics.fromARKit(fx: K[0][0], fy: K[1][1], cx: K[2][0], cy: K[2][1],
                                              captureWidth: Int(res.width), captureHeight: Int(res.height),
                                              workingWidth: Self.working.width, workingHeight: Self.working.height)
        return SessionManifest(deviceModel: Self.deviceModel(), captureWidth: Int(res.width), captureHeight: Int(res.height),
                               depthWidth: CVPixelBufferGetWidth(depth.depthMap), depthHeight: CVPixelBufferGetHeight(depth.depthMap),
                               intrinsics: intrinsics)
    }

    private func makeStore(_ manifest: SessionManifest) throws -> SessionStore {
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
