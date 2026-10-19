import Foundation
import OSCDCore
import SplatXTrainC

enum PreviewError: LocalizedError {
    case message(String)
    var errorDescription: String? { if case .message(let m) = self { return m }; return nil }
}

/// Renders the reference splat model (trained on the Mac) at a world-to-camera pose with splatx-metal.
/// splatx's contract: open, render and close a session on one thread (PreviewWorker's thread).
final class SplatRenderer {
    private let session: OpaquePointer

    /// `kernelSourceDir`: splatx's Sources/SplatXKernelSrc, bundled with the app. With no metallib
    /// given, splatx compiles these kernels with the device's Metal compiler when the session opens.
    init(plyPath: String, kernelSourceDir: String) throws {
        var err = [CChar](repeating: 0, count: 1024)
        let opened: OpaquePointer? = plyPath.withCString { ply in
            kernelSourceDir.withCString { src in
                var args = SxRenderSessionArgs()
                args.ply_path = ply
                args.kernel_src_dir = src
                return sx_render_session_open(&args, &err, Int32(err.count))
            }
        }
        guard let opened else { throw PreviewError.message("Could not open the reference model: \(String(cString: err))") }
        session = opened
    }

    deinit { sx_render_session_close(session) }

    /// Interleaved RGBA (alpha 255) at `pose` — for display.
    func renderRGBA(pose: Pose, intrinsics K: Intrinsics) throws -> [UInt8] {
        let R = pose.R, t = pose.t
        let viewmat: [Float] = [R[0][0], R[1][0], R[2][0], t.x,
                                R[0][1], R[1][1], R[2][1], t.y,
                                R[0][2], R[1][2], R[2][2], t.z,
                                0, 0, 0, 1]
        var rgba = [UInt8](repeating: 0, count: K.width * K.height * 4)
        var err = [CChar](repeating: 0, count: 512)
        let rc = sx_render_session_frame(session, viewmat, K.f, K.f, K.cx, K.cy, Int32(K.width), Int32(K.height),
                                         &rgba, &err, Int32(err.count))
        guard rc == 0 else { throw PreviewError.message("Render failed (\(rc)): \(String(cString: err))") }
        for i in 0..<(K.width * K.height) { rgba[i * 4 + 3] = 255 }
        return rgba
    }

    /// `pose` is world-to-camera in OpenCV axes, which is splatx's viewmat convention (row-major).
    func render(pose: Pose, intrinsics K: Intrinsics) throws -> FloatImage {
        let R = pose.R, t = pose.t
        let viewmat: [Float] = [R[0][0], R[1][0], R[2][0], t.x,
                                R[0][1], R[1][1], R[2][1], t.y,
                                R[0][2], R[1][2], R[2][2], t.z,
                                0, 0, 0, 1]
        let n = K.width * K.height
        var rgba = [UInt8](repeating: 0, count: n * 4)
        var err = [CChar](repeating: 0, count: 512)
        let rc = sx_render_session_frame(session, viewmat, K.f, K.f, K.cx, K.cy, Int32(K.width), Int32(K.height),
                                         &rgba, &err, Int32(err.count))
        guard rc == 0 else { throw PreviewError.message("Reference render failed (\(rc)): \(String(cString: err))") }
        var planar = [Float](repeating: 0, count: n * 3)
        for i in 0..<n { for c in 0..<3 { planar[c * n + i] = Float(rgba[i * 4 + c]) / 255 } }
        return FloatImage(width: K.width, height: K.height, channels: 3, data: planar)
    }
}
