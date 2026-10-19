import Foundation
import OSCDCore
import SplatXTrainC

/// Expected depth of the reference model (Method B's depth cue, spec §8.1): Σ wᵢzᵢ / Σ wᵢ per pixel in
/// metres, NaN where the reference covers the pixel with alpha below `minAlpha`. Not thread-safe:
/// create, use and release it on one thread.
public final class ReferenceDepth {
    private let handle: OpaquePointer

    public init(plyPath: String, kernelSourceDir: String? = nil) throws {
        var err = [CChar](repeating: 0, count: 1024)
        let opened: OpaquePointer? = plyPath.withCString { ply in
            if let kernelSourceDir { return kernelSourceDir.withCString { oscd_depth_open(ply, $0, &err, Int32(err.count)) } }
            return oscd_depth_open(ply, nil, &err, Int32(err.count))
        }
        guard let opened else { throw FusionError.failed("Could not open the reference for depth: \(String(cString: err))") }
        handle = opened
    }

    deinit { oscd_depth_close(handle) }

    public func expectedDepth(pose: Pose, intrinsics K: Intrinsics, minAlpha: Float = 0.5) throws -> [Float] {
        var err = [CChar](repeating: 0, count: 512)
        var out = [Float](repeating: 0, count: K.width * K.height)
        let rc = oscd_depth_render(handle, ChangeFusion.viewmat(pose), K.f, K.f, K.cx, K.cy, Int32(K.width), Int32(K.height),
                                   minAlpha, &out, &err, Int32(err.count))
        guard rc == 0 else { throw FusionError.failed("expectedDepth failed (\(rc)): \(String(cString: err))") }
        return out
    }
}
