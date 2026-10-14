import Foundation
import OSCDCore
import SplatXTrainC
import simd

public enum FusionError: LocalizedError {
    case failed(String)
    public var errorDescription: String? { if case .failed(let m) = self { return m }; return nil }
}

/// O-SCD's self-supervised fusion loss (see splatx/OscdFusion.h).
public enum SSF {
    public static func lossAndGrad(render: [Float], cue: [Float], valid: [Bool], width: Int, height: Int) -> (loss: Float, grad: [Float]) {
        var grad = [Float](repeating: 0, count: render.count)
        let v = valid.map { UInt8($0 ? 1 : 0) }
        let loss = oscd_ssf_loss_and_grad(render, cue, v, Int32(width), Int32(height), &grad)
        return (loss, grad)
    }
}

/// One torch.optim.Adam step (eps 1e-15), as O-SCD's optimiser.
public enum TorchAdam {
    public static func step(param: inout [Float], grad: [Float], expAvg: inout [Float], expAvgSq: inout [Float], step: Int, lr: Float) {
        oscd_torch_adam(&param, grad, &expAvg, &expAvgSq, Int64(param.count), Int32(step), lr)
    }
}

/// O-SCD change fusion (R_change + SSF) on splatx. Not thread-safe: create, use and release it on
/// one thread.
public final class ChangeFusion {
    private let handle: OpaquePointer

    /// - plyPath: the reference model; referenceCentres: world camera centres of the reference scan.
    /// - kernelSourceDir: splatx's kernel sources bundled with an app (nil = the build checkout).
    public init(plyPath: String, kernelSourceDir: String? = nil, referenceCentres: [SIMD3<Float>], seed: UInt32 = 0) throws {
        var err = [CChar](repeating: 0, count: 1024)
        let centres = referenceCentres.flatMap { [$0.x, $0.y, $0.z] }
        let opened: OpaquePointer? = plyPath.withCString { ply in
            if let kernelSourceDir {
                return kernelSourceDir.withCString { oscd_fusion_open(ply, $0, centres, Int32(referenceCentres.count), seed, &err, Int32(err.count)) }
            }
            return oscd_fusion_open(ply, nil, centres, Int32(referenceCentres.count), seed, &err, Int32(err.count))
        }
        guard let opened else { throw FusionError.failed("Could not open the change model: \(String(cString: err))") }
        handle = opened
    }

    deinit { oscd_fusion_close(handle) }

    /// World-to-camera pose → splatx viewmat (row-major [R | t]).
    static func viewmat(_ p: Pose) -> [Float] {
        let R = p.R, t = p.t
        return [R[0][0], R[1][0], R[2][0], t.x, R[0][1], R[1][1], R[2][1], t.y, R[0][2], R[1][2], R[2][2], t.z, 0, 0, 0, 1]
    }

    private func check(_ rc: Int32, _ err: [CChar], _ what: String) throws {
        guard rc == 0 else { throw FusionError.failed("\(what) failed (\(rc)): \(String(cString: err))") }
    }

    /// Adds a frame's O-SCD cue map (width·height, in [0,2]) and D3 validity at its world-to-camera pose.
    public func addFrame(pose: Pose, intrinsics K: Intrinsics, cue: [Float], valid: [Bool]) throws {
        var err = [CChar](repeating: 0, count: 512)
        let v = valid.map { UInt8($0 ? 1 : 0) }
        try check(oscd_fusion_add_frame(handle, Self.viewmat(pose), K.f, K.f, K.cx, K.cy, Int32(K.width), Int32(K.height),
                                        cue, v, &err, Int32(err.count)), err, "addFrame")
    }

    /// O-SCD's per-frame loop (16 steps, densify at step 4) for the newest frame. Returns the loss.
    @discardableResult public func stepFrame() throws -> Float {
        var err = [CChar](repeating: 0, count: 512)
        var loss: Float = 0
        try check(oscd_fusion_step_frame(handle, &loss, &err, Int32(err.count)), err, "stepFrame")
        return loss
    }

    /// O-SCD --refine: continue up to 3,000 total steps over all frames.
    public func refine() throws {
        var err = [CChar](repeating: 0, count: 512)
        try check(oscd_fusion_refine(handle, &err, Int32(err.count)), err, "refine")
    }

    /// Change values at a pose (channel mean of the clamped render); O-SCD's mask is value > 0.5.
    public func changeMap(pose: Pose, intrinsics K: Intrinsics) throws -> [Float] {
        var err = [CChar](repeating: 0, count: 512)
        var out = [Float](repeating: 0, count: K.width * K.height)
        try check(oscd_fusion_change_map(handle, Self.viewmat(pose), K.f, K.f, K.cx, K.cy, Int32(K.width), Int32(K.height),
                                         &out, &err, Int32(err.count)), err, "changeMap")
        return out
    }

    public var count: Int { Int(oscd_fusion_count(handle)) }
    public var totalSteps: Int { Int(oscd_fusion_total_steps(handle)) }

    // MARK: test hooks (O-SCD densification primitives)

    func debugDensify(grads: [Float], threshold: Float, extent: Float) throws {
        var err = [CChar](repeating: 0, count: 512)
        try check(oscd_fusion_debug_densify(handle, grads, Int64(grads.count), threshold, extent, &err, Int32(err.count)), err, "densify")
    }

    func debugPrune(minOpacity: Float) throws {
        var err = [CChar](repeating: 0, count: 512)
        try check(oscd_fusion_debug_prune(handle, minOpacity, &err, Int32(err.count)), err, "prune")
    }

    func debugResetOpacity() throws {
        var err = [CChar](repeating: 0, count: 512)
        try check(oscd_fusion_debug_reset_opacity(handle, &err, Int32(err.count)), err, "resetOpacity")
    }

    private func debugParams() -> (scales: [Float], opacities: [Float]) {
        var s = [Float](repeating: 0, count: count * 3), o = [Float](repeating: 0, count: count)
        _ = oscd_fusion_debug_params(handle, &s, &o, Int64(count))
        return (s, o)
    }

    func debugLogScales() -> [Float] { debugParams().scales }
    func debugOpacities() -> [Float] { debugParams().opacities }
}
