import simd

public struct MiniBAConfig: Sendable {
    public var iters: Int
    public var huberDelta: Double = 1
    public var madScale: Double            // 0 disables MAD outlier rejection
    public var lm: Double = 1e-5
    public var ep: Double = 1e-2
    public var k: Double = 2

    public init(iters: Int, madScale: Double) {
        self.iters = iters
        self.madScale = madScale
    }

    /// O-SCD PoseInitializer.miniBA: MiniBA defaults with iters_miniba_incr = 20.
    public static let poseRefine = MiniBAConfig(iters: 20, madScale: 4)
    /// O-SCD RANSACEstimator P4P hypothesis solver.
    public static let p4p = MiniBAConfig(iters: 20, madScale: 0)
}

public struct MiniBAResult: Sendable {
    public var r6: [Double]
    public var t: SIMD3<Double>
    /// Residuals at the last *tentative* parameters, as O-SCD returns them (code quirk).
    public var residuals: [Double]
    public var initialResiduals: [Double]
    /// Per residual component (2 per point): 1 = inlier, 0 = masked.
    public var mask: [Double]
    public var inlierComponents: Int { Int(mask.reduce(0, +)) }
}

/// Pose-only Levenberg–Marquardt over O-SCD's 9 raw parameters (6D rotation + translation),
/// ported from poses/mini_ba.py with optimize_focal = optimize_3Dpts = False. Points whose uv
/// has a negative coordinate are padding and never contribute.
public enum MiniBA {
    public static func solve(r6 r6Init: [Double], t tInit: SIMD3<Double>, f: Double, centre: SIMD2<Double>,
                             xyz: [SIMD3<Double>], uv: [SIMD2<Double>], config: MiniBAConfig) -> MiniBAResult {
        let n = xyz.count, m = 2 * n
        let valid = uv.map { $0.x >= 0 && $0.y >= 0 }
        var p = r6Init + [tInit.x, tInit.y, tInit.z]
        var pTmp = p
        var lm = config.lm
        var initial: [Double] = []
        for it in 0..<config.iters {
            let rIn = residuals(p, f, centre, xyz, uv)
            if it == 0 { initial = rIn }
            let J = jacobian(p, f, centre, xyz, uv)
            let mask = componentMask(rIn, valid: valid, madScale: config.madScale)
            var w = [Double](repeating: 0, count: m), r = [Double](repeating: 0, count: m)
            for i in 0..<m where mask[i] != 0 {
                w[i] = huberWeight(rIn[i], config.huberDelta)
                r[i] = rIn[i] * w[i]
            }
            var JtJ = [Double](repeating: 0, count: 81), Jtr = [Double](repeating: 0, count: 9)
            for i in 0..<m where w[i] != 0 {
                for a in 0..<9 {
                    let ja = J[i * 9 + a] * w[i]
                    Jtr[a] += ja * r[i]
                    for b in 0..<9 { JtJ[a * 9 + b] += ja * J[i * 9 + b] * w[i] }
                }
            }
            for a in 0..<9 { JtJ[a * 9 + a] = max(JtJ[a * 9 + a] * (1 + lm), config.ep) }
            let d = LinearSolve.solve(JtJ, Jtr, n: 9).map { $0.isFinite ? $0 : 0 }
            for k in 0..<9 { pTmp[k] = p[k] - d[k] }
            let rNew = residuals(pTmp, f, centre, xyz, uv)
            var newSq = 0.0, oldSq = 0.0
            for i in 0..<m where mask[i] != 0 {
                let v = rNew[i] * huberWeight(rNew[i], config.huberDelta)
                newSq += v * v
                oldSq += r[i] * r[i]
            }
            let success = newSq < oldSq
            if success { p = pTmp }
            lm *= success ? 1 / config.k : config.k
            let r6n = SixD.fromMatrix(SixD.toMatrix(p[0..<6]))   // re-orthonormalise every iteration
            for k in 0..<6 { p[k] = r6n[k] }
        }
        let rFinal = residuals(pTmp, f, centre, xyz, uv)
        return MiniBAResult(r6: Array(p[0..<6]), t: SIMD3(p[6], p[7], p[8]), residuals: rFinal,
                            initialResiduals: initial,
                            mask: componentMask(rFinal, valid: valid, madScale: config.madScale))
    }

    static func residuals(_ p: [Double], _ f: Double, _ c: SIMD2<Double>,
                          _ xyz: [SIMD3<Double>], _ uv: [SIMD2<Double>]) -> [Double] {
        let R = SixD.toMatrix(p[0..<6]), t = SIMD3(p[6], p[7], p[8])
        var r = [Double](repeating: 0, count: 2 * xyz.count)
        for i in 0..<xyz.count {
            let q = R * xyz[i] + t
            r[2 * i] = f * q.x / q.z + c.x - uv[i].x
            r[2 * i + 1] = f * q.y / q.z + c.y - uv[i].y
        }
        return r
    }

    /// Central differences in Double (O-SCD uses forward-mode autodiff in fp32).
    static func jacobian(_ p: [Double], _ f: Double, _ c: SIMD2<Double>,
                         _ xyz: [SIMD3<Double>], _ uv: [SIMD2<Double>]) -> [Double] {
        let m = 2 * xyz.count
        var J = [Double](repeating: 0, count: m * 9)
        var q = p
        for k in 0..<9 {
            let h = 1e-6 * max(1, abs(p[k]))
            q[k] = p[k] + h
            let rp = residuals(q, f, c, xyz, uv)
            q[k] = p[k] - h
            let rm = residuals(q, f, c, xyz, uv)
            q[k] = p[k]
            for i in 0..<m { J[i * 9 + k] = (rp[i] - rm[i]) / (2 * h) }
        }
        return J
    }

    @inline(__always) static func huberWeight(_ r: Double, _ delta: Double) -> Double {
        abs(r) <= delta ? 1 : delta / abs(r).squareRoot()
    }

    /// O-SCD get_mask: optional MAD rejection on per-point error, expanded to both components.
    static func componentMask(_ r: [Double], valid: [Bool], madScale: Double) -> [Double] {
        let n = valid.count
        var mask = [Double](repeating: 0, count: 2 * n)
        guard n > 0 else { return mask }
        if madScale > 0 {
            var err = [Double](repeating: 0, count: n)
            var nValid = 0
            for i in 0..<n where valid[i] {
                err[i] = (r[2 * i] * r[2 * i] + r[2 * i + 1] * r[2 * i + 1]).squareRoot()
                nValid += 1
            }
            let q = 1 - 0.5 * Double(nValid) / Double(n)
            let med = quantile(err, q)
            let mad = quantile(err.map { abs($0 - med) }, q)
            let cutoff = max(med + madScale * mad, 5)
            for i in 0..<n where valid[i] && err[i] < cutoff { mask[2 * i] = 1; mask[2 * i + 1] = 1 }
        } else {
            for i in 0..<n where valid[i] { mask[2 * i] = 1; mask[2 * i + 1] = 1 }
        }
        return mask
    }

    /// torch.quantile with linear interpolation.
    static func quantile(_ v: [Double], _ q: Double) -> Double {
        let s = v.sorted()
        let pos = q * Double(s.count - 1)
        let lo = Int(pos.rounded(.down)), hi = min(lo + 1, s.count - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - Double(lo))
    }
}
