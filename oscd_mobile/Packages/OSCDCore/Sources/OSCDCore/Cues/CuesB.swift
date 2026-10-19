import Foundation
import simd

/// Method B additions (spec §8): LiDAR depth cue and exposure-aligned, chroma-weighted color cue.
public enum CuesB {
    public struct Params: Sendable {
        public var depthMinTol: Float = 0.01
        public var depthRelTol: Float = 0.02
        public var depthRamp: Float = 0.02
        /// D13: subtract the frame's median (D_live − D_ren) over judged pixels before the tolerance.
        public var alignDepthOffset = true
        public var gainAgreeDepth: Float = 0.01
        public var gainMinPixels = 500
        public var blurSigma: Float = 1
        public var lightnessWeight: Float = 0.25
        public var deltaE0: Float = 10
        public var deltaE1: Float = 25
        public var minL: Float = 15
        public var maxL: Float = 95
        public var lambdaDepth: Float = 1
        public var lambdaColor: Float = 1
        public init() {}
    }

    /// LiDAR depth (metres) to working resolution by nearest neighbour. Pixels with ARKit confidence
    /// below high (2), or a non-finite or non-positive depth, become NaN.
    public static func upsampleDepth(_ d: [Float], confidence: [UInt8], width: Int, height: Int,
                                     toWidth: Int, toHeight: Int) -> [Float] {
        var out = [Float](repeating: .nan, count: toWidth * toHeight)
        for y in 0..<toHeight {
            let sy = min(y * height / toHeight, height - 1)
            for x in 0..<toWidth {
                let i = sy * width + min(x * width / toWidth, width - 1)
                if confidence[i] >= 2, d[i].isFinite, d[i] > 0 { out[y * toWidth + x] = d[i] }
            }
        }
        return out
    }

    /// C_depth = clamp((|D_live − D_ren − o| − τ) / ramp, 0, 1) with τ = max(1 cm, 2%·D_ren) and o the
    /// frame's median D_live − D_ren over judged pixels (D13; o = 0 when alignDepthOffset is off).
    public static func depthCue(live: [Float], rendered: [Float], valid: [Bool],
                                params p: Params = Params()) -> (cue: [Float], sign: [Int8]) {
        var cue = [Float](repeating: 0, count: live.count), sign = [Int8](repeating: 0, count: live.count)
        let judged = (0..<live.count).filter { valid[$0] && live[$0].isFinite && rendered[$0].isFinite && rendered[$0] > 0 }
        var offset: Float = 0
        if p.alignDepthOffset, !judged.isEmpty {
            let d = judged.map { live[$0] - rendered[$0] }.sorted()
            offset = d.count % 2 == 1 ? d[d.count / 2] : (d[d.count / 2 - 1] + d[d.count / 2]) / 2
        }
        for i in judged {
            let tau = max(p.depthMinTol, p.depthRelTol * rendered[i])
            let d = live[i] - rendered[i] - offset
            cue[i] = min(1, max(0, (abs(d) - tau) / p.depthRamp))
            if cue[i] > 0 { sign[i] = d < 0 ? 1 : -1 }
        }
        return (cue, sign)
    }

    public static func colorCue(live: FloatImage, render: FloatImage, liveDepth: [Float], renderDepth: [Float],
                                valid: [Bool], params p: Params = Params()) -> [Float] {
        let n = live.width * live.height
        // 1. Per-channel gain aligning the render to the live frame, from pixels where depth agrees.
        var gain: [Float] = [1, 1, 1]
        var ratios: [[Float]] = [[], [], []]
        for i in 0..<n where valid[i] && liveDepth[i].isFinite && renderDepth[i].isFinite
                             && abs(liveDepth[i] - renderDepth[i]) < p.gainAgreeDepth {
            for c in 0..<3 where render.data[c * n + i] > 1e-3 { ratios[c].append(live.data[c * n + i] / render.data[c * n + i]) }
        }
        for c in 0..<3 where ratios[c].count >= p.gainMinPixels {
            let s = ratios[c].sorted()
            gain[c] = s[s.count / 2]
        }
        var adjusted = render
        for c in 0..<3 { for i in 0..<n { adjusted.data[c * n + i] = min(1, render.data[c * n + i] * gain[c]) } }
        // 2. Light blur. 3. CIELAB difference with lightness down-weighted. 4. Absolute ramp. 5. L* masks.
        let a = gaussianBlur(live, sigma: p.blurSigma), b = gaussianBlur(adjusted, sigma: p.blurSigma)
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n where valid[i] {
            let la = srgbToLab(a.data[i], a.data[n + i], a.data[2 * n + i])
            let lb = srgbToLab(b.data[i], b.data[n + i], b.data[2 * n + i])
            if min(la.x, lb.x) < p.minL || max(la.x, lb.x) > p.maxL { continue }
            let d = SIMD3(p.lightnessWeight * (la.x - lb.x), la.y - lb.y, la.z - lb.z)
            out[i] = min(1, max(0, (simd_length(d) - p.deltaE0) / (p.deltaE1 - p.deltaE0)))
        }
        return out
    }

    /// C_B = clamp(C_A + λd·C_depth + λc·C_color, 0, 2); 0 outside the validity mask.
    public static func combine(cA: [Float], depth: [Float], color: [Float], valid: [Bool],
                               params p: Params = Params()) -> [Float] {
        (0..<cA.count).map { i in valid[i] ? min(2, max(0, cA[i] + p.lambdaDepth * depth[i] + p.lambdaColor * color[i])) : 0 }
    }

    /// sRGB (D65) in [0, 1] → CIELAB.
    public static func srgbToLab(_ r: Float, _ g: Float, _ b: Float) -> SIMD3<Float> {
        func lin(_ c: Float) -> Float { c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4) }
        let R = lin(r), G = lin(g), B = lin(b)
        let X = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / 0.95047
        let Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B
        let Z = (0.0193339 * R + 0.1191920 * G + 0.9503041 * B) / 1.08883
        func f(_ t: Float) -> Float { t > 0.008856452 ? cbrt(t) : t / 0.12841855 + 4.0 / 29.0 }
        let fx = f(X), fy = f(Y), fz = f(Z)
        return SIMD3(116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))
    }

    /// Separable Gaussian blur, radius 3σ, clamp-to-edge.
    static func gaussianBlur(_ img: FloatImage, sigma: Float) -> FloatImage {
        let r = Int((3 * sigma).rounded(.up))
        var k = (-r...r).map { exp(-Float($0 * $0) / (2 * sigma * sigma)) }
        let s = k.reduce(0, +)
        k = k.map { $0 / s }
        let w = img.width, h = img.height
        var tmp = img, out = img
        for c in 0..<img.channels {
            for y in 0..<h { for x in 0..<w {
                var v: Float = 0
                for i in -r...r { v += k[i + r] * img[c, y, min(max(x + i, 0), w - 1)] }
                tmp[c, y, x] = v
            } }
            for y in 0..<h { for x in 0..<w {
                var v: Float = 0
                for i in -r...r { v += k[i + r] * tmp[c, min(max(y + i, 0), h - 1), x] }
                out[c, y, x] = v
            } }
        }
        return out
    }
}
