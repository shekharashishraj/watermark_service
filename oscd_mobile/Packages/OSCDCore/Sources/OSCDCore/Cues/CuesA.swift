import Foundation

public struct CueMapsA: Sendable {
    public var combined: [Float]      // C_A in [0, 2]; 0 at invalid pixels
    public var pixel: [Float]         // normalised pixel cue
    public var feature: [Float]       // feature cue after O-SCD's re-normalisation, in [0, 1]
    public var rawMin: Float
    public var rawMax: Float
}

/// O-SCD oscd.py `generate_candidate_map`.
public enum CuesA {
    /// Pixel half: 0.8·mean_c|render − live| + 0.2·(1 − SSIM), min-max normalised.
    public static func pixel(live: FloatImage, render: FloatImage, valid: [Bool]? = nil)
        -> (normalized: [Float], raw: [Float], vmin: Float, vmax: Float) {
        let n = live.width * live.height
        let ssim = SSIM.meanMap(render, live)
        var raw = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var l1: Float = 0
            for c in 0..<live.channels { l1 += abs(render.data[c * n + i] - live.data[c * n + i]) }
            raw[i] = 0.8 * (l1 / Float(live.channels)) + 0.2 * (1 - ssim[i])
        }
        var vmin = Float.infinity, vmax = -Float.infinity
        for i in 0..<n where valid?[i] ?? true {
            vmin = min(vmin, raw[i])
            vmax = max(vmax, raw[i])
        }
        if vmin > vmax { return ([Float](repeating: 0, count: n), raw, 0, 0) }
        return (raw.map { ($0 - vmin) / (vmax - vmin + 1e-8) }, raw, vmin, vmax)
    }

    /// Feature half from SAM2 embeddings [channels, grid, grid] of the render and the live image:
    /// channel-mean |diff| → min-max → bilinear upsample → re-normalise with the pixel cue's raw
    /// min/max → clamp to [0, 1] (O-SCD code quirk, reproduced).
    public static func feature(embRender: [Float], embLive: [Float], channels: Int = 256, grid: Int = 64,
                               width: Int, height: Int, vmin: Float, vmax: Float, valid: [Bool]? = nil) -> [Float] {
        let cells = grid * grid
        var diff = [Float](repeating: 0, count: cells)
        for c in 0..<channels {
            for i in 0..<cells { diff[i] += abs(embRender[c * cells + i] - embLive[c * cells + i]) }
        }
        for i in 0..<cells { diff[i] /= Float(channels) }
        var cellValid = [Bool](repeating: valid == nil, count: cells)
        if let valid {
            for y in 0..<height {
                for x in 0..<width where valid[y * width + x] { cellValid[(y * grid / height) * grid + x * grid / width] = true }
            }
        }
        var dmin = Float.infinity, dmax = -Float.infinity
        for i in 0..<cells where cellValid[i] {
            dmin = min(dmin, diff[i])
            dmax = max(dmax, diff[i])
        }
        if dmin > dmax { return [Float](repeating: 0, count: width * height) }
        let norm = FloatImage(width: grid, height: grid, channels: 1, data: diff.map { ($0 - dmin) / (dmax - dmin + 1e-8) })
        return norm.resizedBilinear(width: width, height: height).data.map { min(1, max(0, ($0 - vmin) / (vmax - vmin + 1e-8))) }
    }

    public static func compute(live: FloatImage, render: FloatImage, embRender: [Float], embLive: [Float],
                               valid: [Bool]? = nil) -> CueMapsA {
        let p = pixel(live: live, render: render, valid: valid)
        let f = feature(embRender: embRender, embLive: embLive, width: live.width, height: live.height,
                        vmin: p.vmin, vmax: p.vmax, valid: valid)
        var combined = zip(p.normalized, f).map(+)
        if let valid { for i in combined.indices where !valid[i] { combined[i] = 0 } }
        return CueMapsA(combined: combined, pixel: p.normalized, feature: f, rawMin: p.vmin, rawMax: p.vmax)
    }
}
