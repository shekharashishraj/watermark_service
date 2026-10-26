import Foundation
import simd

public enum SAM2Pool {
    /// Per-channel mean of a SAM2 embedding [channels, grid, grid] over the cells the pixels fall in
    /// (pixel (x, y) at width×height maps to cell (y·grid/height)·grid + x·grid/width, as `CuesA.feature`).
    public static func mean(embedding: [Float], pixels: [Int], width: Int, height: Int,
                            grid: Int = 64, channels: Int = 256) -> [Float] {
        var out = [Float](repeating: 0, count: channels)
        guard !pixels.isEmpty else { return out }
        let cells = grid * grid
        let cellIndex = pixels.map { ($0 / width * grid / height) * grid + ($0 % width) * grid / width }
        for c in 0..<channels {
            var s: Float = 0
            for cell in cellIndex { s += embedding[c * cells + cell] }
            out[c] = s / Float(pixels.count)
        }
        return out
    }
}

public enum LabHistogram {
    public static let bins = 256     // L 4 × a 8 × b 8; L in [0,100], a,b in [−100,100]

    /// Normalised CIELAB histogram of the pixels (sums to 1; all zeros for no pixels). `gain` scales
    /// the RGB first (clamped to 1), so a render can be aligned to the live exposure.
    public static func compute(image: FloatImage, pixels: [Int], gain: SIMD3<Float> = SIMD3(repeating: 1)) -> [Float] {
        var h = [Float](repeating: 0, count: bins)
        guard !pixels.isEmpty else { return h }
        let n = image.width * image.height
        func bin(_ v: Float, _ lo: Float, _ hi: Float, _ count: Int) -> Int {
            min(count - 1, max(0, Int((v - lo) / (hi - lo) * Float(count))))
        }
        for i in pixels {
            let lab = CuesB.srgbToLab(min(1, image.data[i] * gain.x), min(1, image.data[n + i] * gain.y),
                                      min(1, image.data[2 * n + i] * gain.z))
            h[(bin(lab.x, 0, 100, 4) * 8 + bin(lab.y, -100, 100, 8)) * 8 + bin(lab.z, -100, 100, 8)] += 1
        }
        let inv = 1 / Float(pixels.count)
        return h.map { $0 * inv }
    }

    /// Σ min(aᵢ, bᵢ): 1 for identical histograms, 0 for disjoint ones.
    public static func intersection(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + min($1.0, $1.1) }
    }
}
