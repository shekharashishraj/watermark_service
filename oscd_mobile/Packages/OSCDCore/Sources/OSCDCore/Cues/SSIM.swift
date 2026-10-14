import Foundation

/// O-SCD utils/loss_utils.py `ssim(..., window_size=11, map=True)`.
public enum SSIM {
    /// Normalised 1-D Gaussian, σ = 1.5, 11 taps (the 2-D window is its outer product).
    public static let window: [Float] = {
        let g = (0..<11).map { exp(-Double(($0 - 5) * ($0 - 5)) / (2 * 1.5 * 1.5)) }
        let s = g.reduce(0, +)
        return g.map { Float($0 / s) }
    }()

    /// Zero-padded separable Gaussian filter of one plane (F.conv2d with padding = 5).
    static func blur(_ x: [Float], _ w: Int, _ h: Int) -> [Float] {
        let k = window, r = 5
        var tmp = [Float](repeating: 0, count: w * h), out = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            for x0 in 0..<w {
                var s: Float = 0
                for i in -r...r { let xx = x0 + i; if xx >= 0 && xx < w { s += k[i + r] * x[y * w + xx] } }
                tmp[y * w + x0] = s
            }
        }
        for y0 in 0..<h {
            for x in 0..<w {
                var s: Float = 0
                for i in -r...r { let yy = y0 + i; if yy >= 0 && yy < h { s += k[i + r] * tmp[yy * w + x] } }
                out[y0 * w + x] = s
            }
        }
        return out
    }

    /// Per-pixel SSIM averaged over channels (O-SCD `ssim(img1, img2, map=True).mean(dim=0)`).
    public static func meanMap(_ a: FloatImage, _ b: FloatImage) -> [Float] {
        let w = a.width, h = a.height, n = w * h
        let C1: Float = 0.01 * 0.01, C2: Float = 0.03 * 0.03
        var acc = [Float](repeating: 0, count: n)
        for c in 0..<a.channels {
            let x = Array(a.plane(c)), y = Array(b.plane(c))
            let mu1 = blur(x, w, h), mu2 = blur(y, w, h)
            let xx = blur(zip(x, x).map(*), w, h), yy = blur(zip(y, y).map(*), w, h), xy = blur(zip(x, y).map(*), w, h)
            for i in 0..<n {
                let m1 = mu1[i], m2 = mu2[i]
                let s1 = xx[i] - m1 * m1, s2 = yy[i] - m2 * m2, s12 = xy[i] - m1 * m2
                acc[i] += ((2 * m1 * m2 + C1) * (2 * s12 + C2)) / ((m1 * m1 + m2 * m2 + C1) * (s1 + s2 + C2))
            }
        }
        return acc.map { $0 / Float(a.channels) }
    }
}
