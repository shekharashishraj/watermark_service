import Foundation

/// O-SCD `Detector.detectAndCompute` after the network (poses/feature_detector.py).
public enum XFeatPostprocess {
    public static let detectionThreshold: Float = 0.05
    public static let nmsKernel = 5

    /// - feats: 64×h×w descriptors, already L2-normalised over channels
    /// - heatmap: H×W keypoint heatmap (H = 8h, W = 8w), XFeat `get_kpts_heatmap`
    /// - reliability: h×w
    /// Returns keypoints with a positive score, best first, in the original image's pixel grid.
    public static func run(feats: [Float], heatmap: [Float], reliability: [Float],
                           h: Int, w: Int, topK: Int, imageWidth: Int, imageHeight: Int) -> DescribedKeypoints {
        let H = h * 8, W = w * 8
        let nvalid = Int(1.5 * Double(topK))

        // 1. NMS: thresholded local maxima of a 5×5 window (MaxPool2d pads with -inf; ties count as maxima).
        var cand: [(v: Float, i: Int)] = []
        let r = nmsKernel / 2
        for y in 0..<H {
            for x in 0..<W {
                let v = heatmap[y * W + x]
                guard v > detectionThreshold else { continue }
                var isMax = true
                outer: for yy in max(0, y - r)...min(H - 1, y + r) {
                    for xx in max(0, x - r)...min(W - 1, x + r) where heatmap[yy * W + xx] > v {
                        isMax = false
                        break outer
                    }
                }
                if isMax { cand.append((v, y * W + x)) }
            }
        }
        cand.sort { $0.v != $1.v ? $0.v > $1.v : $0.i < $1.i }
        if cand.count > nvalid { cand.removeLast(cand.count - nvalid) }

        // 2. score = nearest(heatmap) × bilinear(reliability); a keypoint at (0, 0) scores -1.
        var scored: [(s: Float, x: Int, y: Int)] = cand.map { c in
            let x = c.i % W, y = c.i / W
            if x == 0 && y == 0 { return (-1, x, y) }
            let gx = gridCoord(x, W), gy = gridCoord(y, H)
            return (sampleNearest(heatmap, W, H, gx, gy) * sampleBilinear(reliability, w, h, gx, gy), x, y)
        }
        scored.sort { $0.s != $1.s ? $0.s > $1.s : ($0.y * W + $0.x) < ($1.y * W + $1.x) }
        if scored.count > topK { scored.removeLast(scored.count - topK) }
        scored.removeAll { $0.s <= 0 }

        // 3. Bicubic descriptors, L2-normalised. 4. Rescale positions to the original image.
        let rw = Float(imageWidth) / Float(W), rh = Float(imageHeight) / Float(H)
        var kpts: [SIMD2<Float>] = []
        kpts.reserveCapacity(scored.count)
        var out = [Float](repeating: 0, count: scored.count * 64)
        for (n, k) in scored.enumerated() {
            sampleBicubic(feats, channels: 64, w, h, gridCoord(k.x, W), gridCoord(k.y, H), into: &out, offset: n * 64)
            var norm: Float = 0
            for c in 0..<64 { norm += out[n * 64 + c] * out[n * 64 + c] }
            norm = max(norm.squareRoot(), 1e-12)
            for c in 0..<64 { out[n * 64 + c] /= norm }
            kpts.append(SIMD2(Float(k.x) * rw, Float(k.y) * rh))
        }
        return DescribedKeypoints(kpts: kpts, feats: out)
    }

    /// InterpolateSparse2d.normgrid, then the cast to the feature dtype (fp16 in O-SCD).
    @inline(__always) static func gridCoord(_ p: Int, _ size: Int) -> Float {
        let g = 2 * (Float(p) / Float(size - 1)) - 1
        #if arch(arm64)
        return Float(Float16(g))
        #else
        return g
        #endif
    }

    /// grid_sample un-normalisation with align_corners = False.
    @inline(__always) static func unnormalize(_ g: Float, _ size: Int) -> Float { ((g + 1) * Float(size) - 1) / 2 }

    static func sampleNearest(_ m: [Float], _ w: Int, _ h: Int, _ gx: Float, _ gy: Float) -> Float {
        let ix = Int(unnormalize(gx, w).rounded(.toNearestOrEven))
        let iy = Int(unnormalize(gy, h).rounded(.toNearestOrEven))
        guard ix >= 0, ix < w, iy >= 0, iy < h else { return 0 }
        return m[iy * w + ix]
    }

    static func sampleBilinear(_ m: [Float], _ w: Int, _ h: Int, _ gx: Float, _ gy: Float) -> Float {
        let px = unnormalize(gx, w), py = unnormalize(gy, h)
        let x0 = Int(floor(px)), y0 = Int(floor(py))
        let tx = px - Float(x0), ty = py - Float(y0)
        func at(_ x: Int, _ y: Int) -> Float { (x >= 0 && x < w && y >= 0 && y < h) ? m[y * w + x] : 0 }
        return at(x0, y0) * (1 - tx) * (1 - ty) + at(x0 + 1, y0) * tx * (1 - ty)
             + at(x0, y0 + 1) * (1 - tx) * ty + at(x0 + 1, y0 + 1) * tx * ty
    }

    /// PyTorch cubic convolution weights (A = -0.75) for fractional offset t.
    static func cubicWeights(_ t: Float) -> [Float] {
        let A: Float = -0.75
        func c1(_ x: Float) -> Float { ((A + 2) * x - (A + 3)) * x * x + 1 }
        func c2(_ x: Float) -> Float { ((A * x - 5 * A) * x + 8 * A) * x - 4 * A }
        return [c2(t + 1), c1(t), c1(1 - t), c2(2 - t)]
    }

    /// grid_sample(mode='bicubic', padding_mode='zeros', align_corners=False) at one point, all channels.
    static func sampleBicubic(_ m: [Float], channels: Int, _ w: Int, _ h: Int, _ gx: Float, _ gy: Float,
                              into out: inout [Float], offset: Int) {
        let px = unnormalize(gx, w), py = unnormalize(gy, h)
        let x0 = Int(floor(px)), y0 = Int(floor(py))
        let wx = cubicWeights(px - Float(x0)), wy = cubicWeights(py - Float(y0))
        for c in 0..<channels {
            var acc: Float = 0
            for j in 0..<4 {
                let yy = y0 - 1 + j
                guard yy >= 0, yy < h else { continue }
                var row: Float = 0
                for i in 0..<4 {
                    let xx = x0 - 1 + i
                    guard xx >= 0, xx < w else { continue }
                    row += wx[i] * m[(c * h + yy) * w + xx]
                }
                acc += wy[j] * row
            }
            out[offset + c] = acc
        }
    }
}
