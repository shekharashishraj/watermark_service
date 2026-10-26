import Foundation

public struct PixelRect: Equatable, Sendable {
    public var minX: Int, minY: Int, maxX: Int, maxY: Int
    public init(minX: Int, minY: Int, maxX: Int, maxY: Int) {
        self.minX = minX; self.minY = minY; self.maxX = maxX; self.maxY = maxY
    }
}

public enum ChangeSign {
    /// d = dLive − dRen − medianOffset over pixels where both are finite and valid; NaN elsewhere.
    /// sign = +1 if d < −tolerance (arrived), −1 if d > tolerance (left), else 0.
    /// offset = the frame's median(dLive − dRen) over those pixels (D13); NaN when there are none.
    public static func compute(dLive: [Float], dRen: [Float], valid: [Bool],
                               tolerance: Float = 0.005) -> (sign: [Int8], d: [Float], offset: Float) {
        let n = dLive.count
        var sign = [Int8](repeating: 0, count: n), d = [Float](repeating: .nan, count: n)
        let judged = (0..<n).filter { valid[$0] && dLive[$0].isFinite && dRen[$0].isFinite }
        guard !judged.isEmpty else { return (sign, d, .nan) }
        let diffs = judged.map { dLive[$0] - dRen[$0] }.sorted()
        let offset = diffs.count % 2 == 1 ? diffs[diffs.count / 2]
                                          : (diffs[diffs.count / 2 - 1] + diffs[diffs.count / 2]) / 2
        for i in judged {
            let v = dLive[i] - dRen[i] - offset
            d[i] = v
            if v < -tolerance { sign[i] = 1 } else if v > tolerance { sign[i] = -1 }
        }
        return (sign, d, offset)
    }
}

public struct Component2D: Sendable {
    public var sign: Int8
    public var pixels: [Int]
    public var rect: PixelRect
    public var feature: [Float] = []
    public var histogram: [Float] = []
}

public enum ChangeComponents {
    /// 4-connected components of the changed mask, split by depth sign where a blob mixes arrivals and
    /// departures. A blob with under 30 % signed pixels is one flat (sign 0) component; one whose
    /// smaller signed mass is at least 20 % of the total is re-labelled separately per sign (its
    /// unsigned pixels are dropped); otherwise it keeps the sign of the larger mass.
    public static func extract(changed: [Bool], sign: [Int8], d: [Float], width: Int, height: Int,
                               minPixels: Int = 40) -> [Component2D] {
        var out: [Component2D] = []
        for blob in label(changed, width: width, height: height) {
            let signed = blob.filter { sign[$0] != 0 }
            if Float(signed.count) / Float(blob.count) < 0.3 {
                out.append(component(sign: 0, blob, width: width)); continue
            }
            var plus: Float = 0, minus: Float = 0
            for i in signed { if sign[i] > 0 { plus += abs(d[i]) } else { minus += abs(d[i]) } }
            let total = plus + minus
            if total > 0, min(plus, minus) / total >= 0.2 {
                for s: Int8 in [1, -1] {
                    var mask = [Bool](repeating: false, count: width * height)
                    for i in signed where sign[i] == s { mask[i] = true }
                    for part in label(mask, width: width, height: height) { out.append(component(sign: s, part, width: width)) }
                }
            } else {
                let dominant: Int8 = total == 0 ? 0 : (plus >= minus ? 1 : -1)
                out.append(component(sign: dominant, blob, width: width))
            }
        }
        return out.filter { $0.pixels.count >= minPixels }
    }

    private static func component(sign: Int8, _ pixels: [Int], width: Int) -> Component2D {
        var r = PixelRect(minX: Int.max, minY: Int.max, maxX: Int.min, maxY: Int.min)
        for i in pixels {
            let x = i % width, y = i / width
            r.minX = min(r.minX, x); r.maxX = max(r.maxX, x)
            r.minY = min(r.minY, y); r.maxY = max(r.maxY, y)
        }
        return Component2D(sign: sign, pixels: pixels, rect: r)
    }

    /// 4-connected components of a boolean mask, in scan order of their first pixel.
    private static func label(_ mask: [Bool], width: Int, height: Int) -> [[Int]] {
        var seen = [Bool](repeating: false, count: mask.count)
        var blobs: [[Int]] = []
        var stack: [Int] = []
        for start in 0..<mask.count where mask[start] && !seen[start] {
            var pixels: [Int] = []
            seen[start] = true; stack.append(start)
            while let i = stack.popLast() {
                pixels.append(i)
                let x = i % width, y = i / width
                if x > 0, mask[i - 1], !seen[i - 1] { seen[i - 1] = true; stack.append(i - 1) }
                if x < width - 1, mask[i + 1], !seen[i + 1] { seen[i + 1] = true; stack.append(i + 1) }
                if y > 0, mask[i - width], !seen[i - width] { seen[i - width] = true; stack.append(i - width) }
                if y < height - 1, mask[i + width], !seen[i + width] { seen[i + width] = true; stack.append(i + width) }
            }
            blobs.append(pixels)
        }
        return blobs
    }
}
