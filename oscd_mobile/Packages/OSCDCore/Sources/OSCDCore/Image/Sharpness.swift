/// PocketGS's sharpness heuristic (Eq. 2): the mean of |I(x+∆, y) − I(x, y)| + |I(x, y+∆) − I(x, y)| over a
/// luma grid with stride ∆. Blur spreads edges over many pixels, so neighbours ∆ apart differ less.
/// Only comparable between frames of nearly the same view: a sharp blank wall also scores low.
public enum Sharpness {
    public static func score(luma: UnsafePointer<UInt8>, width: Int, height: Int, rowBytes: Int, stride step: Int = 4) -> Double {
        guard width > step, height > step else { return 0 }
        var sum = 0, n = 0
        for y in Swift.stride(from: 0, to: height - step, by: step) {
            let row = luma + y * rowBytes, below = row + step * rowBytes
            var x = 0
            while x < width - step {
                let c = Int(row[x])
                sum += abs(Int(row[x + step]) - c) + abs(Int(below[x]) - c)
                n += 1
                x += step
            }
        }
        return Double(sum) / Double(n)
    }

    public static func score(luma: [UInt8], width: Int, height: Int, stride step: Int = 4) -> Double {
        luma.withUnsafeBufferPointer { score(luma: $0.baseAddress!, width: width, height: height, rowBytes: width, stride: step) }
    }
}
