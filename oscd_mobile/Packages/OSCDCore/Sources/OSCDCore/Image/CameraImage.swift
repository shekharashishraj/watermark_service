/// ARKit `capturedImage` (420f: full-range BT.601 luma plane + interleaved CbCr plane at half size).
public enum CameraImage {
    /// The integer block size that maps the capture size exactly onto the working size, or nil when
    /// there is none (wrong aspect ratio, non-integer ratio, or capture smaller than working).
    public static func areaFactor(captureWidth: Int, captureHeight: Int, workingWidth: Int, workingHeight: Int) -> Int? {
        guard workingWidth > 0, workingHeight > 0, captureWidth % workingWidth == 0 else { return nil }
        let k = captureWidth / workingWidth
        return k >= 1 && captureHeight == k * workingHeight ? k : nil
    }

    /// RGB bytes (H/factor × W/factor × 3), each the mean of a factor×factor block (OpenCV INTER_AREA).
    public static func rgbArea(y: UnsafePointer<UInt8>, yStride: Int, cbcr: UnsafePointer<UInt8>, cbcrStride: Int,
                               width: Int, height: Int, factor: Int) -> [UInt8] {
        let ow = width / factor, oh = height / factor
        var out = [UInt8](repeating: 0, count: ow * oh * 3)
        let inv = 1 / Float(factor * factor)
        for oy in 0..<oh {
            for ox in 0..<ow {
                var r: Float = 0, g: Float = 0, b: Float = 0
                for dy in 0..<factor {
                    let yy = oy * factor + dy
                    for dx in 0..<factor {
                        let xx = ox * factor + dx
                        let Y = Float(y[yy * yStride + xx])
                        let ci = (yy / 2) * cbcrStride + (xx / 2) * 2
                        let cb = Float(cbcr[ci]) - 128, cr = Float(cbcr[ci + 1]) - 128
                        r += min(255, max(0, Y + 1.402 * cr))
                        g += min(255, max(0, Y - 0.344136 * cb - 0.714136 * cr))
                        b += min(255, max(0, Y + 1.772 * cb))
                    }
                }
                let o = (oy * ow + ox) * 3
                out[o] = UInt8((r * inv).rounded()); out[o + 1] = UInt8((g * inv).rounded()); out[o + 2] = UInt8((b * inv).rounded())
            }
        }
        return out
    }
}
