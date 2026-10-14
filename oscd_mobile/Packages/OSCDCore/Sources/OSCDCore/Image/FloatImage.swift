import Foundation

/// Planar (CHW) float image; RGB values are in [0, 1].
public struct FloatImage: Sendable {
    public let width: Int
    public let height: Int
    public let channels: Int
    public var data: [Float]   // data[(c * height + y) * width + x]

    public init(width: Int, height: Int, channels: Int, data: [Float]) {
        precondition(data.count == width * height * channels, "data size does not match dimensions")
        self.width = width; self.height = height; self.channels = channels; self.data = data
    }

    public init(width: Int, height: Int, channels: Int, fill: Float = 0) {
        self.init(width: width, height: height, channels: channels,
                  data: [Float](repeating: fill, count: width * height * channels))
    }

    /// From interleaved HWC values (e.g. a numpy (H, W, C) array).
    public init(hwc: [Float], width: Int, height: Int, channels: Int) {
        var d = [Float](repeating: 0, count: hwc.count)
        for y in 0..<height { for x in 0..<width { for c in 0..<channels {
            d[(c * height + y) * width + x] = hwc[(y * width + x) * channels + c]
        } } }
        self.init(width: width, height: height, channels: channels, data: d)
    }

    /// From interleaved 8-bit RGB.
    public init(rgb8: [UInt8], width: Int, height: Int) {
        self.init(hwc: rgb8.map { Float($0) / 255 }, width: width, height: height, channels: 3)
    }

    @inline(__always) public subscript(c: Int, y: Int, x: Int) -> Float {
        get { data[(c * height + y) * width + x] }
        set { data[(c * height + y) * width + x] = newValue }
    }

    public func plane(_ c: Int) -> ArraySlice<Float> { data[(c * width * height)..<((c + 1) * width * height)] }

    /// torch F.interpolate(mode='bilinear', align_corners=False).
    public func resizedBilinear(width ow: Int, height oh: Int) -> FloatImage {
        let xs = Self.linearTaps(inSize: width, outSize: ow)
        let ys = Self.linearTaps(inSize: height, outSize: oh)
        var out = FloatImage(width: ow, height: oh, channels: channels)
        for c in 0..<channels {
            for (oy, ty) in ys.enumerated() {
                for (ox, tx) in xs.enumerated() {
                    let v00 = self[c, ty.i0, tx.i0], v01 = self[c, ty.i0, tx.i1]
                    let v10 = self[c, ty.i1, tx.i0], v11 = self[c, ty.i1, tx.i1]
                    out[c, oy, ox] = (1 - ty.w) * ((1 - tx.w) * v00 + tx.w * v01)
                                   + ty.w * ((1 - tx.w) * v10 + tx.w * v11)
                }
            }
        }
        return out
    }

    /// Source taps for one axis: src = max(0, (dst + 0.5) * in / out - 0.5).
    static func linearTaps(inSize: Int, outSize: Int) -> [(i0: Int, i1: Int, w: Float)] {
        let scale = Float(inSize) / Float(outSize)
        return (0..<outSize).map { o in
            let src = max(0, (Float(o) + 0.5) * scale - 0.5)
            let i0 = min(Int(src), inSize - 1)
            return (i0, min(i0 + 1, inSize - 1), src - Float(i0))
        }
    }

    /// torch F.interpolate(mode='nearest'): src = min(floor(dst * in / out), in - 1).
    public func resizedNearest(width ow: Int, height oh: Int) -> FloatImage {
        let sx = Float(width) / Float(ow), sy = Float(height) / Float(oh)
        let xi = (0..<ow).map { min(Int(Float($0) * sx), width - 1) }
        let yi = (0..<oh).map { min(Int(Float($0) * sy), height - 1) }
        var out = FloatImage(width: ow, height: oh, channels: channels)
        for c in 0..<channels { for y in 0..<oh { for x in 0..<ow { out[c, y, x] = self[c, yi[y], xi[x]] } } }
        return out
    }

    /// `resizedNearest` written straight into a CHW float buffer of `channels × height × width`.
    public func writeResizedNearest(into dst: UnsafeMutablePointer<Float>, width ow: Int, height oh: Int) {
        let sx = Float(width) / Float(ow), sy = Float(height) / Float(oh)
        let xi = (0..<ow).map { min(Int(Float($0) * sx), width - 1) }
        data.withUnsafeBufferPointer { src in
            for c in 0..<channels { for y in 0..<oh {
                let row = src.baseAddress! + (c * height + min(Int(Float(y) * sy), height - 1)) * width
                let out = dst + (c * oh + y) * ow
                for x in 0..<ow { out[x] = row[xi[x]] }
            } }
        }
    }

    /// Mean over factor×factor blocks (OpenCV INTER_AREA for an integer factor).
    public func downsampledArea(factor k: Int) -> FloatImage {
        let ow = width / k, oh = height / k
        var out = FloatImage(width: ow, height: oh, channels: channels)
        let inv = 1 / Float(k * k)
        for c in 0..<channels { for y in 0..<oh { for x in 0..<ow {
            var s: Float = 0
            for dy in 0..<k { for dx in 0..<k { s += self[c, y * k + dy, x * k + dx] } }
            out[c, y, x] = s * inv
        } } }
        return out
    }
}
