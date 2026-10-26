import CoreGraphics
import Foundation
import ImageIO
import OSCDCore
import UniformTypeIdentifiers

/// PNG output for `--crops` (macOS only: the tool, not OSCDCore, links ImageIO).
enum PNG {
    /// A 3-plane [0, 1] image as 8-bit RGB, each pixel repeated `scale`×`scale` times, with optional 1-pixel
    /// rectangles (in source pixels) drawn in a colour.
    static func write(_ img: FloatImage, to url: URL, scale: Int = 1,
                      boxes: [(rect: PixelRect, rgb: SIMD3<UInt8>)] = []) throws {
        let s = max(scale, 1), w = img.width * s, h = img.height * s
        var bytes = [UInt8](repeating: 255, count: w * h * 4)
        func put(_ x: Int, _ y: Int, _ c: SIMD3<UInt8>) {
            for dy in 0..<s { for dx in 0..<s {
                let o = ((y * s + dy) * w + x * s + dx) * 4
                bytes[o] = c.x; bytes[o + 1] = c.y; bytes[o + 2] = c.z
            } }
        }
        for y in 0..<img.height {
            for x in 0..<img.width {
                func b(_ c: Int) -> UInt8 { UInt8(max(0, min(255, (img[min(c, img.channels - 1), y, x] * 255).rounded()))) }
                put(x, y, SIMD3(b(0), b(1), b(2)))
            }
        }
        for (r, c) in boxes {
            let x0 = max(r.minX, 0), x1 = min(r.maxX, img.width - 1), y0 = max(r.minY, 0), y1 = min(r.maxY, img.height - 1)
            guard x0 <= x1, y0 <= y1 else { continue }
            for x in x0...x1 { put(x, y0, c); put(x, y1, c) }
            for y in y0...y1 { put(x0, y, c); put(x1, y, c) }
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                               space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                               provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw CocoaError(.fileWriteUnknown) }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
    }

    /// Two images of the same height side by side, with a 4-pixel white gap.
    static func sideBySide(_ a: FloatImage, _ b: FloatImage) -> FloatImage {
        let gap = 4, w = a.width + gap + b.width
        var out = FloatImage(width: w, height: a.height, channels: 3, fill: 1)
        for c in 0..<3 {
            for y in 0..<a.height {
                for x in 0..<a.width { out[c, y, x] = a[c, y, x] }
                for x in 0..<b.width { out[c, y, a.width + gap + x] = b[c, y, x] }
            }
        }
        return out
    }
}
