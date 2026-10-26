import CoreGraphics
import Foundation
import ImageIO
import OSCDCore
import Vision

/// Names a changed object with Vision's image classifier (spec §4.9). Optional: the Settings toggle "Name objects"
/// (`nameObjects`, on by default) turns it off, and an object without a name is shown as "Object N".
final class ObjectNamer {
    /// Labels below this confidence are dropped; labels under 0.6 are shown with a "?" (`ObjectEvent.summary`).
    static let minConfidence: Float = 0.3
    static let defaultsKey = "nameObjects"

    /// True unless the user switched "Name objects" off.
    static var enabled: Bool {
        UserDefaults.standard.object(forKey: defaultsKey) == nil || UserDefaults.standard.bool(forKey: defaultsKey)
    }

    /// The images are the sensor's landscape frames; the phone is held upright, so they are shown rotated `.right`.
    private static let orientation = CGImagePropertyOrientation.right

    /// The Vision label `LabelChoice.pick` chooses (R34: a specific label tied with a generic top wins) at or above
    /// `minConfidence`, else nil. Runs synchronously on the calling thread.
    func name(_ image: CGImage) -> (label: String, confidence: Float)? {
        let request = VNClassifyImageRequest()
        let handler = VNImageRequestHandler(cgImage: image, orientation: Self.orientation, options: [:])
        do { try handler.perform([request]) } catch {
            AppLog.write("naming: Vision failed: \(error.localizedDescription)")
            return nil
        }
        let ranked = request.results ?? []
        // Vision lists a label's ancestors with (almost) its confidence ("recreation", "sport", "ball"), so the top few are logged.
        AppLog.write("naming: Vision on \(image.width)×\(image.height) px → "
                     + ranked.prefix(5).map { String(format: "%@ %.3f", $0.identifier, $0.confidence) }.joined(separator: ", "))
        guard let chosen = LabelChoice.pick(ranked.map { ($0.identifier, $0.confidence) }, minConfidence: Self.minConfidence) else {
            return nil
        }
        return (Self.display(chosen.0), chosen.1)
    }

    /// Vision's identifiers are lower case with underscores or commas ("tennis_ball", "sunglasses,shades"): the first
    /// alternative with the underscores as spaces, first letter capitalised. No mapping table.
    static func display(_ identifier: String) -> String {
        let first = identifier.split(separator: ",").first.map(String.init) ?? identifier
        let s = first.replacingOccurrences(of: "_", with: " ").trimmingCharacters(in: .whitespaces)
        return s.prefix(1).uppercased() + s.dropFirst()
    }

    static func cgImage(_ crop: FloatImage) -> CGImage? {
        let w = crop.width, h = crop.height, n = w * h
        guard crop.channels >= 3, w > 0, h > 0 else { return nil }
        var rgba = [UInt8](repeating: 255, count: n * 4)
        for i in 0..<n { for c in 0..<3 { rgba[i * 4 + c] = UInt8(max(0, min(255, (crop.data[c * n + i] * 255).rounded()))) } }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                       space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }

    /// Full-resolution crop of a before keyframe: `rect` is in working pixels; scaled by captureWidth / intrinsics.width,
    /// with a 20 % margin. Nil when the keyframe has no full-size image (the scan was recorded without one).
    static func beforeCrop(store: SessionStore, keyframeID: Int, rect: PixelRect) -> CGImage? {
        let url = store.fullImageURL(.keyframes, id: keyframeID)
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let full = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            // No full-size image: the working-resolution keyframe, cropped the way the tracker crops.
            guard let rgb = try? store.rgb(.keyframes, id: keyframeID) else { return nil }
            return cgImage(ObjectTracker.crop(rgb, rect))
        }
        let K = store.manifest.intrinsics
        let sx = Double(full.width) / Double(K.width), sy = Double(full.height) / Double(K.height)
        let mx = Double(rect.maxX - rect.minX + 1) * 0.2, my = Double(rect.maxY - rect.minY + 1) * 0.2
        let x0 = max(0, (Double(rect.minX) - mx) * sx), x1 = min(Double(full.width), (Double(rect.maxX + 1) + mx) * sx)
        let y0 = max(0, (Double(rect.minY) - my) * sy), y1 = min(Double(full.height), (Double(rect.maxY + 1) + my) * sy)
        guard x1 > x0, y1 > y0 else { return nil }
        return full.cropping(to: CGRect(x: x0.rounded(.down), y: y0.rounded(.down),
                                        width: (x1 - x0).rounded(.up), height: (y1 - y0).rounded(.up)))
    }
}
