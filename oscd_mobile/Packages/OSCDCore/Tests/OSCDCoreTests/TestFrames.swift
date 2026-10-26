@testable import OSCDCore

/// A small synthetic `FrameContext`: flat grey live and render images, `valid` everywhere, both SAM2
/// embeddings painted in every cell with `feature` (256 values; zeros when empty), and depths and the
/// changed mask from the closures, called with (x, y).
func frame(id: Int, width: Int = 32, height: Int = 24, pose: Pose, f: Float = 30,
           dLive: (Int, Int) -> Float, dRen: (Int, Int) -> Float, changed: (Int, Int) -> Bool,
           feature: [Float] = []) -> FrameContext {
    let n = width * height
    let pixels = (0..<n).map { ($0 % width, $0 / width) }
    let cells = 64 * 64
    var emb = [Float](repeating: 0, count: 256 * cells)
    for (c, v) in feature.enumerated() where v != 0 {
        for k in 0..<cells { emb[c * cells + k] = v }
    }
    let grey = FloatImage(width: width, height: height, channels: 3, fill: 0.5)
    return FrameContext(frameID: id, pose: pose, intrinsics: .oscdCentered(f: f, width: width, height: height),
                        changed: pixels.map { changed($0.0, $0.1) }, valid: [Bool](repeating: true, count: n),
                        dLive: pixels.map { dLive($0.0, $0.1) }, dRen: pixels.map { dRen($0.0, $0.1) },
                        live: grey, render: grey, embLive: emb, embRender: emb)
}
