import simd

/// A removed object has no live pixels, so its appearance comes from the before-scan's own photos:
/// the keyframe that sees most of the object's points, pooled over where those points land.
public enum BeforePhotoFeature {
    /// Points nearer than this to the camera plane do not project.
    static let minDepth: Float = 0.01

    /// The before keyframe into which most of `points` project (in front of the camera, inside the image), with those
    /// pixel indices (y·W + x, one per projecting point, duplicates kept). Ties go to the lower keyframe index; nil when
    /// no keyframe sees any point.
    public static func bestKeyframe(for points: [SIMD3<Float>], store: SessionStore) -> (record: FrameRecord, pixels: [Int])? {
        let K = store.manifest.intrinsics
        var best: (record: FrameRecord, pixels: [Int])?
        for record in store.manifest.keyframes {
            let pose = Pose.fromARKitCameraToWorld(simd_float4x4(columnMajor: record.arkitCameraToWorld))
            var pixels: [Int] = []
            for p in points {
                let c = pose.apply(p)
                guard c.z > minDepth else { continue }
                let uv = K.project(c)
                guard uv.x.isFinite, uv.y.isFinite else { continue }
                let x = Int(uv.x.rounded()), y = Int(uv.y.rounded())
                guard x >= 0, x < K.width, y >= 0, y < K.height else { continue }
                pixels.append(y * K.width + x)
            }
            if !pixels.isEmpty, pixels.count > (best?.pixels.count ?? 0) { best = (record, pixels) }
        }
        return best
    }

    /// Pools `encode(image)` over the projected pixels of the best keyframe's working-resolution RGB. `cache` holds one
    /// embedding per keyframe id, so instances on the same photo share one encode. Nil when no keyframe sees the points.
    public static func pooled(for points: [SIMD3<Float>], store: SessionStore, cache: inout [Int: [Float]],
                              encode: (FloatImage) throws -> [Float]) throws -> (feature: [Float], record: FrameRecord, rect: PixelRect)? {
        guard let best = bestKeyframe(for: points, store: store) else { return nil }
        let embedding: [Float]
        if let hit = cache[best.record.id] {
            embedding = hit
        } else {
            embedding = try encode(try store.rgb(.keyframes, id: best.record.id))
            cache[best.record.id] = embedding
        }
        let K = store.manifest.intrinsics
        let feature = SAM2Pool.mean(embedding: embedding, pixels: best.pixels, width: K.width, height: K.height)
        let xs = best.pixels.map { $0 % K.width }, ys = best.pixels.map { $0 / K.width }
        let rect = PixelRect(minX: xs.min()!, minY: ys.min()!, maxX: xs.max()!, maxY: ys.max()!)
        return (feature, best.record, rect)
    }
}
