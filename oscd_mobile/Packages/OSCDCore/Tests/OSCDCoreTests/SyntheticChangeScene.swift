import simd
@testable import OSCDCore

/// A table (the plane y = `tableY`, world +y up) with spheres and yawed boxes on it, ray-cast analytically into the
/// `FrameContext` the object layer consumes: the render (`dRen`, `render`, `embRender`) from `before`, the live frame
/// (`dLive`, `live`, `embLive`) from `after`.
struct SyntheticChangeScene {
    enum Shape {
        case sphere(centre: SIMD3<Float>, radius: Float)
        /// `yawDeg` turns the box about +y, as `PlanarICP.rotationY` (counter-clockwise seen from above).
        case box(centre: SIMD3<Float>, half: SIMD3<Float>, yawDeg: Float)
    }
    struct Object { var shape: Shape; var feature: [Float]; var colour: SIMD3<Float> }
    /// A disc of the table (world x, z) where the after scene's live depth reads `offset` metres farther than the
    /// table: the sheet-level false positives Method B's masks show on the phone (|d| 10–50 mm, flagged as changed).
    struct SheetPatch { var centre: SIMD2<Float>; var radius: Float; var offset: Float }

    var before: [Object]
    var after: [Object]
    var tableY: Float = 0
    var tableFeature: [Float] = (0..<256).map { $0 >= 224 ? 1 : 0 }
    var tableColour = SIMD3<Float>(0.55, 0.45, 0.35)
    /// Sheet-level depth noise added to the after scene's table pixels (none by default).
    var sheetNoise: [SheetPatch] = []
    /// The splat model's own depth error over the sheet, added to the before scene's table pixels in `dRen` only
    /// (negative = the model's sheet floats nearer the camera, as on Scan 4). Changed pixels follow the noisy `dRen`.
    var modelNoise: [SheetPatch] = []
    /// Also fill `dBefore`: the before scene's true depth (the before-scan's LiDAR, free of `modelNoise`).
    var beforeLiDAR = false

    static let intrinsics = Intrinsics.oscdCentered(f: 140, width: 160, height: 120)
    static let grid = 64, channels = 256

    /// Ray-casts both scenes at a pose (160×120, f = 140): dRen from `before`, dLive from `after`, images from colours,
    /// embeddings painted per cell from the object most of the cell's pixels see (table cells get `tableFeature`),
    /// valid = both rays hit inside `roi`, changed = valid ∧ |dLive − dRen| > 0.01.
    func frame(id: Int, pose: Pose, roi: AABB) -> FrameContext {
        let K = Self.intrinsics
        var ren = cast(before, pose: pose)
        let trueBefore = ren.depth
        for i in ren.depth.indices where ren.hit[i] == -1 {
            let p = SIMD2(ren.points[i].x, ren.points[i].z)
            if let patch = modelNoise.first(where: { simd_distance(p, $0.centre) < $0.radius }) { ren.depth[i] += patch.offset }
        }
        var live = cast(after, pose: pose)
        for i in live.depth.indices where live.hit[i] == -1 {
            let p = SIMD2(live.points[i].x, live.points[i].z)
            if let patch = sheetNoise.first(where: { simd_distance(p, $0.centre) < $0.radius }) { live.depth[i] += patch.offset }
        }
        let n = K.width * K.height
        var valid = [Bool](repeating: false, count: n), changed = valid
        for i in 0..<n {
            guard ren.depth[i].isFinite, live.depth[i].isFinite,
                  inside(ren.points[i], roi), inside(live.points[i], roi) else { continue }
            valid[i] = true
            changed[i] = abs(live.depth[i] - ren.depth[i]) > 0.01
        }
        return FrameContext(frameID: id, pose: pose, intrinsics: K, changed: changed, valid: valid,
                            dLive: live.depth, dRen: ren.depth,
                            live: image(live.hit, after), render: image(ren.hit, before),
                            embLive: embedding(live.hit, after), embRender: embedding(ren.hit, before),
                            dBefore: beforeLiDAR ? trueBefore : nil)
    }

    /// `count` cameras evenly spaced on a circle of `radius` at `height` above the origin, all looking at the origin
    /// (world→camera, OpenCV axes).
    static func orbitPoses(count: Int = 10, radius: Float = 0.5, height: Float = 0.45) -> [Pose] {
        (0..<count).map { i in
            OrbitCamera(target: .zero, yaw: 2 * .pi * Float(i) / Float(count), pitch: atan2(height, radius),
                        distance: (radius * radius + height * height).squareRoot()).pose
        }
    }

    // MARK: - Ray casting

    /// Per pixel: depth along the optical axis (NaN for no hit), world hit point, and what was hit
    /// (object index, −1 for the table, −2 for nothing).
    private func cast(_ objects: [Object], pose: Pose) -> (depth: [Float], points: [SIMD3<Float>], hit: [Int]) {
        let K = Self.intrinsics, n = K.width * K.height
        let origin = pose.centre, Rt = pose.R.transpose
        var depth = [Float](repeating: .nan, count: n), points = [SIMD3<Float>](repeating: .zero, count: n)
        var hit = [Int](repeating: -2, count: n)
        for y in 0..<K.height {
            for x in 0..<K.width {
                // Camera ray with z = 1, so the ray parameter is the depth.
                let dir = Rt * SIMD3((Float(x) - K.cx) / K.f, (Float(y) - K.cy) / K.f, 1)
                var best = Float.infinity, what = -2
                if dir.y < 0, origin.y > tableY { best = (tableY - origin.y) / dir.y; what = -1 }
                for (k, o) in objects.enumerated() {
                    if let t = Self.intersect(o.shape, origin: origin, dir: dir), t < best { best = t; what = k }
                }
                guard what != -2 else { continue }
                let i = y * K.width + x
                depth[i] = best; points[i] = origin + best * dir; hit[i] = what
            }
        }
        return (depth, points, hit)
    }

    /// Nearest positive ray parameter at which `origin + t·dir` meets the shape.
    static func intersect(_ shape: Shape, origin: SIMD3<Float>, dir: SIMD3<Float>) -> Float? {
        switch shape {
        case let .sphere(c, r):
            let oc = origin - c
            let a = simd_dot(dir, dir), b = simd_dot(oc, dir), cc = simd_dot(oc, oc) - r * r
            let disc = b * b - a * cc
            guard disc >= 0 else { return nil }
            let s = disc.squareRoot()
            let t0 = (-b - s) / a, t1 = (-b + s) / a
            return t0 > 0 ? t0 : (t1 > 0 ? t1 : nil)
        case let .box(c, half, yaw):
            // Slab test in the box's own axes.
            let toLocal = PlanarICP.rotationY(yaw).transpose
            let o = toLocal * (origin - c), d = toLocal * dir
            var tNear = -Float.infinity, tFar = Float.infinity
            for k in 0..<3 {
                if abs(d[k]) < 1e-9 {
                    if abs(o[k]) > half[k] { return nil }
                    continue
                }
                var t0 = (-half[k] - o[k]) / d[k], t1 = (half[k] - o[k]) / d[k]
                if t0 > t1 { swap(&t0, &t1) }
                tNear = max(tNear, t0); tFar = min(tFar, t1)
                if tNear > tFar { return nil }
            }
            return tNear > 0 ? tNear : (tFar > 0 ? tFar : nil)
        }
    }

    private func inside(_ p: SIMD3<Float>, _ roi: AABB) -> Bool {
        all(p .>= roi.min) && all(p .<= roi.max)
    }

    // MARK: - Images and embeddings

    private func image(_ hit: [Int], _ objects: [Object]) -> FloatImage {
        let K = Self.intrinsics, n = K.width * K.height
        var img = FloatImage(width: K.width, height: K.height, channels: 3)
        for i in 0..<n where hit[i] != -2 {
            let c = hit[i] == -1 ? tableColour : objects[hit[i]].colour
            img.data[i] = c.x; img.data[n + i] = c.y; img.data[2 * n + i] = c.z
        }
        return img
    }

    /// [256, 64, 64] channel-major; a cell takes the feature of whatever most of its pixels hit (zeros for nothing).
    private func embedding(_ hit: [Int], _ objects: [Object]) -> [Float] {
        let K = Self.intrinsics, g = Self.grid, cells = g * g
        // Votes per cell for: nothing, table, object 0, object 1, ...
        let kinds = objects.count + 2
        var votes = [Int](repeating: 0, count: cells * kinds)
        for y in 0..<K.height {
            for x in 0..<K.width {
                let cell = (y * g / K.height) * g + x * g / K.width
                votes[cell * kinds + hit[y * K.width + x] + 2] += 1
            }
        }
        var emb = [Float](repeating: 0, count: Self.channels * cells)
        for cell in 0..<cells {
            let row = votes[(cell * kinds)..<(cell * kinds + kinds)]
            guard let winner = row.indices.max(by: { row[$0] < row[$1] }), row[winner] > 0 else { continue }
            let what = winner - cell * kinds - 2
            guard what != -2 else { continue }
            let f = what == -1 ? tableFeature : objects[what].feature
            for c in 0..<min(f.count, Self.channels) { emb[c * cells + cell] = f[c] }
        }
        return emb
    }
}
