import Foundation
import simd

/// Measures a move as a planar rigid motion (spec §4.7): translation plus yaw about the world +y axis.
///
/// The motion turns the source about its own centroid: `p' = R_y(yaw)·(p − c_s) + c_s + translation`, so
/// `translation` is the displacement of the source centroid wherever the object sits.
public enum PlanarICP {
    public struct Config: Sendable {
        public var yawCandidates = 12; public var iterations = 20; public var trim: Float = 0.8
        public var minPoints = 10
        /// A candidate counts as equally good when its RMS is within this share of the best RMS (10 %) ...
        public var symmetricRatio: Float = 0.1
        /// ... plus `rmsFloor` metres, so an exact copy (best RMS near zero) is not out-voted by sampling error alone.
        public var rmsFloor: Float = 0.002
        /// Equally good candidates must converge to the best yaw, or the best yaw + 180°, within this many degrees.
        public var symmetryToleranceDeg: Float = 15
        public var nnVoxel: Float = 0.01
        public var inlierDistance: Float = 0.02
        /// Nearest neighbours farther than this are not paired.
        public var maxPairDistance: Float = 0.08
        /// Source points used while iterating (evenly strided); the inlier share still uses every point.
        public var maxIterationPoints = 600
        public init() {}
    }

    /// Rigid planar motion (translation + yaw about +y) taking `source` (the reference surface) onto `target` (LiDAR now).
    /// Non-finite points are dropped first. nil when either set then has fewer than `minPoints`, or when any
    /// reported number would not be finite.
    public static func fit(source: [SIMD3<Float>], target: [SIMD3<Float>], config: Config = Config()) -> Motion? {
        let src = source.filter(isFinite), tgt = target.filter(isFinite)
        guard src.count >= config.minPoints, tgt.count >= config.minPoints, config.yawCandidates > 0 else { return nil }

        let cs = centroid(src), ct = centroid(tgt)
        let stride = max(1, src.count / max(1, config.maxIterationPoints))
        let iterSrc = Swift.stride(from: 0, to: src.count, by: stride).map { src[$0] }
        let grid = VoxelHash(points: tgt, cell: config.nnVoxel)

        var candidates: [State] = []
        for k in 0..<config.yawCandidates {
            let start = State(yaw: 360 * Float(k) / Float(config.yawCandidates), translation: ct - cs, rms: .infinity)
            candidates.append(refine(start, source: iterSrc, grid: grid, cs: cs, config: config))
        }
        guard let best = candidates.min(by: { $0.rms < $1.rms }), best.rms.isFinite else { return nil }

        let limit = best.rms * (1 + config.symmetricRatio) + config.rmsFloor
        let ambiguous = candidates.contains { c in
            c.rms <= limit && !(angleGap(c.yaw, best.yaw) <= config.symmetryToleranceDeg
                                || angleGap(c.yaw, best.yaw + 180) <= config.symmetryToleranceDeg)
        }

        let R = rotationY(best.yaw)
        let radius = Int((config.inlierDistance / config.nnVoxel).rounded(.up))
        let inliers = src.filter { p in
            grid.nearest(to: R * (p - cs) + cs + best.translation, maxRing: radius, maxDistance: config.inlierDistance) != nil
        }.count

        let motion = Motion(translation: best.translation,
                            horizontalDistance: simd_length(SIMD2<Float>(best.translation.x, best.translation.z)),
                            yawDeg: ambiguous ? nil : wrapHalfTurn(best.yaw),
                            fitRMS: best.rms, inlierFraction: Float(inliers) / Float(src.count))
        let numbers = [motion.translation.x, motion.translation.y, motion.translation.z, motion.horizontalDistance,
                       motion.yawDeg ?? 0, motion.fitRMS, motion.inlierFraction]
        return numbers.allSatisfy(\.isFinite) ? motion : nil
    }

    /// Standard right-handed rotation about +y: positive turns counter-clockwise seen from above.
    public static func rotationY(_ deg: Float) -> simd_float3x3 {
        let a = deg * .pi / 180, c = cos(a), s = sin(a)
        return simd_float3x3(columns: (SIMD3<Float>(c, 0, -s), SIMD3<Float>(0, 1, 0), SIMD3<Float>(s, 0, c)))
    }

    // MARK: - Internals

    private struct State { var yaw: Float; var translation: SIMD3<Float>; var rms: Float }

    private static func isFinite(_ p: SIMD3<Float>) -> Bool { p.x.isFinite && p.y.isFinite && p.z.isFinite }

    private static func centroid(_ pts: [SIMD3<Float>]) -> SIMD3<Float> {
        pts.reduce(SIMD3<Float>.zero, +) / Float(pts.count)
    }

    /// Smallest absolute difference between two angles in degrees, 0…180.
    private static func angleGap(_ a: Float, _ b: Float) -> Float {
        var d = (a - b).truncatingRemainder(dividingBy: 360)
        if d > 180 { d -= 360 } else if d < -180 { d += 360 }
        return abs(d)
    }

    /// An angle in degrees folded into (−90, 90]: a half turn looks the same, so the smaller rotation is reported.
    private static func wrapHalfTurn(_ deg: Float) -> Float {
        var d = deg.truncatingRemainder(dividingBy: 180)
        if d > 90 { d -= 180 } else if d <= -90 { d += 180 }
        return d
    }

    /// Trimmed point-to-point ICP from `start`; the returned RMS is that of the trimmed pairs under the final motion.
    private static func refine(_ start: State, source: [SIMD3<Float>], grid: VoxelHash, cs: SIMD3<Float>, config: Config) -> State {
        var s = start
        let radius = Int((config.maxPairDistance / config.nnVoxel).rounded(.up))
        for iteration in 0...config.iterations {
            let R = rotationY(s.yaw)
            var pairs: [(src: SIMD3<Float>, tgt: SIMD3<Float>, d2: Float)] = []
            pairs.reserveCapacity(source.count)
            for p in source {
                let moved = R * (p - cs) + cs + s.translation
                if let q = grid.nearest(to: moved, maxRing: radius, maxDistance: config.maxPairDistance) {
                    pairs.append((p, q, simd_distance_squared(moved, q)))
                }
            }
            guard pairs.count >= config.minPoints else { s.rms = .infinity; return s }
            // Trimming drops the worst pairs, which carry the rotation: the first half of the run keeps them all.
            let share = iteration < config.iterations / 2 ? 1 : config.trim
            pairs.sort { $0.d2 < $1.d2 }
            let kept = pairs.prefix(max(config.minPoints, Int((Float(pairs.count) * share).rounded())))
            s.rms = (kept.reduce(0) { $0 + $1.d2 } / Float(kept.count)).squareRoot()
            if iteration == config.iterations { break }

            // Closed form: yaw from the xz cross-covariance of the centred pairs, then the translation.
            let n = Float(kept.count)
            let ms = kept.reduce(SIMD3<Float>.zero) { $0 + $1.src } / n, mt = kept.reduce(SIMD3<Float>.zero) { $0 + $1.tgt } / n
            var a: Float = 0, b: Float = 0
            for k in kept {
                let u = k.src - ms, v = k.tgt - mt
                a += v.x * u.x + v.z * u.z
                b += v.x * u.z - v.z * u.x
            }
            if a != 0 || b != 0 { s.yaw = atan2(b, a) * 180 / .pi }
            s.translation = mt - cs - rotationY(s.yaw) * (ms - cs)
        }
        return s
    }

    /// Nearest-neighbour lookup through a hash of voxel cells.
    private struct VoxelHash {
        let cell: Float
        var cells: [Int: [SIMD3<Float>]] = [:]

        init(points: [SIMD3<Float>], cell: Float) {
            self.cell = cell
            for p in points { cells[VoxelHash.key(VoxelHash.index(p, cell)), default: []].append(p) }
        }

        static func index(_ p: SIMD3<Float>, _ cell: Float) -> SIMD3<Int> {
            SIMD3<Int>((p.x / cell).rounded(.down).clampedToInt, (p.y / cell).rounded(.down).clampedToInt,
                       (p.z / cell).rounded(.down).clampedToInt)
        }

        static func key(_ i: SIMD3<Int>) -> Int {
            let m = 0x1FFFFF
            return ((i.x + 0x100000) & m) << 42 | ((i.y + 0x100000) & m) << 21 | ((i.z + 0x100000) & m)
        }

        /// The closest stored point within `maxDistance`, scanning shells of cells outward until nothing nearer can exist.
        func nearest(to p: SIMD3<Float>, maxRing: Int, maxDistance: Float) -> SIMD3<Float>? {
            let c = VoxelHash.index(p, cell)
            var best: SIMD3<Float>?, bestD2 = maxDistance * maxDistance
            for r in 0...maxRing {
                for dx in -r...r { for dy in -r...r {
                    // Interior of the shell is already scanned: only the faces touch ring r.
                    let step = (abs(dx) == r || abs(dy) == r) ? 1 : max(1, 2 * r)
                    for dz in Swift.stride(from: -r, through: r, by: step) {
                        guard let pts = cells[VoxelHash.key(SIMD3<Int>(c.x + dx, c.y + dy, c.z + dz))] else { continue }
                        for q in pts {
                            let d2 = simd_distance_squared(q, p)
                            if d2 <= bestD2 { bestD2 = d2; best = q }
                        }
                    }
                } }
                // Everything not yet scanned is at least r cells away.
                if bestD2 <= (Float(r) * cell) * (Float(r) * cell), best != nil { break }
            }
            return best
        }
    }
}

private extension Float {
    /// Whole-number cell index; keeps a wild coordinate from trapping the conversion.
    var clampedToInt: Int { Int(Swift.max(-1_000_000, Swift.min(1_000_000, self))) }
}
