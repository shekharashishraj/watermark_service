import Foundation
import simd

/// The height of the surface the scene's objects rest on (the table), in world y (+y up). Changes whose points lie
/// within a centimetre of it are the sheet itself (a cloth's wrinkles, LiDAR and model depth disagreeing on a flat
/// surface), not objects (spec §4.4). Without a reliable estimate the object layer is off for the run (R35, R37).
public enum SupportPlane {
    /// The densest 1 cm band of y and how much of the evidence it holds.
    public struct Estimate: Equatable, Sendable {
        /// The median y of the band's points (world y, metres).
        public var height: Float
        /// The band's points as a fraction of all the points inside the ROI (0…1).
        public var share: Float
        /// Points inside the ROI.
        public var points: Int
        /// True when the band holds at least `minShare` of the points: a table, not one band among many (R37).
        public var isReliable: Bool { share >= SupportPlane.minShare }
    }

    /// The smallest share of the in-ROI points the densest band must hold to be taken as the table (R37). A table
    /// seen from above puts most of its points in one or two 1 cm bands; a cluttered or tilted scene spreads them.
    public static let minShare: Float = 0.15

    /// The densest `binSize` band of y over the points inside `roi` (all points when nil), its height refined by the
    /// median y of the band's points, with its share of the points, however small. Nil when no point is inside.
    public static func densestBand(points: [SIMD3<Float>], roi: AABB?, binSize: Float = 0.01) -> Estimate? {
        guard binSize > 0 else { return nil }
        var bins: [Int: [Float]] = [:]
        var total = 0
        for p in points where p.y.isFinite {
            if let roi, !roi.contains(p) { continue }
            bins[Int((p.y / binSize).rounded(.down)), default: []].append(p.y)
            total += 1
        }
        // Most points wins; on a tie, the lower bin (deterministic).
        guard let winner = bins.max(by: { ($0.value.count, -$0.key) < ($1.value.count, -$1.key) }) else { return nil }
        let ys = winner.value.sorted()
        let median = ys.count % 2 == 1 ? ys[ys.count / 2] : (ys[ys.count / 2 - 1] + ys[ys.count / 2]) / 2
        return Estimate(height: median, share: Float(ys.count) / Float(total), points: total)
    }

    /// The table's height: the densest band's (`densestBand`), or nil when no point is inside or the band holds less
    /// than `minShare` of them (no reliable table; the object layer is then off, R35).
    public static func height(points: [SIMD3<Float>], roi: AABB?, binSize: Float = 0.01, minShare: Float = minShare) -> Float? {
        densestBand(points: points, roi: roi, binSize: binSize).flatMap { $0.share >= minShare ? $0.height : nil }
    }

    /// `densestBand` over a scan's high-confidence LiDAR (as `SceneRegion.estimate` reads it), at most 60 keyframes
    /// evenly strided.
    public static func densestBand(of store: SessionStore, roi: AABB?) throws -> Estimate? {
        densestBand(points: try lidarPoints(of: store), roi: roi)
    }

    /// `height` over a scan's high-confidence LiDAR, as `densestBand(of:roi:)` reads it.
    public static func height(of store: SessionStore, roi: AABB?) throws -> Float? {
        try densestBand(of: store, roi: roi).flatMap { $0.isReliable ? $0.height : nil }
    }

    static func lidarPoints(of store: SessionStore) throws -> [SIMD3<Float>] {
        let m = store.manifest
        let stride = max((m.keyframes.count + 59) / 60, 1)
        var points: [SIMD3<Float>] = []
        for (i, kf) in m.keyframes.enumerated() where i % stride == 0 {
            let rgb = try store.array(.keyframes, id: kf.id, name: "rgb").uint8()
            let (depth, conf) = try store.depth(.keyframes, id: kf.id)
            points += ColmapExport.lidarPoints(depth: depth, confidence: conf, depthWidth: m.depthWidth, depthHeight: m.depthHeight,
                                               intrinsics: m.intrinsics, arkitCameraToWorld: kf.arkitCameraToWorld, rgb: rgb)
                .map { SIMD3<Float>($0.xyz) }
        }
        return points
    }
}
