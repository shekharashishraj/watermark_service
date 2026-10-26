import Foundation
import simd

public struct TrackerConfig: Sendable {
    public var voxelSize: Float = 0.01; public var minPixels = 40; public var overlapMin: Float = 0.3
    public var confirmFrames = 3; public var confirmVoxels = 15; public var roiEdgeMargin: Float = 0.02
    public var signTolerance: Float = 0.005; public var dropAfterObserved = 10; public var dropMinMisses = 5
    public var reservoir = 2000; public var matchAccept: Float = 0.55; public var matchLow: Float = 0.40
    public var roi: AABB?
    /// World y of the table (`SupportPlane`); when set, points less than `minHeightAboveSupport` above it are ignored.
    public var supportHeight: Float? = nil; public var minHeightAboveSupport: Float = 0.01
    public init() {}
}

/// A world point's voxel: floor(p / size) per axis.
public struct VoxelKey: Hashable, Sendable {
    public var x, y, z: Int32
    public init(_ p: SIMD3<Float>, size: Float) {
        let k = (p / size).rounded(.down)
        x = Int32(k.x); y = Int32(k.y); z = Int32(k.z)
    }
    var lower: SIMD3<Float> { SIMD3(Float(x), Float(y), Float(z)) }
}

public enum FeatureSource: String, Codable, Sendable { case render, live, beforePhoto }

/// One changed thing in 3D, accumulated from signed 2D components over frames (spec §4.4).
public struct TrackedInstance: Identifiable, Sendable {
    public let id: Int; public var sign: Int8
    public var voxels: [VoxelKey: Int] = [:]; public var frames: Set<Int> = []
    public var feature: [Float] = []; public var featureSource: FeatureSource = .live; public var histogram: [Float] = []
    public var points: [SIMD3<Float>] = []                // reservoir
    public var bestFrame = 0; public var bestRect = PixelRect(minX: 0, minY: 0, maxX: 0, maxY: 0); public var bestPixelCount = 0
    public var bestCrop: FloatImage?                      // live (sign ≥ 0) or render (sign < 0) crop of bestRect with 20 % margin
    public var firstSeen = 0; public var lastSeen = 0; public var misses = 0; public var observed = 0
    public var label: String?; public var labelConfidence: Float?
    public var voxelCount: Int { voxels.count }
    /// Hit-weighted voxel centres; cached by the tracker whenever `voxels` changes.
    public var centroid = SIMD3<Float>.zero
    /// Voxel centres ± half a voxel; cached with `centroid`.
    public var bbox = AABB(min: .zero, max: .zero)
    /// Pixels behind `feature` and `histogram` (their running means are pixel-weighted).
    var descriptorPixels = 0
    /// Points ever offered to the reservoir.
    var pointsSeen = 0

    public func isConfirmed(_ c: TrackerConfig) -> Bool { frames.count >= c.confirmFrames && voxelCount >= c.confirmVoxels }

    mutating func refreshGeometry(voxelSize s: Float) {
        guard !voxels.isEmpty else { return }
        var sum = SIMD3<Float>.zero, hits: Float = 0
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        for (k, n) in voxels {
            let c = k.lower
            sum += Float(n) * (c + 0.5); hits += Float(n)
            lo = simd_min(lo, c); hi = simd_max(hi, c)
        }
        centroid = sum / hits * s
        bbox = AABB(min: lo * s, max: (hi + 1) * s)
    }
}

public struct IngestStats: Sendable {
    public var components: Int; public var merged: Int; public var created: Int; public var dropped: Int
    public var milliseconds: Double
}

/// Incremental association of signed change components into 3D voxel instances (spec §4.4).
/// Instance ids count up from 0 in creation order and are never reused; merges keep the lowest id.
public final class ObjectTracker {
    public let config: TrackerConfig
    public private(set) var instances: [TrackedInstance] = []     // ascending id
    private var nextID = 0
    private var rng = SeededSampler(seed: 0x5EED)
    /// Measured moves per matched pair (see `setMotion`).
    var motionCache: [PairKey: CachedMotion] = [:]

    public init(config: TrackerConfig = TrackerConfig()) { self.config = config }

    /// One fused frame. `components` counts the components that kept at least one back-projected point.
    @discardableResult public func ingest(_ f: FrameContext) -> IngestStats {
        let start = DispatchTime.now().uptimeNanoseconds
        var stats = IngestStats(components: 0, merged: 0, created: 0, dropped: 0, milliseconds: 0)
        let w = f.intrinsics.width
        let (sign, d, offset) = ChangeSign.compute(dLive: f.dLive, dRen: f.dRen, valid: f.valid, tolerance: config.signTolerance)
        let components = ChangeComponents.extract(changed: f.changed, sign: sign, d: d, width: w,
                                                  height: f.intrinsics.height, minPixels: config.minPixels)
        let toWorld = f.pose.inverse
        var gain: SIMD3<Float>?
        var touched = Set<Int>()
        // Points less than `minHeightAboveSupport` above the table are the sheet, not an object (spec §4.4).
        let floorY = config.supportHeight.map { $0 + config.minHeightAboveSupport }
        // The model's depth aligned to LiDAR with the frame's D13 offset, so left points share LiDAR's table
        // (spec §4.4); plain dRen when no pixel was judged.
        let o: Float = offset.isFinite ? offset : 0
        for c in components {
            var points: [SIMD3<Float>] = [], cells: [VoxelKey: Int] = [:]
            for i in c.pixels {
                let z = c.sign >= 0 ? f.dLive[i] : f.dRen[i] + o
                guard z.isFinite, z > 0 else { continue }
                let uv = SIMD2(Float(i % w), Float(i / w))
                let p = toWorld.apply(f.intrinsics.unproject(uv, depth: z))
                if let floorY, p.y < floorY {
                    // A flat pixel counts when either surface (live or the aligned reference) clears the table.
                    let zRen = f.dRen[i] + o
                    guard c.sign == 0, zRen.isFinite, zRen > 0,
                          toWorld.apply(f.intrinsics.unproject(uv, depth: zRen)).y >= floorY else { continue }
                }
                points.append(p); cells[VoxelKey(p, size: config.voxelSize), default: 0] += 1
            }
            guard !points.isEmpty else { continue }
            stats.components += 1
            let feature: [Float], histogram: [Float]
            if c.sign >= 0 {
                feature = SAM2Pool.mean(embedding: f.embLive, pixels: c.pixels, width: w, height: f.intrinsics.height)
                histogram = LabHistogram.compute(image: f.live, pixels: c.pixels)
            } else {
                feature = SAM2Pool.mean(embedding: f.embRender, pixels: c.pixels, width: w, height: f.intrinsics.height)
                if gain == nil {
                    gain = CuesB.renderGain(live: f.live, render: f.render, liveDepth: f.dLive, renderDepth: f.dRen, valid: f.valid)
                }
                histogram = LabHistogram.compute(image: f.render, pixels: c.pixels, gain: gain!)
            }
            // Same-sign instances the component overlaps enough; all of them merge into the lowest id.
            let hits = instances.indices.filter {
                instances[$0].sign == c.sign && overlap(cells, instances[$0].voxels) >= config.overlapMin
            }
            var target: Int
            if let first = hits.first {
                for j in hits.dropFirst().reversed() { instances[first].absorb(instances[j], descriptors: true, rng: &rng, reservoir: config.reservoir) }
                for j in hits.dropFirst().reversed() { instances.remove(at: j) }
                target = first
                stats.merged += 1
            } else {
                var fresh = TrackedInstance(id: nextID, sign: c.sign)
                fresh.featureSource = c.sign >= 0 ? .live : .render
                fresh.firstSeen = f.frameID
                nextID += 1
                instances.append(fresh)
                target = instances.count - 1
                stats.created += 1
            }
            instances[target].add(c, cells: cells, points: points, feature: feature, histogram: histogram, frame: f,
                                  config: config, rng: &rng)
            touched.insert(instances[target].id)
        }

        // Flat instances overlapping a signed one are absorbed into it (lowest qualifying id).
        for j in instances.indices.reversed() where instances[j].sign == 0 {
            guard let k = instances.indices.first(where: {
                instances[$0].sign != 0 && overlap(instances[j].voxels, instances[$0].voxels) >= config.overlapMin
            }) else { continue }
            instances[k].absorb(instances[j], descriptors: false, rng: &rng, reservoir: config.reservoir)
            instances[k].refreshGeometry(voxelSize: config.voxelSize)
            if touched.contains(instances[j].id) { touched.insert(instances[k].id) }
            instances.remove(at: j)
        }

        // Instances reaching within the edge margin of the ROI box's four vertical faces are dropped, for all signs.
        // Not the floor or ceiling: an object resting on the table always reaches the ROI's floor band (spec §4.4).
        if let roi = config.roi {
            let m = config.roiEdgeMargin
            let before = instances.count
            instances.removeAll {
                let lo = $0.bbox.min - roi.min, hi = roi.max - $0.bbox.max
                return lo.x < m || lo.z < m || hi.x < m || hi.z < m
            }
            stats.dropped += before - instances.count
        }

        // Observation bookkeeping, then drop unconfirmed instances that keep being looked at without change.
        for j in instances.indices where isObserved(instances[j].centroid, in: f) {
            instances[j].observed += 1
            if !touched.contains(instances[j].id) { instances[j].misses += 1 }
        }
        let before = instances.count
        instances.removeAll {
            !$0.isConfirmed(config) && $0.observed >= config.dropAfterObserved && $0.misses >= config.dropMinMisses
        }
        stats.dropped += before - instances.count

        stats.milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
        return stats
    }

    public func setFeature(instance id: Int, _ feature: [Float], source: FeatureSource) {
        guard let j = instances.firstIndex(where: { $0.id == id }) else { return }
        instances[j].feature = feature; instances[j].featureSource = source
    }

    public func setLabel(instance id: Int, _ label: String, confidence: Float) {
        guard let j = instances.firstIndex(where: { $0.id == id }) else { return }
        instances[j].label = label; instances[j].labelConfidence = confidence
    }

    // MARK: - Internals

    /// |a ∩ b| / min(|a|, |b|).
    private func overlap(_ a: [VoxelKey: Int], _ b: [VoxelKey: Int]) -> Float {
        let (small, large) = a.count <= b.count ? (a, b) : (b, a)
        guard !small.isEmpty else { return 0 }
        return Float(small.keys.reduce(0) { large[$1] != nil ? $0 + 1 : $0 }) / Float(small.count)
    }

    /// Pixel-weighted mean of two descriptors; either side may be empty.
    static func blend(_ a: [Float], _ wa: Int, _ b: [Float], _ wb: Int) -> [Float] {
        if a.isEmpty || wa == 0 { return b }
        if b.isEmpty || wb == 0 { return a }
        let s = 1 / Float(wa + wb)
        return zip(a, b).map { (Float(wa) * $0.0 + Float(wb) * $0.1) * s }
    }

    /// `rect` grown by 20 % of its size on every side, clamped to the image.
    static func crop(_ image: FloatImage, _ rect: PixelRect) -> FloatImage {
        let mx = Int((Float(rect.maxX - rect.minX + 1) * 0.2).rounded())
        let my = Int((Float(rect.maxY - rect.minY + 1) * 0.2).rounded())
        let x0 = max(0, rect.minX - mx), x1 = min(image.width - 1, rect.maxX + mx)
        let y0 = max(0, rect.minY - my), y1 = min(image.height - 1, rect.maxY + my)
        var out = FloatImage(width: x1 - x0 + 1, height: y1 - y0 + 1, channels: image.channels)
        for c in 0..<image.channels { for y in y0...y1 { for x in x0...x1 { out[c, y - y0, x - x0] = image[c, y, x] } } }
        return out
    }

    /// The point lies in front of the camera and projects onto a `valid` pixel of this frame.
    private func isObserved(_ p: SIMD3<Float>, in f: FrameContext) -> Bool {
        let pc = f.pose.apply(p)
        guard pc.z > 0 else { return false }
        let uv = f.intrinsics.project(pc).rounded(.toNearestOrAwayFromZero)
        guard uv.x >= 0, uv.y >= 0, uv.x < Float(f.intrinsics.width), uv.y < Float(f.intrinsics.height) else { return false }
        return f.valid[Int(uv.y) * f.intrinsics.width + Int(uv.x)]
    }
}

extension TrackedInstance {
    /// One component's voxels, points and descriptors; it becomes the best view when it has the most pixels yet.
    mutating func add(_ c: Component2D, cells: [VoxelKey: Int], points newPoints: [SIMD3<Float>], feature f: [Float],
                      histogram h: [Float], frame: FrameContext, config: TrackerConfig, rng: inout SeededSampler) {
        for (k, n) in cells { voxels[k, default: 0] += n }
        frames.insert(frame.frameID)
        firstSeen = min(firstSeen, frame.frameID); lastSeen = max(lastSeen, frame.frameID)
        for p in newPoints { offer(p, rng: &rng, reservoir: config.reservoir) }
        let n = c.pixels.count
        if featureSource != .beforePhoto { feature = ObjectTracker.blend(feature, descriptorPixels, f, n) }
        histogram = ObjectTracker.blend(histogram, descriptorPixels, h, n)
        descriptorPixels += n
        if n > bestPixelCount {
            bestPixelCount = n; bestFrame = frame.frameID; bestRect = c.rect
            bestCrop = ObjectTracker.crop(c.sign >= 0 ? frame.live : frame.render, c.rect)
        }
        refreshGeometry(voxelSize: config.voxelSize)
    }

    /// Reservoir sampling (algorithm R) of the instance's world points.
    mutating func offer(_ p: SIMD3<Float>, rng: inout SeededSampler, reservoir: Int) {
        pointsSeen += 1
        if points.count < reservoir { points.append(p); return }
        let k = Int(rng.next() % UInt64(pointsSeen))
        if k < reservoir { points[k] = p }
    }

    /// Folds `other` into this instance (the caller refreshes the cached geometry). Without `descriptors`, only geometry, frames and counters move
    /// (a flat instance absorbed into a signed one must not mix the live look into a `left` feature).
    mutating func absorb(_ other: TrackedInstance, descriptors: Bool, rng: inout SeededSampler, reservoir: Int) {
        for (k, n) in other.voxels { voxels[k, default: 0] += n }
        frames.formUnion(other.frames)
        firstSeen = min(firstSeen, other.firstSeen); lastSeen = max(lastSeen, other.lastSeen)
        observed = max(observed, other.observed); misses = min(misses, other.misses)
        // Merged reservoir: both samples together, thinned at random to the capacity.
        points += other.points
        pointsSeen += other.pointsSeen
        while points.count > reservoir { points.swapAt(Int(rng.next() % UInt64(points.count)), points.count - 1); points.removeLast() }
        if descriptors {
            if featureSource != .beforePhoto { feature = ObjectTracker.blend(feature, descriptorPixels, other.feature, other.descriptorPixels) }
            histogram = ObjectTracker.blend(histogram, descriptorPixels, other.histogram, other.descriptorPixels)
            descriptorPixels += other.descriptorPixels
            if other.bestPixelCount > bestPixelCount {
                bestPixelCount = other.bestPixelCount; bestFrame = other.bestFrame
                bestRect = other.bestRect; bestCrop = other.bestCrop
            }
            if label == nil { label = other.label; labelConfidence = other.labelConfidence }
        }
    }
}
