import Foundation
import simd

/// Where an instance sits: voxel-hit-weighted centroid, world box, voxel count and the frames that saw it.
public struct Placement: Codable, Sendable {
    public var centroid: SIMD3<Float>; public var bbox: AABB; public var voxels: Int; public var frames: [Int]
    public init(centroid: SIMD3<Float>, bbox: AABB, voxels: Int, frames: [Int]) {
        self.centroid = centroid; self.bbox = bbox; self.voxels = voxels; self.frames = frames
    }
}

/// A measured move from the planar ICP (spec §4.7). `yawDeg` is nil when the rotation could not be told apart.
public struct Motion: Codable, Sendable {
    public var translation: SIMD3<Float>; public var horizontalDistance: Float; public var yawDeg: Float?
    public var fitRMS: Float; public var inlierFraction: Float
    public init(translation: SIMD3<Float>, horizontalDistance: Float, yawDeg: Float?, fitRMS: Float, inlierFraction: Float) {
        self.translation = translation; self.horizontalDistance = horizontalDistance; self.yawDeg = yawDeg
        self.fitRMS = fitRMS; self.inlierFraction = inlierFraction
    }
}

public struct ObjectEvent: Codable, Identifiable, Sendable {
    public enum Kind: String, Codable, Sendable { case moved, added, removed, altered }
    /// The left id for moved/removed, the arrived id for added, the instance id for altered.
    public var id: Int
    public var kind: Kind; public var confidence: Float; public var hint: Bool      // hint is true only for altered
    public var label: String?; public var labelConfidence: Float?
    public var before: Placement?; public var after: Placement?; public var motion: Motion?

    public init(id: Int, kind: Kind, confidence: Float, hint: Bool, label: String? = nil, labelConfidence: Float? = nil,
                before: Placement? = nil, after: Placement? = nil, motion: Motion? = nil) {
        self.id = id; self.kind = kind; self.confidence = confidence; self.hint = hint
        self.label = label; self.labelConfidence = labelConfidence
        self.before = before; self.after = after; self.motion = motion
    }

    /// "Ball · moved 14 cm · turned 30°"; the name is "Object N" without a label, with "?" when the label is unsure.
    public func summary(matchAccept: Float = 0.55) -> String {
        var name = label ?? "Object \(id)"
        if label != nil, let c = labelConfidence, c < 0.6 { name += "?" }
        switch kind {
        case .added: return "\(name) · added"
        case .removed: return "\(name) · removed"
        case .altered: return "\(name) · look changed here?"
        case .moved:
            var s = "\(name) · " + (confidence < matchAccept ? "probably moved" : "moved")
            guard let m = motion else { return s }
            s += " \(Int((m.horizontalDistance * 100).rounded())) cm"
            if let yaw = m.yawDeg { s += " · turned \(Int(yaw.rounded()))°" } else { s += " · rotation unknown" }
            return s
        }
    }
}

public struct ObjectReport: Codable, Sendable {
    public var beforeID: String; public var afterID: String?; public var method: String
    public var worldFrame = "before-scan ARKit world, +y up"; public var voxelSizeM: Float
    public var events: [ObjectEvent]; public var generatedAt: Date; public var timingsMs: [String: Double]
    public static let fileName = "objects.json"

    public init(beforeID: String, afterID: String?, method: String, voxelSizeM: Float, events: [ObjectEvent],
                generatedAt: Date = Date(), timingsMs: [String: Double] = [:]) {
        self.beforeID = beforeID; self.afterID = afterID; self.method = method; self.voxelSizeM = voxelSizeM
        self.events = events; self.generatedAt = generatedAt; self.timingsMs = timingsMs
    }

    /// Writes `objects.json` next to the detection's `summary.json`.
    public func save(in afterScan: URL) throws {
        let dir = DetectionSummary.folder(in: afterScan, beforeID: beforeID, method: method)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try SessionStore.encoder.encode(self).write(to: dir.appendingPathComponent(Self.fileName), options: .atomic)
    }

    /// nil when the file is missing or cannot be decoded.
    public static func load(in afterScan: URL, beforeID: String, method: String) -> ObjectReport? {
        let url = DetectionSummary.folder(in: afterScan, beforeID: beforeID, method: method).appendingPathComponent(fileName)
        return (try? Data(contentsOf: url)).flatMap { try? SessionStore.decoder.decode(ObjectReport.self, from: $0) }
    }
}

public struct Match: Equatable, Sendable {
    public var leftID: Int; public var arrivedID: Int; public var score: Float
    public init(leftID: Int, arrivedID: Int, score: Float) { self.leftID = leftID; self.arrivedID = arrivedID; self.score = score }
}

public enum InstanceMatcher {
    /// 0.5·cos(feature) + 0.3·histogram intersection + 0.2·size similarity (spec §4.6).
    public static func score(_ l: TrackedInstance, _ a: TrackedInstance) -> Float {
        0.5 * cosine(l.feature, a.feature) + 0.3 * LabHistogram.intersection(l.histogram, a.histogram)
            + 0.2 * sizeSimilarity(l.voxelCount, a.voxelCount)
    }

    /// 0 for empty or mismatched vectors.
    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        guard !a.isEmpty, a.count == b.count else { return 0 }
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in a.indices { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        let d = (na * nb).squareRoot()
        return d > 0 ? dot / d : 0
    }

    static func sizeSimilarity(_ a: Int, _ b: Int) -> Float {
        guard a > 0, b > 0 else { return 0 }
        return min(1, max(0, 1 - abs(log(Float(a) / Float(b))) / log(3)))
    }
}

struct PairKey: Hashable { var leftID: Int; var arrivedID: Int }

/// A measured move with the voxel counts of both instances when it was measured.
struct CachedMotion { var motion: Motion?; var leftVoxels: Int; var arrivedVoxels: Int }

extension ObjectTracker {
    /// Confirmed left × arrived instances paired by Hungarian assignment on the full score matrix; pairs under `matchLow` are dropped.
    public func matches() -> [Match] {
        let lefts = instances.filter { $0.sign < 0 && $0.isConfirmed(config) }
        let arrived = instances.filter { $0.sign > 0 && $0.isConfirmed(config) }
        let score = lefts.map { l in arrived.map { InstanceMatcher.score(l, $0) } }
        var out: [Match] = []
        for (i, j) in Hungarian.assign(score: score).enumerated() {
            guard let j, score[i][j] >= config.matchLow else { continue }
            out.append(Match(leftID: lefts[i].id, arrivedID: arrived[j].id, score: score[i][j]))
        }
        return out
    }

    /// Caches the measured move of a pair (nil records a failed measurement so it is not retried). The cache is
    /// dropped once either instance has grown by more than 20 % in voxels.
    public func setMotion(leftID: Int, arrivedID: Int, _ motion: Motion?) {
        guard let l = instance(leftID), let a = instance(arrivedID) else { return }
        motionCache[PairKey(leftID: leftID, arrivedID: arrivedID)] = CachedMotion(motion: motion, leftVoxels: l.voxelCount, arrivedVoxels: a.voxelCount)
    }

    /// Matched pairs without a usable cached motion.
    public func pairsNeedingMotion() -> [Match] {
        matches().filter { cached($0) == nil }
    }

    /// The ICP inputs of a pair: the left instance's points (source) and the arrived instance's (target); nil when
    /// either instance is gone.
    public func motionInputs(_ m: Match) -> (source: [SIMD3<Float>], target: [SIMD3<Float>])? {
        guard let l = instance(m.leftID), let a = instance(m.arrivedID) else { return nil }
        return (l.points, a.points)
    }

    /// The move from one point set to the other, as every caller of the layer measures it (spec §4.7).
    public static func measureMotion(source: [SIMD3<Float>], target: [SIMD3<Float>]) -> Motion? {
        PlanarICP.fit(source: source, target: target)
    }

    /// Measures every pair in `pairsNeedingMotion()` in turn and caches the result with `setMotion`, so Detect and the
    /// Mac replay fit the same pairs the same way. `willFit` runs before each fit; `stop` is checked before each fit.
    /// Returns the number of fits.
    @discardableResult
    public func fitMotions(willFit: (Match) -> Void = { _ in }, stop: () -> Bool = { false }) -> Int {
        var fits = 0
        for pair in pairsNeedingMotion() {
            if stop() { break }
            guard let (source, target) = motionInputs(pair) else { continue }
            willFit(pair)
            setMotion(leftID: pair.leftID, arrivedID: pair.arrivedID, Self.measureMotion(source: source, target: target))
            fits += 1
        }
        return fits
    }

    /// One event per confirmed instance or matched pair, ordered by when the instance named by `id` first appeared.
    public func events() -> [ObjectEvent] {
        let confirmed = instances.filter { $0.isConfirmed(config) }
        var events: [(first: Int, event: ObjectEvent)] = []
        var paired = Set<Int>()
        for m in matches() {
            guard let l = instance(m.leftID), let a = instance(m.arrivedID) else { continue }
            paired.insert(l.id); paired.insert(a.id)
            let named = (a.labelConfidence ?? -1) > (l.labelConfidence ?? -1) ? a : l
            events.append((l.firstSeen, ObjectEvent(
                id: l.id, kind: .moved, confidence: m.score, hint: false, label: named.label, labelConfidence: named.labelConfidence,
                before: Self.placement(l), after: Self.placement(a), motion: cached(m)?.motion)))
        }
        for i in confirmed where !paired.contains(i.id) {
            let kind: ObjectEvent.Kind = i.sign < 0 ? .removed : (i.sign > 0 ? .added : .altered)
            let support = min(1, Float(i.frames.count) / 6)
            events.append((i.firstSeen, ObjectEvent(
                id: i.id, kind: kind, confidence: kind == .altered ? 0 : support, hint: kind == .altered,
                label: i.label, labelConfidence: i.labelConfidence,
                before: i.sign < 0 ? Self.placement(i) : nil, after: i.sign >= 0 ? Self.placement(i) : nil)))
        }
        return events.sorted { ($0.first, $0.event.id) < ($1.first, $1.event.id) }.map(\.event)
    }

    public func report(beforeID: String, afterID: String?, method: String, timingsMs: [String: Double] = [:]) -> ObjectReport {
        ObjectReport(beforeID: beforeID, afterID: afterID, method: method, voxelSizeM: config.voxelSize,
                     events: events(), generatedAt: Date(), timingsMs: timingsMs)
    }

    // MARK: - Internals

    private func instance(_ id: Int) -> TrackedInstance? { instances.first { $0.id == id } }

    /// The cached motion entry, or nil when none was set or either instance has since grown by more than 20 %.
    private func cached(_ m: Match) -> CachedMotion? {
        guard let c = motionCache[PairKey(leftID: m.leftID, arrivedID: m.arrivedID)],
              let l = instance(m.leftID), let a = instance(m.arrivedID),
              Float(l.voxelCount) <= 1.2 * Float(c.leftVoxels), Float(a.voxelCount) <= 1.2 * Float(c.arrivedVoxels)
        else { return nil }
        return c
    }

    private static func placement(_ i: TrackedInstance) -> Placement {
        Placement(centroid: i.centroid, bbox: i.bbox, voxels: i.voxelCount, frames: i.frames.sorted())
    }
}
