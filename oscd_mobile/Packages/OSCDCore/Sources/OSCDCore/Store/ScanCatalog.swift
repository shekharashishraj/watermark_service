import Foundation

/// One recorded scan as the scans list shows it.
public struct ScanInfo: Identifiable, Hashable, Sendable {
    public let id: String                 // folder name under sessions/
    public let url: URL
    public let label: String
    public let createdAt: Date
    public let frameCount: Int
    public let durationSeconds: Double
    public let deviceModel: String
    /// The scan's trained 3D model (reference.ply, built on the Mac for now) is on the phone.
    public let hasReference: Bool
    public let latestDetection: DetectionSummary?
}

public enum ScanCatalog {
    public static let referenceFile = "reference.ply"

    /// Every session folder in `dir`, oldest first, labelled "Scan N" unless the scan has a name.
    public static func list(_ dir: URL) throws -> [ScanInfo] {
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir.path) else { return [] }
        let folders = try fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        let scans = folders.compactMap { url -> (URL, SessionManifest)? in
            guard let store = try? SessionStore(open: url) else { return nil }
            return (url, store.manifest)
        }.sorted { $0.1.createdAt != $1.1.createdAt ? $0.1.createdAt < $1.1.createdAt : $0.0.lastPathComponent < $1.0.lastPathComponent }
        return scans.enumerated().map { i, scan in
            let (url, m) = scan
            let times = m.keyframes.map(\.timestamp)
            return ScanInfo(id: url.lastPathComponent, url: url, label: m.name ?? "Scan \(i + 1)", createdAt: m.createdAt,
                            frameCount: m.keyframes.count, durationSeconds: (times.max() ?? 0) - (times.min() ?? 0),
                            deviceModel: m.deviceModel,
                            hasReference: fm.fileExists(atPath: url.appendingPathComponent(referenceFile).path),
                            latestDetection: (try? DetectionSummary.all(in: url))?.max { $0.finishedAt < $1.finishedAt })
        }
    }
}

/// The outcome of one Detect run, saved in the after-scan at detections/<before-scan id>/summary.json
/// next to that run's mask images (a new run against the same before-scan replaces it).
public struct DetectionSummary: Codable, Hashable, Sendable {
    public var beforeID: String
    public var afterID: String
    public var method: String
    public var refined: Bool
    public var frames: Int
    public var framesWithPose: Int
    public var judgedPixels: Int
    public var changedPixels: Int
    public var totalSeconds: Double
    public var timings: StageTimings
    public var finishedAt = Date()

    public init(beforeID: String, afterID: String, method: String, refined: Bool, frames: Int, framesWithPose: Int,
                judgedPixels: Int, changedPixels: Int, totalSeconds: Double, timings: StageTimings) {
        self.beforeID = beforeID; self.afterID = afterID; self.method = method; self.refined = refined
        self.frames = frames; self.framesWithPose = framesWithPose; self.judgedPixels = judgedPixels
        self.changedPixels = changedPixels; self.totalSeconds = totalSeconds; self.timings = timings
    }

    /// Changed share of the judged table area over all frames with a pose.
    public var changedFraction: Double { judgedPixels == 0 ? 0 : Double(changedPixels) / Double(judgedPixels) }

    public static func folder(in afterScan: URL, beforeID: String) -> URL {
        afterScan.appendingPathComponent("detections").appendingPathComponent(beforeID)
    }

    public func save(in afterScan: URL) throws {
        let dir = Self.folder(in: afterScan, beforeID: beforeID)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try SessionStore.encoder.encode(self).write(to: dir.appendingPathComponent("summary.json"), options: .atomic)
    }

    public static func all(in afterScan: URL) throws -> [DetectionSummary] {
        let root = afterScan.appendingPathComponent("detections")
        guard FileManager.default.fileExists(atPath: root.path) else { return [] }
        return try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
            .compactMap { try? SessionStore.decoder.decode(DetectionSummary.self, from: Data(contentsOf: $0.appendingPathComponent("summary.json"))) }
    }
}
