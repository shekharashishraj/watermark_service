import Foundation

/// Per-stage wall times (ms) collected over a detection run, in the order stages first ran.
public struct StageTimings: Codable, Hashable, Sendable {
    public private(set) var stages: [String] = []
    private var samples: [String: [Double]] = [:]

    public init() {}

    public mutating func add(_ stage: String, _ ms: Double) {
        if samples[stage] == nil { stages.append(stage) }
        samples[stage, default: []].append(ms)
    }

    public func count(_ stage: String) -> Int { samples[stage]?.count ?? 0 }

    public func total(_ stage: String) -> Double { samples[stage]?.reduce(0, +) ?? 0 }

    public func median(_ stage: String) -> Double? {
        guard let v = samples[stage]?.sorted(), !v.isEmpty else { return nil }
        return v.count % 2 == 1 ? v[v.count / 2] : (v[v.count / 2 - 1] + v[v.count / 2]) / 2
    }
}
