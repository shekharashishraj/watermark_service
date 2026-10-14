import Foundation

/// Draws m distinct indices from 0..<n (one RANSAC hypothesis or one subset).
public protocol IndexSampler {
    mutating func sample(_ m: Int, from n: Int) -> [Int]
}

/// SplitMix64 generator; deterministic for a seed.
public struct SeededSampler: IndexSampler, Sendable {
    private var state: UInt64

    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    public mutating func uniform() -> Double { Double(next() >> 11) * 0x1.0p-53 }

    public mutating func sample(_ m: Int, from n: Int) -> [Int] {
        precondition(m <= n, "cannot sample \(m) of \(n)")
        if m > 16 { return subset(m, of: n) }
        var picked: [Int] = []
        picked.reserveCapacity(m)
        while picked.count < m {
            let k = Int(next() % UInt64(n))
            if !picked.contains(k) { picked.append(k) }
        }
        return picked
    }

    /// m distinct indices from 0..<n by a partial Fisher–Yates shuffle.
    public mutating func subset(_ m: Int, of n: Int) -> [Int] {
        var a = Array(0..<n)
        for i in 0..<m {
            let j = i + Int(next() % UInt64(n - i))
            a.swapAt(i, j)
        }
        return Array(a.prefix(m))
    }
}

/// Replays index sets recorded from O-SCD (parity tests).
public struct InjectedSampler: IndexSampler {
    private let rows: [[Int]]
    private var position = 0

    public init(rows: [[Int]]) { self.rows = rows }

    public mutating func sample(_ m: Int, from n: Int) -> [Int] {
        defer { position += 1 }
        return Array(rows[position].prefix(m))
    }
}
