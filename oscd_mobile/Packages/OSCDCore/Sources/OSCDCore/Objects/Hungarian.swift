import Foundation

/// Optimal assignment on a rectangular score matrix (Kuhn–Munkres with potentials, O(n²·m)).
public enum Hungarian {
    /// Maximises Σ score. Returns, for every row, the column assigned to it, or nil when the row is left
    /// without one (more rows than columns). Empty rows or columns give all nil / [].
    public static func assign(score: [[Float]]) -> [Int?] {
        let rows = score.count
        let cols = score.first?.count ?? 0
        guard rows > 0, cols > 0 else { return [Int?](repeating: nil, count: rows) }
        if rows > cols {
            // The potentials method needs rows ≤ columns: solve the transpose and invert the result.
            let t = (0..<cols).map { c in (0..<rows).map { score[$0][c] } }
            let byCol = assign(score: t)
            var out = [Int?](repeating: nil, count: rows)
            for (c, r) in byCol.enumerated() { if let r { out[r] = c } }
            return out
        }
        // Minimise cost = −score; indices are 1-based with a virtual row/column 0.
        var u = [Double](repeating: 0, count: rows + 1), v = [Double](repeating: 0, count: cols + 1)
        var owner = [Int](repeating: 0, count: cols + 1)          // owner[j] = row assigned to column j
        var way = [Int](repeating: 0, count: cols + 1)
        for i in 1...rows {
            owner[0] = i
            var j0 = 0
            var minv = [Double](repeating: .infinity, count: cols + 1)
            var used = [Bool](repeating: false, count: cols + 1)
            repeat {
                used[j0] = true
                let i0 = owner[j0]
                var delta = Double.infinity, j1 = 0
                for j in 1...cols where !used[j] {
                    let cur = -Double(score[i0 - 1][j - 1]) - u[i0] - v[j]
                    if cur < minv[j] { minv[j] = cur; way[j] = j0 }
                    if minv[j] < delta { delta = minv[j]; j1 = j }
                }
                for j in 0...cols {
                    if used[j] { u[owner[j]] += delta; v[j] -= delta } else { minv[j] -= delta }
                }
                j0 = j1
            } while owner[j0] != 0
            repeat {
                let j1 = way[j0]
                owner[j0] = owner[j1]
                j0 = j1
            } while j0 != 0
        }
        var out = [Int?](repeating: nil, count: rows)
        for j in 1...cols where owner[j] != 0 { out[owner[j] - 1] = j - 1 }
        return out
    }
}
