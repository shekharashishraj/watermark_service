/// Dense Gaussian elimination with partial pivoting for small systems. Returns NaNs when singular.
public enum LinearSolve {
    public static func solve(_ A0: [Double], _ b0: [Double], n: Int) -> [Double] {
        var A = A0, b = b0
        for col in 0..<n {
            var piv = col
            for r in (col + 1)..<n where abs(A[r * n + col]) > abs(A[piv * n + col]) { piv = r }
            if !(abs(A[piv * n + col]) > 1e-300) { return [Double](repeating: .nan, count: n) }
            if piv != col {
                for c in 0..<n { A.swapAt(piv * n + c, col * n + c) }
                b.swapAt(piv, col)
            }
            for r in (col + 1)..<n {
                let f = A[r * n + col] / A[col * n + col]
                if f == 0 { continue }
                for c in col..<n { A[r * n + c] -= f * A[col * n + c] }
                b[r] -= f * b[col]
            }
        }
        var x = [Double](repeating: 0, count: n)
        for r in stride(from: n - 1, through: 0, by: -1) {
            var s = b[r]
            for c in (r + 1)..<n { s -= A[r * n + c] * x[c] }
            x[r] = s / A[r * n + r]
        }
        return x
    }
}
