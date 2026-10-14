import CoreML

public enum ModelError: Error, Equatable {
    case missingOutput(String)
    case unsupportedDataType
}

public extension MLMultiArray {
    /// Dense row-major copy of a float32 or float16 array, honouring the array's strides.
    func denseFloats() -> [Float] {
        let shape = self.shape.map(\.intValue), strides = self.strides.map(\.intValue)
        let n = shape.reduce(1, *)
        var contiguous = true
        var expected = 1
        for d in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[d] != expected { contiguous = false }
            expected *= shape[d]
        }
        var out = [Float](repeating: 0, count: n)
        let isF32 = dataType == .float32
        withUnsafeBytes { raw in
            if contiguous && isF32 {
                out.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[0..<(n * 4)])) }
                return
            }
            var idx = [Int](repeating: 0, count: shape.count)
            for lin in 0..<n {
                var off = 0
                for d in 0..<shape.count { off += idx[d] * strides[d] }
                if isF32 {
                    out[lin] = raw.load(fromByteOffset: off * 4, as: Float.self)
                } else {
                    #if arch(arm64)
                    out[lin] = Float(raw.load(fromByteOffset: off * 2, as: Float16.self))
                    #endif
                }
                var d = shape.count - 1
                while d >= 0 {
                    idx[d] += 1
                    if idx[d] < shape[d] { break }
                    idx[d] = 0
                    d -= 1
                }
            }
        }
        return out
    }
}
