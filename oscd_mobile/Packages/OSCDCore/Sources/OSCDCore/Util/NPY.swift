import Foundation

public enum NPYError: Error, Equatable {
    case badMagic
    case badHeader(String)
    case unsupportedDType(String)
    case fortranOrder
    case truncated
}

/// Minimal NumPy .npy (v1–v3) reader/writer for little-endian C-order arrays.
public struct NPYArray: Sendable {
    public enum DType: String, Sendable {
        case f4 = "<f4", f8 = "<f8", i4 = "<i4", i8 = "<i8", u1 = "|u1", u2 = "<u2", b1 = "|b1"
        var size: Int {
            switch self {
            case .f4, .i4: return 4
            case .f8, .i8: return 8
            case .u2: return 2
            case .u1, .b1: return 1
            }
        }
    }

    public let shape: [Int]
    public let dtype: DType
    public let payload: Data

    public var count: Int { shape.reduce(1, *) }

    public init(shape: [Int], dtype: DType, payload: Data) {
        precondition(payload.count == shape.reduce(1, *) * dtype.size, "payload size does not match shape")
        self.shape = shape
        self.dtype = dtype
        self.payload = payload
    }

    public init(floats: [Float], shape: [Int]) {
        self.init(shape: shape, dtype: .f4, payload: floats.withUnsafeBufferPointer { Data(buffer: $0) })
    }
    public init(uint8: [UInt8], shape: [Int]) { self.init(shape: shape, dtype: .u1, payload: Data(uint8)) }
    public init(uint16: [UInt16], shape: [Int]) {
        self.init(shape: shape, dtype: .u2, payload: uint16.withUnsafeBufferPointer { Data(buffer: $0) })
    }
    public init(int64: [Int64], shape: [Int]) {
        self.init(shape: shape, dtype: .i8, payload: int64.withUnsafeBufferPointer { Data(buffer: $0) })
    }
    public init(bools: [Bool], shape: [Int]) {
        self.init(shape: shape, dtype: .b1, payload: Data(bools.map { $0 ? 1 : 0 }))
    }

    public static func load(_ url: URL) throws -> NPYArray { try parse(Data(contentsOf: url)) }

    public static func parse(_ bytes: Data) throws -> NPYArray {
        let b = [UInt8](bytes.prefix(12))
        guard b.count >= 10, b[0] == 0x93, Array(b[1...5]) == Array("NUMPY".utf8) else { throw NPYError.badMagic }
        let headerLen: Int, headerStart: Int
        if b[6] == 1 {
            headerLen = Int(b[8]) | Int(b[9]) << 8
            headerStart = 10
        } else {
            guard b.count >= 12 else { throw NPYError.truncated }
            headerLen = Int(b[8]) | Int(b[9]) << 8 | Int(b[10]) << 16 | Int(b[11]) << 24
            headerStart = 12
        }
        guard bytes.count >= headerStart + headerLen else { throw NPYError.truncated }
        let base = bytes.startIndex
        let headerData = bytes.subdata(in: base + headerStart ..< base + headerStart + headerLen)
        guard let header = String(data: headerData, encoding: .ascii) else { throw NPYError.badHeader("not ASCII") }
        guard let descr = capture(header, #"'descr':\s*'([^']+)'"#) else { throw NPYError.badHeader(header) }
        guard let dtype = DType(rawValue: descr) else { throw NPYError.unsupportedDType(descr) }
        if capture(header, #"'fortran_order':\s*(True|False)"#) == "True" { throw NPYError.fortranOrder }
        guard let shapeString = capture(header, #"'shape':\s*\(([^)]*)\)"#) else { throw NPYError.badHeader(header) }
        let shape = shapeString.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        let start = base + headerStart + headerLen
        let n = shape.reduce(1, *) * dtype.size
        guard bytes.endIndex - start >= n else { throw NPYError.truncated }
        return NPYArray(shape: shape, dtype: dtype, payload: bytes.subdata(in: start ..< start + n))
    }

    private static func capture(_ s: String, _ pattern: String) -> String? {
        guard let re = try? NSRegularExpression(pattern: pattern),
              let m = re.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)),
              let r = Range(m.range(at: 1), in: s) else { return nil }
        return String(s[r])
    }

    private func raw<T>(_: T.Type) -> [T] {
        [T](unsafeUninitializedCapacity: count) { buffer, initialized in
            _ = payload.copyBytes(to: buffer)
            initialized = count
        }
    }

    public func floats() -> [Float] {
        switch dtype {
        case .f4: return raw(Float.self)
        case .f8: return raw(Double.self).map(Float.init)
        case .i4: return raw(Int32.self).map(Float.init)
        case .i8: return raw(Int64.self).map(Float.init)
        case .u2: return raw(UInt16.self).map(Float.init)
        case .u1, .b1: return raw(UInt8.self).map(Float.init)
        }
    }

    public func ints() -> [Int] {
        switch dtype {
        case .i8: return raw(Int64.self).map(Int.init)
        case .i4: return raw(Int32.self).map(Int.init)
        case .u2: return raw(UInt16.self).map(Int.init)
        case .u1, .b1: return raw(UInt8.self).map(Int.init)
        case .f4, .f8: return floats().map { Int($0) }
        }
    }

    public func bools() -> [Bool] {
        switch dtype {
        case .b1, .u1: return raw(UInt8.self).map { $0 != 0 }
        default: return floats().map { $0 != 0 }
        }
    }

    public func uint8() -> [UInt8] {
        precondition(dtype == .u1 || dtype == .b1, "uint8() needs a u1/b1 array")
        return raw(UInt8.self)
    }

    public func encoded() -> Data {
        let shapeString = shape.count == 1 ? "(\(shape[0]),)" : "(" + shape.map(String.init).joined(separator: ", ") + ")"
        var header = "{'descr': '\(dtype.rawValue)', 'fortran_order': False, 'shape': \(shapeString), }"
        let unpadded = 10 + header.utf8.count + 1
        header += String(repeating: " ", count: (64 - unpadded % 64) % 64) + "\n"
        var out = Data([0x93]) + Data("NUMPY".utf8) + Data([1, 0])
        let len = UInt16(header.utf8.count)
        out.append(contentsOf: [UInt8(len & 0xff), UInt8(len >> 8)])
        out.append(Data(header.utf8))
        out.append(payload)
        return out
    }

    public func write(to url: URL) throws { try encoded().write(to: url, options: .atomic) }
}
