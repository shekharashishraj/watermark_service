/// IEEE 754 binary16 conversion by bit manipulation (no `Float16` type, so it also builds for Intel Macs).
/// Used for on-disk caches where half precision is enough (SAM2 embeddings).
public enum Float16Pack {
    /// binary16 bit patterns, rounded to nearest (ties to even); too-large values become ±infinity, NaN stays NaN.
    public static func pack(_ v: [Float]) -> [UInt16] {
        [UInt16](unsafeUninitializedCapacity: v.count) { out, n in
            v.withUnsafeBufferPointer { src in
                for i in 0..<src.count { out[i] = half(src[i]) }
            }
            n = v.count
        }
    }

    public static func unpack(_ v: [UInt16]) -> [Float] {
        [Float](unsafeUninitializedCapacity: v.count) { out, n in
            v.withUnsafeBufferPointer { src in
                for i in 0..<src.count { out[i] = float(src[i]) }
            }
            n = v.count
        }
    }

    @inline(__always)
    static func half(_ f: Float) -> UInt16 {
        let x = f.bitPattern
        let sign = UInt16(truncatingIfNeeded: (x >> 16) & 0x8000)
        let exp = Int((x >> 23) & 0xFF)
        let mant = x & 0x7F_FFFF
        if exp == 0xFF {                                     // infinity or NaN (kept quiet)
            return sign | 0x7C00 | (mant != 0 ? 0x200 | UInt16(truncatingIfNeeded: mant >> 13) : 0)
        }
        let e = exp - 127 + 15
        if e >= 0x1F { return sign | 0x7C00 }                // overflow
        if e <= 0 {                                          // subnormal half, or zero
            if e < -10 { return sign }                       // under half the smallest subnormal
            let m = mant | 0x80_0000
            let shift = UInt32(14 - e)
            var r = m >> shift
            let rem = m & ((1 << shift) - 1), halfway = UInt32(1) << (shift - 1)
            if rem > halfway || (rem == halfway && r & 1 == 1) { r += 1 }   // may carry into the smallest normal
            return sign | UInt16(truncatingIfNeeded: r)
        }
        var r = UInt32(e) << 10 | mant >> 13
        let rem = mant & 0x1FFF
        if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) { r += 1 }         // a carry may reach infinity, as it should
        return sign | UInt16(truncatingIfNeeded: r)
    }

    @inline(__always)
    static func float(_ h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        let e = UInt32((h >> 10) & 0x1F), m = UInt32(h & 0x3FF)
        if e == 0 {
            let magnitude = Float(m) * 0x1p-24                // zero or subnormal
            return sign != 0 ? -magnitude : magnitude
        }
        if e == 0x1F { return Float(bitPattern: sign | 0x7F80_0000 | m << 13) }
        return Float(bitPattern: sign | (e + 112) << 23 | m << 13)
    }
}
