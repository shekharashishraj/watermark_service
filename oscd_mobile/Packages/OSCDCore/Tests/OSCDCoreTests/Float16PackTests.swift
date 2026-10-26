import XCTest
@testable import OSCDCore

final class Float16PackTests: XCTestCase {
    func testFloat16RoundTrip() {
        let v: [Float] = [0, 1, -2.5, 0.1, 65504]
        let back = Float16Pack.unpack(Float16Pack.pack(v))
        XCTAssertEqual(back.count, v.count)
        for (a, b) in zip(v, back) {
            XCTAssertLessThanOrEqual(abs(a - b), 1e-3 * abs(a), "\(a) came back as \(b)")
        }
        let nan = Float16Pack.unpack(Float16Pack.pack([.nan]))
        XCTAssertTrue(nan[0].isNaN)
    }

    /// IEEE binary16 bit patterns, including rounding to nearest even, overflow, signed zero and subnormals.
    func testKnownBitPatterns() {
        XCTAssertEqual(Float16Pack.pack([1, -2.5, 65504, -0.0, .infinity, 1e6, 0.1]),
                       [0x3C00, 0xC100, 0x7BFF, 0x8000, 0x7C00, 0x7C00, 0x2E66])
        // 1 + 2^-11 lies halfway between 1 and the next half (1 + 2^-10): ties go to the even mantissa.
        XCTAssertEqual(Float16Pack.pack([1 + 0x1p-11, 1 + 3 * 0x1p-11]), [0x3C00, 0x3C02])
        // Smallest subnormal half (2^-24) and the largest subnormal.
        XCTAssertEqual(Float16Pack.pack([0x1p-24, 1023 * 0x1p-24, 0x1p-26]), [0x0001, 0x03FF, 0x0000])
        XCTAssertEqual(Float16Pack.unpack([0x0001, 0x03FF, 0x7C00, 0xFC00]), [0x1p-24, 1023 * 0x1p-24, .infinity, -.infinity])
        // 65520 is the first value that rounds up past the largest finite half.
        XCTAssertEqual(Float16Pack.pack([65519, 65520]), [0x7BFF, 0x7C00])
    }
}
