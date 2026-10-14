import XCTest
@testable import OSCDCore

final class NPYTests: XCTestCase {
    func testFloatRoundTrip() throws {
        let a = NPYArray(floats: [1, 2.5, -3, 4, 5, 6], shape: [2, 3])
        let b = try NPYArray.parse(a.encoded())
        XCTAssertEqual(b.shape, [2, 3])
        XCTAssertEqual(b.dtype, .f4)
        XCTAssertEqual(b.floats(), [1, 2.5, -3, 4, 5, 6])
    }

    func testHeaderPaddedTo64Bytes() {
        let a = NPYArray(floats: [1, 2, 3], shape: [3])
        XCTAssertEqual((a.encoded().count - 3 * 4) % 64, 0)
    }

    /// Bytes laid out the way np.save writes them: magic, v1.0, header padded with spaces so the
    /// preamble is a multiple of 64 bytes and ends in a newline, then the payload.
    func npyBytes(_ header: String, _ payload: Data) -> Data {
        let pad = (64 - (10 + header.utf8.count + 1) % 64) % 64
        let h = header + String(repeating: " ", count: pad) + "\n"
        return Data([0x93]) + Data("NUMPY".utf8) + Data([1, 0, UInt8(h.utf8.count & 0xff), UInt8(h.utf8.count >> 8)]) + Data(h.utf8) + payload
    }

    func testParsesNumpyStyleBytes() throws {
        // Exactly what np.save writes for np.array([True, False, True]).
        let bytes = npyBytes("{'descr': '|b1', 'fortran_order': False, 'shape': (3,), }", Data([1, 0, 1]))
        let a = try NPYArray.parse(bytes)
        XCTAssertEqual(a.shape, [3])
        XCTAssertEqual(a.bools(), [true, false, true])
    }

    func testIntsAndScalars() throws {
        let a = try NPYArray.parse(NPYArray(int64: [7], shape: []).encoded())
        XCTAssertEqual(a.shape, [])
        XCTAssertEqual(a.count, 1)
        XCTAssertEqual(a.ints(), [7])
    }

    func testRejectsBadMagicAndFortranOrder() {
        XCTAssertThrowsError(try NPYArray.parse(Data([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])))
        let bytes = npyBytes("{'descr': '<f4', 'fortran_order': True, 'shape': (1,), }", Data(count: 4))
        XCTAssertThrowsError(try NPYArray.parse(bytes)) { XCTAssertEqual($0 as? NPYError, .fortranOrder) }
    }

    func testFileRoundTrip() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".npy")
        try NPYArray(uint8: [1, 2, 3, 4], shape: [2, 2]).write(to: url)
        let b = try NPYArray.load(url)
        XCTAssertEqual(b.uint8(), [1, 2, 3, 4])
    }
}
