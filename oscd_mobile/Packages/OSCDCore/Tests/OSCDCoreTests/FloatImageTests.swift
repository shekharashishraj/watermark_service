import XCTest
@testable import OSCDCore

final class FloatImageTests: XCTestCase {
    func row(_ v: [Float]) -> FloatImage { FloatImage(width: v.count, height: 1, channels: 1, data: v) }

    func testBilinearMatchesTorchDownAndUp() {
        // torch.nn.functional.interpolate(x, size=..., mode='bilinear', align_corners=False)
        XCTAssertEqual(row([0, 1, 2, 3]).resizedBilinear(width: 2, height: 1).data, [0.5, 2.5])
        XCTAssertEqual(row([0, 4]).resizedBilinear(width: 4, height: 1).data, [0, 1, 3, 4])
        let same = row([3, 1, 4, 1, 5]).resizedBilinear(width: 5, height: 1)
        XCTAssertEqual(same.data, [3, 1, 4, 1, 5])
    }

    func testNearestMatchesTorchLegacyNearest() {
        XCTAssertEqual(row([7, 9]).resizedNearest(width: 4, height: 1).data, [7, 7, 9, 9])
        XCTAssertEqual(row([1, 2, 3]).resizedNearest(width: 4, height: 1).data, [1, 1, 2, 3])
    }

    func testAreaDownsample() {
        let img = FloatImage(width: 4, height: 2, channels: 1, data: [1, 3, 5, 7, 1, 3, 5, 7])
        XCTAssertEqual(img.downsampledArea(factor: 2).data, [2, 6])
    }

    func testHWCAndRGB8Conversion() {
        let img = FloatImage(hwc: [1, 2, 3, 4, 5, 6], width: 2, height: 1, channels: 3)
        XCTAssertEqual(img.data, [1, 4, 2, 5, 3, 6])
        let rgb = FloatImage(rgb8: [255, 0, 51], width: 1, height: 1)
        XCTAssertEqual(rgb.data[0], 1); XCTAssertEqual(rgb.data[1], 0); XCTAssertEqual(rgb.data[2], 0.2, accuracy: 1e-6)
    }
}
