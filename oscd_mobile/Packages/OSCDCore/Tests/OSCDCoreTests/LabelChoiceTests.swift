import XCTest
@testable import OSCDCore

final class LabelChoiceTests: XCTestCase {
    func pick(_ labels: [(String, Float)], min: Float = 0) -> String? { LabelChoice.pick(labels, minConfidence: min)?.0 }

    func testPrefersASpecificLabelTiedWithAGenericTop() {
        // Vision's ancestors first: the ball is within 0.02 of "recreation".
        let r = LabelChoice.pick([("recreation", 0.81), ("sport", 0.80), ("ball", 0.795), ("tennis_ball", 0.70)])
        XCTAssertEqual(r?.0, "ball")
        XCTAssertEqual(r?.1, 0.795)
    }

    func testKeepsVisionsOrderAmongTiedSpecificLabels() {
        XCTAssertEqual(pick([("sport", 0.80), ("tennis_ball", 0.79), ("ball", 0.795)]), "tennis_ball")
    }

    func testOutsideTheTieTheTopStays() {
        XCTAssertEqual(pick([("recreation", 0.81), ("ball", 0.78)]), "recreation")
        XCTAssertEqual(pick([("recreation", 0.81), ("ball", 0.79)]), "ball")          // exactly 0.02 below: tied
    }

    func testAllGenericKeepsTheTop() {
        XCTAssertEqual(pick([("sport", 0.80), ("recreation", 0.81), ("structure", 0.805)]), "recreation")
        XCTAssertEqual(pick([("textile", 0.5), ("clothing", 0.5)]), "textile")
    }

    func testTopNotFirstInTheList() {
        XCTAssertEqual(pick([("ball", 0.4), ("sport", 0.9), ("tennis_ball", 0.89)]), "tennis_ball")
    }

    func testMinimumConfidenceAndEdgeCases() {
        XCTAssertNil(LabelChoice.pick([]))
        XCTAssertNil(pick([("ball", 0.2)], min: 0.3))
        // A specific label tied with the top but under the minimum is not a candidate.
        XCTAssertEqual(pick([("recreation", 0.31), ("ball", 0.295)], min: 0.3), "recreation")
        XCTAssertEqual(pick([("recreation", .nan), ("ball", 0.5)]), "ball")
        XCTAssertTrue(LabelChoice.isGeneric("Sports_Equipment"))
        XCTAssertFalse(LabelChoice.isGeneric("sunglasses,shades"))
        XCTAssertTrue(LabelChoice.isGeneric("container,box"))
    }
}
