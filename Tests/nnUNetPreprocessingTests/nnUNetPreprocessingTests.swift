import XCTest
@testable import nnUNetPreprocessing

final class nnUNetPreprocessingTests: XCTestCase {
    func testVersionExists() {
        XCTAssertFalse(nnUNetPreprocessing.version.isEmpty)
    }
}
