import XCTest
import UIKit
@testable import TableDetector

/// End-to-end pipeline test: loads the bundled `input1.jpeg` reference image (the same file used
/// for cross-validation against the Python `conversion_to_ort/ort_conversion.py` reference
/// script), runs it through preprocessing + ONNX Runtime inference + postprocessing, and checks
/// the result matches the expected "table detected, high confidence" outcome.
///
/// This test runs hosted inside the TableDetector app (see TEST_HOST in project.yml), so
/// `Bundle.main` resolves to the app bundle and the bundled .ort model can be loaded normally.
final class TableDetectorTests: XCTestCase {

    func testDetectsTableInReferenceImage() throws {
        let image = try loadReferenceImage()

        let detector = try TableDetector()
        let result = try detector.detect(image: image)
        print("TableDetector result: containsTable=\(result.containsTable) label=\(result.label) confidence=\(result.confidence)")

        XCTAssertTrue(result.containsTable, "Expected input1.jpeg to be detected as containing a table")
        // Python reference (ort_conversion.py detect_tables) reports score ~0.993 for this image.
        XCTAssertGreaterThan(result.confidence, 0.9, "Expected high confidence for a clear, undistorted table scan")
        XCTAssertLessThanOrEqual(result.confidence, 1.0)
    }

    private func loadReferenceImage() throws -> UIImage {
        let bundle = Bundle(for: Self.self)
        guard let url = bundle.url(forResource: "input1", withExtension: "jpeg") else {
            XCTFail("input1.jpeg not found in test bundle resources")
            throw XCTSkip("Missing test fixture")
        }
        let data = try Data(contentsOf: url)
        guard let image = UIImage(data: data) else {
            XCTFail("Could not decode input1.jpeg")
            throw XCTSkip("Corrupt test fixture")
        }
        return image
    }
}
