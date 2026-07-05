import XCTest
import UIKit
@testable import TableDetectorCoreML

/// End-to-end pipeline test: loads the bundled `input1.jpeg` reference image (the same file used
/// for cross-validation against the Python `conversion_to_ort/convert_to_coreml.py` reference
/// script), runs it through preprocessing + Core ML inference + postprocessing, and checks
/// the result matches the expected "table detected, high confidence" outcome.
///
/// This test runs hosted inside the TableDetectorCoreML app (see TEST_HOST implied by the
/// target dependency in project.yml), so `Bundle.main` resolves to the app bundle and the
/// bundled, Xcode-compiled Core ML model can be loaded normally.
final class TableDetectorTests: XCTestCase {

    func testDetectsTableInReferenceImage() throws {
        let image = try loadReferenceImage()

        let detector = try TableDetector()
        let result = try detector.detect(image: image)
        print("TableDetector result: containsTable=\(result.containsTable) label=\(result.label) confidence=\(result.confidence)")

        XCTAssertTrue(result.containsTable, "Expected input1.jpeg to be detected as containing a table")
        // Python reference (convert_to_coreml.py detect) reports score ~0.972 for this image;
        // slightly lower than the ONNX/.ort model's ~0.99 because this Core ML export letterboxes
        // (stretches) the image to a fixed 800x1000 instead of resizing with aspect ratio preserved.
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
