import CoreML
import Foundation
import UIKit

/// Result of running the Table Transformer detection model on a single image.
struct TableDetectionResult {
    /// Whether the best query's table score exceeded `TableDetector.scoreThreshold`.
    let containsTable: Bool
    /// Best "table" (or "table rotated") score found across all decoder queries, in [0, 1].
    let confidence: Double
    /// Which of the two table labels produced the best score ("table" or "table rotated").
    let label: String
}

enum TableDetectorError: Error {
    case modelNotFoundInBundle
    case unexpectedOutputShape
}

/// Loads the bundled Table Transformer Core ML model and runs on-device table detection.
///
/// Mirrors the pre/postprocessing implemented in `conversion_to_ort/convert_to_coreml.py`
/// (`detect_tables`), including its fixed 800x1000 input size -- unlike the ONNX/.ort export,
/// this Core ML model does not accept dynamic image sizes.
final class TableDetector {

    /// Matches the Python reference script's default `--score-threshold`.
    static let scoreThreshold: Double = 0.7

    private static let numQueries = 15
    private static let numClasses = 3 // "table", "table rotated", "no object"
    private static let tableLabelIndex = 0
    private static let rotatedLabelIndex = 1

    private let model: MLModel

    init() throws {
        // Xcode compiles the bundled .mlpackage into table-transformer-detection.mlmodelc
        // as part of the build; that compiled form is what ships in the app bundle.
        guard let modelURL = Bundle.main.url(forResource: "table-transformer-detection", withExtension: "mlmodelc") else {
            throw TableDetectorError.modelNotFoundInBundle
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        model = try MLModel(contentsOf: modelURL, configuration: configuration)
    }

    /// Runs preprocessing + inference + postprocessing on `image`, synchronously.
    /// Call this off the main thread — a forward pass can take a noticeable fraction of a second.
    func detect(image: UIImage) throws -> TableDetectionResult {
        let preprocessed = try ImagePreprocessor.preprocess(image: image)

        let pixelValuesArray = try Self.makeMultiArray(
            from: preprocessed.pixelValues,
            shape: [1, 3, ImagePreprocessor.targetHeight, ImagePreprocessor.targetWidth]
        )
        let pixelMaskArray = try Self.makeMultiArray(
            from: preprocessed.pixelMask,
            shape: [1, ImagePreprocessor.targetHeight, ImagePreprocessor.targetWidth]
        )

        let inputFeatures: [String: MLFeatureValue] = [
            "pixel_values": MLFeatureValue(multiArray: pixelValuesArray),
            "pixel_mask": MLFeatureValue(multiArray: pixelMaskArray),
        ]
        let inputProvider = try MLDictionaryFeatureProvider(dictionary: inputFeatures)

        let outputProvider = try model.prediction(from: inputProvider)
        guard let logits = outputProvider.featureValue(for: "logits")?.multiArrayValue else {
            throw TableDetectorError.unexpectedOutputShape
        }

        return try Self.postprocess(logits: logits)
    }

    // MARK: - Tensor construction

    private static func makeMultiArray(from floats: [Float], shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        floats.withUnsafeBufferPointer { buffer in
            array.dataPointer.copyMemory(from: buffer.baseAddress!, byteCount: floats.count * MemoryLayout<Float>.stride)
        }
        return array
    }

    private static func makeMultiArray(from ints: [Int32], shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        ints.withUnsafeBufferPointer { buffer in
            array.dataPointer.copyMemory(from: buffer.baseAddress!, byteCount: ints.count * MemoryLayout<Int32>.stride)
        }
        return array
    }

    // MARK: - Postprocessing

    /// `logits` has shape [1, numQueries, numClasses]. For each query we softmax the 3 class
    /// logits and take the best of the two "table" classes; the overall confidence is the max
    /// of that across all queries.
    ///
    /// Reads elements through MLMultiArray's NSNumber-based subscript rather than assuming a
    /// contiguous Float32 buffer, since Core ML may produce this small output in Float16
    /// depending on which compute unit ran that part of the graph.
    private static func postprocess(logits: MLMultiArray) throws -> TableDetectionResult {
        let expectedCount = numQueries * numClasses
        guard logits.count == expectedCount else {
            throw TableDetectorError.unexpectedOutputShape
        }

        var bestScore = 0.0
        var bestLabel = "table"

        for query in 0..<numQueries {
            let base = query * numClasses
            let classLogits = (0..<numClasses).map { logits[base + $0].doubleValue }
            let probabilities = softmax(classLogits)

            let tableProb = probabilities[tableLabelIndex]
            let rotatedProb = probabilities[rotatedLabelIndex]
            let tableScore = max(tableProb, rotatedProb)

            if tableScore > bestScore {
                bestScore = tableScore
                bestLabel = rotatedProb > tableProb ? "table rotated" : "table"
            }
        }

        return TableDetectionResult(
            containsTable: bestScore > scoreThreshold,
            confidence: bestScore,
            label: bestLabel
        )
    }

    private static func softmax(_ logits: [Double]) -> [Double] {
        let maxLogit = logits.max() ?? 0
        let expValues = logits.map { exp($0 - maxLogit) }
        let sumExp = expValues.reduce(0, +)
        return expValues.map { $0 / sumExp }
    }
}
