import Foundation
import UIKit
import OnnxRuntimeBindings

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

/// Loads the bundled Table Transformer .ort model and runs on-device table detection.
///
/// Mirrors the pre/postprocessing implemented in `conversion_to_ort/ort_conversion.py`
/// (`detect_tables`), which itself matches Hugging Face's `DetrFeatureExtractor` for
/// `microsoft/table-transformer-detection`.
final class TableDetector {

    /// Matches the Python reference script's default `--score-threshold`.
    static let scoreThreshold: Double = 0.7

    private static let numQueries = 15
    private static let numClasses = 3 // "table", "table rotated", "no object"
    private static let tableLabelIndex = 0
    private static let rotatedLabelIndex = 1

    private let env: ORTEnv
    private let session: ORTSession

    init() throws {
        // The "with_runtime_opt" .ort variant is required for the CoreML/NNAPI execution
        // providers: the plain "Fixed"-optimized .ort bakes in CPU-only graph optimizations
        // that are incompatible with them (session init throws an axis-range error otherwise).
        guard let modelPath = Bundle.main.path(forResource: "table-transformer-detection.with_runtime_opt", ofType: "ort") else {
            throw TableDetectorError.modelNotFoundInBundle
        }
        env = try ORTEnv(loggingLevel: .warning)

        let sessionOptions = try ORTSessionOptions()
        if ORTIsCoreMLExecutionProviderAvailable() {
            // Runs supported nodes (mainly the ResNet backbone's convolutions) on the
            // Neural Engine/GPU via Core ML; unsupported nodes (e.g. the transformer
            // encoder/decoder's dynamic-shape ops) automatically fall back to CPU within
            // the same session, so this is always safe to enable even for a model that
            // doesn't convert to Core ML end-to-end.
            let coreMLOptions = ORTCoreMLExecutionProviderOptions()
            coreMLOptions.createMLProgram = true
            coreMLOptions.onlyAllowStaticInputShapes = true
            try sessionOptions.appendCoreMLExecutionProvider(with: coreMLOptions)
        }

        session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: sessionOptions)
    }

    /// Runs preprocessing + inference + postprocessing on `image`, synchronously.
    /// Call this off the main thread — a forward pass can take a noticeable fraction of a second.
    func detect(image: UIImage) throws -> TableDetectionResult {
        let preprocessed = try ImagePreprocessor.preprocess(image: image)

        let pixelValuesInput = try Self.makeTensor(
            from: preprocessed.pixelValues,
            shape: [1, 3, preprocessed.height, preprocessed.width]
        )
        let pixelMaskInput = try Self.makeTensor(
            from: preprocessed.pixelMask,
            shape: [1, preprocessed.height, preprocessed.width]
        )

        let outputs = try session.run(
            withInputs: ["pixel_values": pixelValuesInput, "pixel_mask": pixelMaskInput],
            outputNames: ["logits"],
            runOptions: nil
        )

        guard let logitsValue = outputs["logits"] else {
            throw TableDetectorError.unexpectedOutputShape
        }

        return try Self.postprocess(logitsValue: logitsValue)
    }

    // MARK: - Tensor construction

    private static func makeTensor(from floats: [Float], shape: [Int]) throws -> ORTValue {
        let data = floats.withUnsafeBytes { NSMutableData(bytes: $0.baseAddress, length: $0.count) }
        return try ORTValue(tensorData: data, elementType: .float, shape: shape.map { NSNumber(value: $0) })
    }

    private static func makeTensor(from ints: [Int64], shape: [Int]) throws -> ORTValue {
        let data = ints.withUnsafeBytes { NSMutableData(bytes: $0.baseAddress, length: $0.count) }
        return try ORTValue(tensorData: data, elementType: .int64, shape: shape.map { NSNumber(value: $0) })
    }

    // MARK: - Postprocessing

    /// `logits` has shape [1, numQueries, numClasses]. For each query we softmax the 3 class
    /// logits and take the best of the two "table" classes; the overall confidence is the max
    /// of that across all queries.
    private static func postprocess(logitsValue: ORTValue) throws -> TableDetectionResult {
        let tensorData = try logitsValue.tensorData()
        let data = tensorData as Data
        let expectedCount = numQueries * numClasses
        guard data.count == expectedCount * MemoryLayout<Float>.stride else {
            throw TableDetectorError.unexpectedOutputShape
        }

        let logits = data.withUnsafeBytes { rawBuffer -> [Float] in
            Array(rawBuffer.bindMemory(to: Float.self))
        }

        var bestScore = 0.0
        var bestLabel = "table"

        for query in 0..<numQueries {
            let base = query * numClasses
            let classLogits = (0..<numClasses).map { Double(logits[base + $0]) }
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
