import UIKit
import Accelerate

/// Replicates the preprocessing performed by Hugging Face's `DetrFeatureExtractor`
/// for `microsoft/table-transformer-detection` (do_resize=true, size=800, max_size=800),
/// which collapses to "fit the longest side to 800 while preserving aspect ratio".
enum ImagePreprocessor {

    struct PreprocessedImage {
        /// NCHW float32 buffer: [1, 3, height, width], RGB, normalized.
        let pixelValues: [Float]
        /// [1, height, width] all-ones mask (no padding for single-image inference).
        let pixelMask: [Int64]
        let width: Int
        let height: Int
    }

    private static let targetSide: CGFloat = 800.0
    private static let mean: [Float] = [0.485, 0.456, 0.406]
    private static let std: [Float] = [0.229, 0.224, 0.225]

    enum PreprocessError: Error {
        case couldNotDecodeImage
        case couldNotCreateContext
    }

    /// Resizes `image` so its longest side is 800px (preserving aspect ratio), then
    /// converts it to a normalized, channel-first RGB float tensor plus an all-ones mask.
    static func preprocess(image: UIImage) throws -> PreprocessedImage {
        guard let cgImage = image.cgImage else {
            throw PreprocessError.couldNotDecodeImage
        }

        let originalWidth = CGFloat(cgImage.width)
        let originalHeight = CGFloat(cgImage.height)
        let scale = targetSide / max(originalWidth, originalHeight)
        let newWidth = max(1, Int((originalWidth * scale).rounded()))
        let newHeight = max(1, Int((originalHeight * scale).rounded()))

        let rgbaBuffer = try renderRGBA(cgImage: cgImage, width: newWidth, height: newHeight)

        var pixelValues = [Float](repeating: 0, count: 3 * newWidth * newHeight)
        let planeSize = newWidth * newHeight

        rgbaBuffer.withUnsafeBufferPointer { rgba in
            for pixelIndex in 0..<planeSize {
                let base = pixelIndex * 4
                let r = Float(rgba[base]) / 255.0
                let g = Float(rgba[base + 1]) / 255.0
                let b = Float(rgba[base + 2]) / 255.0

                pixelValues[pixelIndex] = (r - mean[0]) / std[0]
                pixelValues[planeSize + pixelIndex] = (g - mean[1]) / std[1]
                pixelValues[2 * planeSize + pixelIndex] = (b - mean[2]) / std[2]
            }
        }

        let pixelMask = [Int64](repeating: 1, count: planeSize)

        return PreprocessedImage(pixelValues: pixelValues, pixelMask: pixelMask, width: newWidth, height: newHeight)
    }

    /// Draws `cgImage` into an 8-bit RGBA bitmap context of the given size, which both
    /// resizes (bilinear, via CoreGraphics' default interpolation) and gives us raw pixels.
    private static func renderRGBA(cgImage: CGImage, width: Int, height: Int) throws -> [UInt8] {
        var buffer = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        guard let context = CGContext(
            data: &buffer,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw PreprocessError.couldNotCreateContext
        }

        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        return buffer
    }
}
