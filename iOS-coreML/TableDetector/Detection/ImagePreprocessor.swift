import UIKit

/// Preprocessing for the fixed-shape Core ML export (see conversion_to_ort/convert_to_coreml.py).
///
/// Unlike the ONNX/.ort model (which accepts any image size and preserves aspect ratio when
/// resizing), this Core ML model was traced with a FIXED 800x1000 input, so the image must be
/// letterboxed/stretched to exactly that size rather than resized preserving aspect ratio.
enum ImagePreprocessor {

    struct PreprocessedImage {
        /// NCHW float32 buffer: [1, 3, targetHeight, targetWidth], RGB, ImageNet-normalized.
        let pixelValues: [Float]
        /// [1, targetHeight, targetWidth] all-ones mask (no padding for single-image inference).
        let pixelMask: [Int32]
    }

    /// Must match the fixed size conversion_to_ort/convert_to_coreml.py traced the model with.
    static let targetHeight = 800
    static let targetWidth = 1000

    private static let mean: [Float] = [0.485, 0.456, 0.406]
    private static let std: [Float] = [0.229, 0.224, 0.225]

    enum PreprocessError: Error {
        case couldNotDecodeImage
        case couldNotCreateContext
    }

    /// Resizes `image` to exactly targetWidth x targetHeight (not aspect-preserving -- this
    /// matches the letterboxing that conversion_to_ort/convert_to_coreml.py's `detect_tables`
    /// reference implementation uses), then converts it to a normalized, channel-first RGB
    /// float tensor plus an all-ones mask.
    static func preprocess(image: UIImage) throws -> PreprocessedImage {
        guard let cgImage = image.cgImage else {
            throw PreprocessError.couldNotDecodeImage
        }

        let rgbaBuffer = try renderRGBA(cgImage: cgImage, width: targetWidth, height: targetHeight)

        var pixelValues = [Float](repeating: 0, count: 3 * targetWidth * targetHeight)
        let planeSize = targetWidth * targetHeight

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

        let pixelMask = [Int32](repeating: 1, count: planeSize)

        return PreprocessedImage(pixelValues: pixelValues, pixelMask: pixelMask)
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
