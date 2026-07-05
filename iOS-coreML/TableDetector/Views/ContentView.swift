import SwiftUI
import UniformTypeIdentifiers

/// Single-screen UI: pick a scanned-document image from the Files app, run on-device
/// table detection against the bundled Table Transformer Core ML model, and show the result.
struct ContentView: View {
    @State private var pickedImage: UIImage?
    @State private var isPickerPresented = false
    @State private var isDetecting = false
    @State private var resultText: String?
    @State private var resultIsTable = false
    @State private var errorText: String?

    /// Lazily created once; if model loading fails we surface that as an error instead of crashing.
    private let detector: Result<TableDetector, Error> = Result { try TableDetector() }

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                imagePreview

                Button {
                    errorText = nil
                    isPickerPresented = true
                } label: {
                    Label("Choose Image File…", systemImage: "doc.badge.plus")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)

                if isDetecting {
                    ProgressView("Detecting table…")
                }

                if let resultText {
                    resultBanner(text: resultText, isTable: resultIsTable)
                }

                if let errorText {
                    Text(errorText)
                        .foregroundStyle(.red)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }

                Spacer()
            }
            .padding()
            .navigationTitle("Table Detector")
            .fileImporter(
                isPresented: $isPickerPresented,
                allowedContentTypes: [.jpeg, .png, .heic],
                allowsMultipleSelection: false
            ) { result in
                handlePickerResult(result)
            }
        }
    }

    @ViewBuilder
    private var imagePreview: some View {
        if let pickedImage {
            Image(uiImage: pickedImage)
                .resizable()
                .scaledToFit()
                .frame(maxHeight: 320)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .shadow(radius: 2)
        } else {
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.gray.opacity(0.15))
                .frame(height: 320)
                .overlay(
                    VStack(spacing: 8) {
                        Image(systemName: "doc.text.image")
                            .font(.system(size: 40))
                        Text("No image selected")
                    }
                    .foregroundStyle(.secondary)
                )
        }
    }

    private func resultBanner(text: String, isTable: Bool) -> some View {
        Label(text, systemImage: isTable ? "checkmark.circle.fill" : "xmark.circle.fill")
            .font(.headline)
            .foregroundStyle(isTable ? .green : .red)
            .multilineTextAlignment(.leading)
            .padding()
            .frame(maxWidth: .infinity)
            .background((isTable ? Color.green : Color.red).opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func handlePickerResult(_ result: Result<[URL], Error>) {
        resultText = nil
        errorText = nil

        switch result {
        case .failure(let error):
            errorText = "Could not open file: \(error.localizedDescription)"
        case .success(let urls):
            guard let url = urls.first else { return }
            loadAndDetect(from: url)
        }
    }

    private func loadAndDetect(from url: URL) {
        let didStartAccess = url.startAccessingSecurityScopedResource()
        defer {
            if didStartAccess { url.stopAccessingSecurityScopedResource() }
        }

        guard let data = try? Data(contentsOf: url), let image = UIImage(data: data) else {
            errorText = "Could not load the selected image."
            return
        }

        pickedImage = image
        runDetection(on: image)
    }

    private func runDetection(on image: UIImage) {
        switch detector {
        case .failure(let error):
            errorText = "Table detection model failed to load: \(error.localizedDescription)"
            return
        case .success(let detector):
            isDetecting = true
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try detector.detect(image: image)
                    DispatchQueue.main.async {
                        isDetecting = false
                        resultIsTable = result.containsTable
                        resultText = Self.describe(result)
                    }
                } catch {
                    DispatchQueue.main.async {
                        isDetecting = false
                        errorText = "Detection failed: \(error.localizedDescription)"
                    }
                }
            }
        }
    }

    private static func describe(_ result: TableDetectionResult) -> String {
        let percentage = Int((result.confidence * 100).rounded())
        return result.containsTable
            ? "Table detected — \(percentage)% confidence"
            : "No table detected (best confidence \(percentage)%)"
    }
}

#Preview {
    ContentView()
}
