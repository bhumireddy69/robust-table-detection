# AGENTS.md

## Project

This project is a proof of concept for robust table detection and table structure recognition in degraded or scanned documents.

The immediate goal is to build an incremental Python PoC that can:

1. Accept scanned PDFs or document images.
2. Convert PDF pages to images when needed.
3. Preprocess images for OCR and layout analysis.
4. Run OCR and preserve word-level coordinates.
5. Detect table regions, rows, columns, and cells.
6. Export results as text, CSV, Excel, images, and later structured JSON.

The research direction is not plain OCR. OCR is one component of a larger Document AI pipeline. The long-term angle is robustness under degradation: skew, blur, noise, low contrast, broken table borders, stains, compression artifacts, and scanned-page quality issues.

## Current Stack

- Python 3.12 in a project-local `.venv`
- Streamlit for the local PoC UI
- PyMuPDF for PDF rendering
- Pillow and OpenCV for image handling/preprocessing
- Tesseract through `pytesseract` for OCR baseline
- pandas/openpyxl for tabular exports

## Local Commands

Run commands from the project root:

```bash
source .venv/bin/activate
streamlit run app.py --server.port 8501
```

Basic verification:

```bash
python -m py_compile app.py src/*.py
python -c "import streamlit, cv2, fitz, pytesseract, pandas; print('imports ok')"
```

The local app should be available at:

```text
http://localhost:8501
```

## Project Structure

```text
app.py                  Streamlit UI and workflow orchestration
src/document_loader.py  PDF/image loading
src/image_processing.py OpenCV/Pillow preprocessing and drawing
src/ocr.py              Tesseract OCR wrappers
input/                  Local test documents
output/                 Generated artifacts
requirements.txt        Frozen Python dependencies
```

## Coding Guidelines

- Keep changes incremental and easy for a beginner Python user to understand.
- Prefer small modules in `src/` over putting all logic into `app.py`.
- Keep `app.py` focused on UI and orchestration.
- Keep reusable document/image/OCR/table logic in `src/`.
- Do not commit `.venv/`, `__pycache__/`, or generated `output/` artifacts.
- Use ASCII text in source files unless there is a clear reason not to.
- Add comments only for non-obvious image-processing or table-structure logic.
- Favor readable code over clever abstractions.

## Development Plan

Current stage:

1. Upload PDF/image.
2. Render PDF page to image if needed.
3. Preprocess image.
4. Run Tesseract OCR.
5. Show OCR text, word boxes, and generated files.

Next stages:

1. Add OpenCV line detection for ruled tables.
2. Detect candidate table bounding boxes.
3. Detect row and column separators.
4. Recover cell grid geometry.
5. Align OCR words into detected cells.
6. Export a first rough CSV/Excel table.
7. Add degradation experiments and metrics.
8. Compare against industry Document AI tools and stronger model-based baselines.

## Research Notes

The conference-paper contribution should focus on table structure robustness, not only OCR accuracy. Useful paper framing:

- Degradation-aware preprocessing.
- Hybrid classical computer vision plus OCR grounding.
- Structure repair for broken or partial ruling lines.
- Evaluation across clean vs degraded scans.
- Comparison against baseline OCR/table extraction systems.

When adding advanced AI/ML models, start with reproducible baselines before custom training.
