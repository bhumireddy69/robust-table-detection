# Project Context: Robust Table Detection And Structure Recognition

## Goal

Build a research-oriented proof of concept for robust table detection and table structure recognition in degraded or scanned documents.

The target is not plain OCR. OCR is only one layer. The actual research problem is:

```text
scanned/degraded PDF or image
-> detect table regions
-> recover rows, columns, cells, headers, and structure
-> align recognized text into cells
-> export CSV/Excel/JSON
```

The long-term research angle is robustness under difficult document conditions:

- scanned pages,
- blur,
- low resolution,
- skew,
- noise,
- low contrast,
- broken table borders,
- borderless or partially ruled tables,
- multi-column scientific documents,
- nearby captions/paragraph text.

Working paper title candidate:

```text
Robust Table Detection and Structure Recognition in Degraded Scanned Documents Using Hybrid Layout Analysis
```

## Current Project Location

```text
/Users/chaitanya/Documents/table-structure
```

Run app:

```bash
cd /Users/chaitanya/Documents/table-structure
source .venv/bin/activate
streamlit run app.py --server.port 8501
```

The app may also be run on another free port, such as `8502`.

## Current Project Structure

```text
table-structure/
  AGENTS.md
  PROJECT_CONTEXT.md
  app.py
  requirements.txt
  input/
  output/
  experiments/
    experiment_log.md
  evaluation/
    README.md
    annotations.example.csv
    run_detection_eval.py
  src/
    document_loader.py
    detection_metrics.py
    detection_service.py
    image_processing.py
    ocr.py
    table_structure.py
    table_engines/
      __init__.py
      paddle_ocr_engine.py
      paddle_table_engine.py
      table_transformer_engine.py
```

## Tools And Libraries Used So Far

System/tools:

- macOS
- VS Code
- Docker Desktop
- Homebrew
- Python 3.12 virtual environment
- Tesseract
- Poppler

Python/app libraries:

- Streamlit: local UI
- PyMuPDF: PDF page rendering
- Pillow: image loading/conversion
- OpenCV: preprocessing, line detection, drawing boxes
- pytesseract: Tesseract OCR wrapper
- pandas/openpyxl: dataframe display and future exports
- PaddleOCR/PaddleX: OCR and table-aware document AI
- PyTorch/Transformers/timm: Microsoft Table Transformer runtime

## What Has Been Built

### 1. Streamlit PoC App

`app.py` can:

- upload scanned PDF/image,
- render selected PDF page,
- display original page,
- preprocess image for OCR,
- run OCR,
- show OCR text and boxes,
- save generated files to `output/`,
- show heuristic table detection results,
- run PaddleOCR table recognition,
- compare custom heuristic table box vs PaddleOCR table-recognition box.

Sidebar controls currently include:

- PDF render DPI,
- OCR Engine: `Tesseract` or `PaddleOCR`,
- Table Engine: `None`, `PaddleOCR Table Recognition`, `Table Transformer Detection`, `Table Transformer Structure Recognition`, or `All Table Engines`,
- manual table region coordinates.

### 2. Tesseract Baseline OCR

File:

```text
src/ocr.py
```

Provides:

- `extract_text(image)`
- `extract_words(image)`

Tesseract returns word-level boxes:

```text
text
conf
left
top
width
height
block_num
par_num
line_num
word_num
```

This worked better with our custom row/column heuristics because the heuristics expect word-level boxes.

### 3. Image Preprocessing And Drawing

File:

```text
src/image_processing.py
```

Current functionality:

- convert Pillow image to OpenCV image,
- convert OpenCV image back to Pillow,
- OCR preprocessing:
  - grayscale,
  - denoise,
  - adaptive threshold,
- draw OCR word boxes,
- detect horizontal/vertical table lines using morphology,
- draw line overlay,
- draw region/table boxes.

Important result:

- Line detection helps for ruled tables.
- It is insufficient for borderless or partially ruled tables.

### 4. Custom Heuristic Table Pipeline

File:

```text
src/table_structure.py
```

Current functions include:

- group OCR words into visual rows,
- filter words by manual or detected region,
- summarize rows,
- find table-like rows,
- find repeated X positions,
- select strongest X band,
- build automatic candidate table box,
- infer column anchors,
- merge column anchors,
- assign words to column groups,
- build rough table rows.

Current custom pipeline:

```text
Tesseract word boxes
-> visual row grouping
-> numeric/layout row summaries
-> candidate table-like rows
-> repeated X alignment
-> strongest X band
-> automatic candidate table box
-> rough column grouping
-> mapped table preview
```

### 5. PaddleOCR Plain OCR Engine

File:

```text
src/table_engines/paddle_ocr_engine.py
```

Adds PaddleOCR as an alternate OCR engine without replacing Tesseract.

Observed behavior:

- PaddleOCR text recognition is cleaner in some cases.
- Installed version observed: `PaddleOCR 3.7.0`.
- Current wrapper returns line-level OCR boxes, not individual word boxes.
- Because our custom heuristic expects word-level boxes, the custom heuristic table detection often fails with PaddleOCR plain OCR.

Important research finding:

```text
A stronger OCR engine alone is not enough for robust table structure extraction.
Table-aware modeling is needed.
```

### 6. PaddleOCR Table Recognition Engine

File:

```text
src/table_engines/paddle_table_engine.py
```

Uses PaddleOCR/PaddleX `TableRecognitionPipelineV2`.

The app shows:

- detected table boxes,
- table-region overlay,
- parsed dataframe from Paddle's predicted HTML,
- raw Paddle table HTML,
- Paddle table OCR texts.

Observed on PubTables-1M test page:

```text
Detected table box:
left=905
top=687
right=1466
bottom=895
score=0.93

Tables returned: 1
Detected cells: 28
Parsed dataframe: 6 x 5
```

This detected the table region very well.

But structure/content reconstruction was still imperfect:

- some cells misplaced,
- row label not reconstructed cleanly,
- some values merged into wrong cells,
- partially ruled/borderless table remains difficult.

Important research finding:

```text
PaddleOCR table recognition can localize the table well, but structure recognition still has errors on the partially ruled/borderless sample.
```

### 7. Table Transformer Detection And Structure Engine

File:

```text
src/table_engines/table_transformer_engine.py
```

Uses Microsoft Table Transformer through Hugging Face Transformers:

```text
microsoft/table-transformer-detection
microsoft/table-transformer-structure-recognition
```

The detection function returns table-region boxes:

```text
label
score
left
top
right
bottom
```

Observed on the PubTables-1M test page:

```text
Detected table box:
left=909
top=717
right=1465
bottom=893
score=0.9994
```

This is very close to the PaddleOCR table-recognition box and validates Table Transformer as a strong table-region detection baseline.

Structure recognition is now integrated as a first pass:

```text
detected table box
-> padded table crop
-> Table Transformer structure-recognition model
-> detected rows and columns
-> Tesseract word boxes mapped into row/column intersections
```

Observed on the PubTables-1M test page:

```text
rows: 5
columns: 6
mapped grid: 5 x 6
```

Example mapped row:

```text
ASCA | better | 19457 (28.9) | 12 (0.02) | 14654 (21.8) | 34,123 (50.8)
```

Remaining limitations:

```text
OCR noise still appears, such as `t equal`, `r Sum`, and `=.`
The structure model provides geometry; OCR quality and OCR-to-cell cleanup still matter.
```

## Test Documents Used

Current input examples:

```text
input/test_table_document.pdf
input/degraded_table_01.png
```

Useful generated outputs are in:

```text
output/
```

Important generated outputs include:

- original page image,
- preprocessed image,
- line overlays,
- OCR text,
- OCR word CSV,
- custom heuristic table box,
- PaddleOCR table box.

## Main Results So Far

### Degraded Image Test

File:

```text
input/degraded_table_01.png
```

Image was low resolution and blurry:

```text
540 x 284 pixels
```

Result:

- OCR text was mostly incorrect.
- Some word regions were detected.
- Source image likely lacks enough character detail.

Research value:

```text
Good degraded failure case for robustness experiments.
```

Updated degraded-table observation from `All Table Engines`:

```text
Custom heuristic: no candidate table box
PaddleOCR table recognition: app/result error, not usable
Table Transformer detection: table detected with score around 0.9705
Table Transformer mapped structure: 11 rows x 8 columns, but text/cell contents mostly wrong
OCR row groups: noisy but preserved useful row-level store/value/percentage patterns
```

Important interpretation:

```text
For degraded images, table localization can succeed while cell-level extraction fails.
OCR row grouping may be a useful fallback signal when final mapped table data is unreliable.
```

Important clarification:

```text
PaddleOCR plain OCR / OCR row grouping is different from PaddleOCR table recognition.
```

In the degraded spreadsheet test:

```text
OCR row groups preserved useful row-level text patterns.
PaddleOCR table recognition did not return a usable degraded-table detection.
Table Transformer detected the degraded table region.
```

So the batch metric row `paddleocr_table` measures PaddleOCR table-recognition detection only. It does not measure whether OCR row groups found readable text.

### Clean PDF / PubTables Page Test

File:

```text
input/test_table_document.pdf
```

Results:

- Tesseract produced usable word boxes.
- Custom heuristic localized a rough table box with Tesseract.
- PaddleOCR plain OCR produced line-level boxes.
- Custom heuristic did not detect the table when using PaddleOCR plain OCR.
- PaddleOCR Table Recognition directly detected the table box well.
- PaddleOCR table structure output was still imperfect.
- Table Transformer detected the table box very accurately as an independent table-specific baseline.

## Key Research Findings So Far

1. **OCR alone is not table extraction.**
   OCR can read text and boxes, but does not fully recover table structure.

2. **Line detection alone is insufficient.**
   Borderless or partially ruled tables lack full grid lines.

3. **Whole-page OCR row grouping is noisy.**
   Paragraph text and table text can share similar Y positions, especially in multi-column documents.

4. **Manual region filtering is useful only as an oracle/debug step.**
   It helps separate table localization from structure recognition, but it is not a final method.

5. **Custom heuristics can work as Baseline 1.**
   Tesseract word boxes plus row/X clustering produced a rough table region and rough dataframe.

6. **PaddleOCR plain OCR is not enough.**
   It gives cleaner OCR but line-level boxes break our word-level custom heuristics.

7. **PaddleOCR Table Recognition is a stronger table-localization baseline.**
   It detected the table region accurately on the PubTables page.

8. **Even table-aware AI still struggles with structure.**
   PaddleOCR table recognition produced imperfect cell/text reconstruction on the partially ruled/borderless sample.

9. **Table Transformer is now a detection and structure baseline.**
   On the PubTables page, it detected the same table region with high confidence and produced a 5 x 6 grid.

## Current Baselines

### Baseline 1: Classical/Custom

```text
Tesseract OCR
+ OpenCV preprocessing/line detection
+ custom row/X/column heuristics
```

Strengths:

- transparent,
- good for learning,
- useful comparison baseline,
- works with word-level boxes.

Weaknesses:

- brittle,
- sensitive to OCR quality,
- struggles with borderless tables,
- multi-token cells and noise rows are hard.

### Baseline 2: PaddleOCR Plain OCR

```text
PaddleOCR OCR
```

Strengths:

- cleaner OCR text in some cases,
- stronger modern OCR baseline.

Weaknesses:

- current wrapper returns line-level boxes,
- not enough for table structure by itself.

### Baseline 3: PaddleOCR Table Recognition

```text
PaddleOCR TableRecognitionPipelineV2
```

Strengths:

- detects table region automatically,
- returns table HTML and cell predictions,
- stronger document-AI baseline.

Weaknesses:

- structure output still imperfect,
- merged/misplaced cells on sample table,
- needs evaluation on degraded documents.

### Baseline 4: Table Transformer Detection/Structure Recognition

```text
Microsoft Table Transformer detection model
+ Microsoft Table Transformer structure-recognition model
+ Tesseract word-box alignment
```

Strengths:

- table-specific AI detector,
- aligned with PubTables-1M-style research,
- detected the PubTables sample table accurately,
- useful independent comparison against PaddleOCR table detection,
- first-pass row/column grid is close to expected output on the PubTables sample.

Weaknesses:

- still depends on OCR word quality,
- noisy OCR tokens still pollute mapped cells,
- current cell mapping is simple row/column intersection logic.

## Current App Behavior To Know

### OCR Engine Selector

```text
Tesseract
PaddleOCR
```

If `Tesseract` is selected:

- custom heuristic is likely to work better because it gets word-level boxes.

If `PaddleOCR` is selected:

- custom heuristic may fail because boxes are line-level.
- PaddleOCR table recognition can still work if enabled separately.

### Table Engine Selector

```text
None
PaddleOCR Table Recognition
Table Transformer Detection
Table Transformer Structure Recognition
All Table Engines
```

When PaddleOCR Table Recognition is selected:

- app runs the table-aware model,
- displays Paddle's detected table box,
- displays parsed dataframe and raw HTML.

When Table Transformer Detection is selected:

- app runs Microsoft's table detector,
- displays detected table boxes,
- draws the strongest detected table box on the original page,
- includes it in Baseline Comparison.

When Table Transformer Structure Recognition is selected:

- app runs Microsoft's table detector,
- crops the detected table with padding,
- runs Microsoft's structure-recognition model,
- displays row/column box counts,
- draws detected rows and columns,
- maps Tesseract word boxes into a first-pass dataframe.

When All Table Engines is selected:

- app runs PaddleOCR Table Recognition,
- app runs Table Transformer detection and structure recognition,
- Baseline Comparison shows custom heuristic, PaddleOCR, and Table Transformer boxes in the same run,
- Structure Comparison shows custom heuristic, PaddleOCR parsed table, and Table Transformer mapped table in the same run.

### Baseline Comparison Section

Shows side-by-side:

- custom heuristic candidate box,
- PaddleOCR table-recognition box.
- Table Transformer detection box when that engine is selected.

Important implementation detail:

```text
The custom heuristic candidate box is computed using Tesseract word-level boxes,
even when the selected OCR Engine is PaddleOCR.
```

Reason:

```text
The custom heuristic baseline depends on word-level boxes.
PaddleOCR plain OCR currently returns line-level boxes in our wrapper.
So fair comparison is:
custom heuristic + Tesseract word boxes
vs
PaddleOCR Table Recognition
```

Important observed comparison:

```text
OCR Engine: PaddleOCR
Table Engine: PaddleOCR Table Recognition

Custom heuristic + Tesseract word boxes: detected successfully
PaddleOCR table-recognition box: detected successfully
```

Observed behavior:

- Both methods detect the same general table region.
- The custom heuristic box is slightly tighter/lower.
- PaddleOCR Table Recognition includes slightly more top/header area.
- Both avoid most surrounding paragraph text.

Next comparison should focus on structure quality:

```text
custom mapped table preview
vs
PaddleOCR parsed dataframe/HTML
```

This has now been added as a Streamlit section:

```text
Structure Comparison
```

It shows side by side:

- custom heuristic mapped table,
- PaddleOCR parsed table dataframe,
- row count and column count for each.

Observed on the PubTables sample:

```text
Custom heuristic: 6 rows x 6 columns
PaddleOCR parsed table: 6 rows x 5 columns
```

Findings:

- Custom heuristic produced more readable body rows.
- PaddleOCR captured the header/table structure better.
- Custom heuristic still had OCR noise and split final value/percentage cells.
- PaddleOCR still misplaced/merged some body values.

Research interpretation:

```text
Both methods can localize the table, but neither perfectly reconstructs structure.
This strengthens the research motivation around robust table structure recognition.
```

## Pending / Next Work

### Immediate Next Step

Table Transformer structure recognition has been integrated. The next step is improving OCR-to-cell cleanup:

```text
Table Transformer row/column boxes
-> map OCR words into cells
-> clean noisy OCR fragments
-> handle spanning/header cells
-> compare against PaddleOCR and custom heuristic outputs
```

### Evaluation Step

After structure recognition is integrated, test these paths on the same documents:

```text
1. Tesseract + custom heuristic
2. PaddleOCR plain OCR
3. PaddleOCR Table Recognition
4. Table Transformer Detection/Structure Recognition
```

For each document, record:

- table detected or missed,
- detected box quality,
- OCR quality,
- dataframe/table structure quality,
- failure modes.

### Later Engineering Step

Export/save support is useful but not the immediate priority. Add later:

- save Paddle table HTML,
- save parsed Paddle dataframe as CSV,
- save detected table boxes as JSON,
- save visual overlay image,
- save comparison summaries.

### Immediate Next Research Step

Table Transformer detection and first-pass structure recognition are integrated.

Planned architecture:

```text
page image
-> Table Transformer detects table region
-> crop table
-> Table Transformer detects rows/columns/cells
-> OCR engine supplies text boxes
-> align text into predicted cells
-> export structured output
```

Table Transformer is useful because:

- it is a focused table detection/structure model,
- it is aligned with PubTables-1M-style research,
- it provides a stronger table-specific comparison than general OCR.

### Later Work

- Evaluate degraded image cases.
- Add preprocessing variants:
  - upscale,
  - sharpen,
  - contrast enhancement,
  - deskew,
  - threshold tuning.
- Compare clean vs degraded performance.
- Define metrics:
  - table detection success,
  - box IoU if ground truth exists,
  - row/column/cell correctness,
  - OCR-to-cell mapping accuracy,
  - CSV reconstruction quality.
- Collect public datasets:
  - ICDAR 2013 Table Competition,
  - Marmot,
  - PubTables-1M,
  - FinTabNet,
  - TableBank.

## Important Design Decisions

1. Do not remove old code.
   The custom Tesseract/OpenCV pipeline is Baseline 1.

2. Add new engines in separate files.
   Current pattern:

   ```text
   src/table_engines/
   ```

3. Keep app selectors.
   The app should allow side-by-side comparison rather than replacing one method with another.

4. Prefer research logging after each experiment.
   Use:

   ```text
   experiments/experiment_log.md
   ```

5. `img2table` is not prioritized now.
   Current focus is:

   ```text
   custom heuristic baseline
   PaddleOCR OCR/table-recognition baseline
   Table Transformer baseline
   ```

## Research Paper Story So Far

The emerging story:

```text
Classical OCR/CV heuristics can provide a transparent baseline but are brittle.
Plain OCR is insufficient for table structure.
Table-aware AI models improve table localization but still struggle with structure on partially ruled/borderless/degraded documents.
Robust table structure recognition requires combining table localization, OCR grounding, structure prediction, and degradation-aware preprocessing/evaluation.
```

This supports a paper focused on robust table detection and structure recognition in degraded scanned documents.

## Useful Commands

Run app:

```bash
cd /Users/chaitanya/Documents/table-structure
source .venv/bin/activate
streamlit run app.py --server.port 8501
```

Syntax check:

```bash
python -m py_compile app.py src/*.py src/table_engines/*.py
```

Update dependencies after installs:

```bash
pip freeze > requirements.txt
```

## Where To Continue

Batch detection evaluation has been added as a separate non-Streamlit flow:

```text
src/detection_service.py
src/detection_metrics.py
evaluation/run_detection_eval.py
evaluation/annotations.example.csv
evaluation/README.md
```

Smoke-tested command:

```bash
python evaluation/run_detection_eval.py \
  --input input \
  --annotations evaluation/annotations.example.csv \
  --output-dir evaluation/results_smoke_all_methods \
  --methods all \
  --only-annotated-pages
```

Outputs:

```text
evaluation/results_smoke_all_methods/predictions.csv
evaluation/results_smoke_all_methods/metrics_summary.csv
evaluation/results_smoke_all_methods/errors.csv
```

Current smoke metrics from example annotations:

```text
custom_heuristic: precision=1.0 recall=0.5 f1=0.6667 mean_best_iou_per_gt=0.3265
paddleocr_table: precision=1.0 recall=0.5 f1=0.6667 mean_best_iou_per_gt=0.5
table_transformer: precision=1.0 recall=1.0 f1=1.0 mean_best_iou_per_gt=0.9193
```

Important caveat:

```text
annotations.example.csv is only for smoke testing the pipeline.
For real research metrics, use dataset-provided or manually verified ground-truth boxes.
```

Recommended next turn:

```text
Improve Table Transformer OCR-to-cell cleanup and header/spanning-cell handling.
```

Then:

```text
Compare custom heuristic, PaddleOCR parsed table, and Table Transformer mapped table on the same document.
```

This comparison has now been tested successfully with `All Table Engines` on the PubTables sample:

```text
Custom heuristic box: left=889 top=740 right=1480 bottom=915
PaddleOCR box: left=905 top=687 right=1466 bottom=895 score=0.9328
Table Transformer box: left=909 top=717 right=1465 bottom=893 score=0.9994

Custom heuristic structure: 6 rows x 6 columns
PaddleOCR parsed structure: 6 rows x 5 columns
Table Transformer mapped structure: 5 rows x 6 columns
```

Interpretation:

```text
All three localize the table.
Structure extraction is still imperfect across all methods.
Table Transformer gives strong geometry, but OCR cleanup/header handling is still needed.
```

Later:

```text
Save/export comparison outputs for experiments.
```
