# Experiment Log

This file is a running rough log for table detection and structure recognition experiments. Keep entries practical: input file, output files, what worked, what failed, and next ideas to test.

## 2026-06-29 - Initial OCR Pipeline Tests

### Current Pipeline

Input document/image goes through:

1. Load PDF/image.
2. Convert PDF page to image when needed.
3. Convert image to RGB with Pillow.
4. Convert to OpenCV BGR image.
5. Convert to grayscale.
6. Apply denoising with `cv2.fastNlMeansDenoising(gray, h=18)`.
7. Apply adaptive thresholding with Gaussian local thresholding.
8. Run Tesseract OCR.
9. Save original image, preprocessed image, OCR text, word CSV, and word-box visualization.

### Test Input: `input/degraded_table_01.png`

File details:

- Type: PNG
- Size: 540 x 284 pixels
- Visual quality: low-resolution, blurry/compressed table screenshot

Generated output files:

- `output/page_1_original.png`
- `output/page_1_preprocessed.png`
- `output/page_1_word_boxes.png`
- `output/page_1_ocr.txt`
- `output/page_1_words.csv`

Observed OCR result:

- OCR text was mostly incorrect/nonsense.
- Tesseract did detect several word regions.
- Word boxes appeared over many table text areas, but the recognized characters were poor.

Likely cause:

- Source image has insufficient resolution for reliable character recognition.
- The text is blurry before preprocessing.
- Current thresholding makes some characters and table lines merge into thick blobs.
- OCR can locate some text-like regions, but cannot read the characters accurately.

Research value:

- Useful degraded failure case.
- Represents low-resolution blurred screenshot-like tables.
- Good candidate for testing robustness improvements.

Next experiments to try:

- Upscale image before OCR.
- Try contrast enhancement before thresholding.
- Try sharpening before OCR.
- Tune denoising strength `h`.
- Tune adaptive threshold block size and `C`.
- Compare OCR on original grayscale vs thresholded image.
- Later try table-line detection even when text OCR fails.

### Test Input: `input/test_table_document.pdf`

Generated output files:

- `output/page_2_original.png`
- `output/page_2_preprocessed.png`
- `output/page_2_word_boxes.png`
- `output/page_2_ocr.txt`
- `output/page_2_words.csv`

Observed OCR result:

- OCR output was much more readable than the degraded PNG.
- The rendered PDF page was larger and clearer.
- Word extraction produced many valid words and coordinates.

Likely cause:

- PDF page rendered at higher effective resolution.
- Text and document layout had enough detail for Tesseract.

Research value:

- Useful baseline clean/clear document case.
- Can compare against degraded image behavior.

Next experiments to try:

- Add table line detection on this clearer page first.
- Use OCR word coordinates to test future cell mapping logic.
- Compare with degraded page after adding preprocessing variants.

## 2026-06-29 - Target 2 Table Line Detection Notes

### Implementation Direction

Added/started OpenCV morphology-based table line detection:

1. Use the preprocessed black/white image.
2. Convert defensively to grayscale with Pillow mode `"L"`.
3. Invert the image so dark text/table lines become white foreground.
4. Use a wide horizontal rectangular kernel to extract horizontal lines.
5. Use a tall vertical rectangular kernel to extract vertical lines.
6. Combine horizontal and vertical line masks.
7. Draw detected line pixels as an overlay on the original image.

### Important Finding: Lines Alone Are Not Enough

Observed/raised case:

- Borderless or partially ruled table with no full grid.
- Example characteristics:
  - Few or no vertical ruling lines.
  - Some horizontal separators.
  - Structure is implied by text alignment, row spacing, column spacing, bold headers, and shaded bands.

Conclusion:

- Pure line detection works mainly for ruled tables.
- It cannot reliably extract data from borderless tables.
- The project must support both visual-line detection and OCR-layout-based structure inference.

Needed second path:

1. Use OCR word boxes.
2. Group words into rows using Y coordinates.
3. Infer columns using X-coordinate alignment and spacing.
4. Build a rough grid from row and column groups.
5. Map OCR text into inferred cells.

Research value:

- This is a key problem statement for the paper: real-world tables may be ruled, partially ruled, or borderless.
- A robust system should not depend only on visible ruling lines.
- Hybrid strategy is needed: line-based extraction for ruled tables and OCR/layout-based extraction for borderless tables.

Next implementation target:

- Target 3A: create OCR-based row grouping from Tesseract word boxes.
- Then Target 3B: infer columns from X positions.
- Then Target 3C: build rough cell grid and export a first CSV.

## 2026-06-29 - Target 3A OCR Row Grouping Notes

### Implementation Direction

Added/started OCR coordinate-based row grouping:

1. Use Tesseract word-level OCR output.
2. Sort words by vertical position.
3. Use each word's vertical center, `top + height / 2`.
4. Group words into the same visual row when their Y-centers are within a tolerance.
5. Sort words in each row left-to-right by `left`.
6. Display row groups in the Streamlit app for debugging.

### Observation From `test_table_document.pdf`

The app output shows:

- Line detection found only a few horizontal rules in the small example table figure.
- The table is partially ruled/borderless, so full grid extraction from lines alone is not possible.
- OCR row grouping produced readable visual rows, including table rows.

Important failure/limitation:

- Because the PDF page has paragraph text on the left and a table figure on the right, row grouping over the whole page can mix unrelated content that shares similar Y coordinates.
- Example pattern: a paragraph line and a table row appear in the same grouped row because they are horizontally separate but vertically aligned.

Conclusion:

- OCR row grouping should not be applied blindly to the whole page for final table extraction.
- We first need to identify or crop candidate table regions.
- After isolating a table region, OCR row grouping becomes much more meaningful.

Research value:

- This exposes another real-world challenge: document pages often contain tables near body text, captions, figures, and multi-column layouts.
- Robust table structure recognition needs table-region detection before row/column inference.
- For partially ruled/borderless tables, the extraction pipeline likely needs:
  1. candidate table localization,
  2. crop/filter words inside candidate region,
  3. row grouping,
  4. column inference,
  5. cell text mapping.

Next implementation target:

- Target 3B should focus on table-region filtering before column inference.
- Possible first simple approach: let the user or heuristic choose a region around detected table-like line/word clusters.
- Later: automate candidate table region detection using line density, word alignment, and whitespace boundaries.

## 2026-06-29 - Manual Region Filtering Result

### Test Observation

Manual table-region filtering was added with sidebar coordinates:

- left
- top
- right
- bottom

The goal was to filter OCR words to a selected rectangle before row grouping.

Observed behavior:

- A loose region still included paper title, author text, paragraph lines, captions, and table text.
- Row grouping then mixed unrelated paragraph words with table words when they shared similar Y positions.
- A tighter region around only the table is expected to improve row grouping, but this requires user tuning.

Conclusion:

- Manual region filtering is useful as a debugging/oracle step.
- It is not a final extraction strategy.
- It helps separate two subproblems:
  1. table localization: finding where the table is,
  2. structure recognition: recovering rows, columns, and cells inside a known table region.

Research value:

- Manual region selection can be described as an oracle setup for early experiments.
- It allows evaluation of structure recognition independently from table detection.
- The final system must replace manual coordinates with automatic candidate table-region detection.

Next target:

- Automatic table-region detection using OCR layout signals and line cues.
- First heuristic idea: detect table-like word clusters by repeated rows, numeric density, repeated X alignments, and compact bounding boxes.

## 2026-06-29 - Row Numeric-Ratio Heuristic

### Implementation Direction

Added row summarization for OCR row groups:

- row text
- row bounding box: left, top, right, bottom
- word count
- numeric-like token count
- numeric ratio

Numeric-like tokens currently include simple forms such as:

- `19457`
- `(28.9)`
- `12`
- `(0.02)`
- `34,123`
- `(50.8)`

The Streamlit row-group display now shows:

```text
Row N: words=<count> numeric_ratio=<ratio> | <row text>
```

### Observed Result

On the PubTables-1M test page, rows containing table values had noticeably higher numeric ratios than ordinary text/header rows.

Example observed pattern from saved OCR:

- normal text/header rows: numeric ratio often `0.00`
- table-containing rows: numeric ratio around `0.33` to `0.50`
- one OCR-fragment row had numeric ratio around `0.75`

### Limitation

Numeric ratio alone is not sufficient for final table detection:

- Some non-table rows can contain citations, section numbers, years, figure numbers, or page numbers.
- Rows from a table can still be mixed with paragraph text if the row grouping is performed on the whole page.
- Numeric-heavy tables are easier; text-heavy tables may not be detected by this heuristic.

### Research Value

Numeric density is a useful weak signal for candidate table rows, especially for financial/statistical tables. It should be combined with:

- repeated X alignment,
- compact row clusters,
- consistent row spacing,
- line cues where available,
- whitespace/layout boundaries.

Next implementation target:

- Detect candidate table-like row clusters from row summaries.
- Use numeric ratio and row proximity to propose an initial table bounding box.

## 2026-06-29 - Candidate Table-Like Rows

### Implementation Direction

Added `find_table_like_rows()`:

- input: grouped OCR rows,
- summarize each row,
- keep rows whose word count and numeric ratio pass thresholds.

Current default heuristic:

- minimum words: `4`
- minimum numeric ratio: `0.3`

### Observed Result

On the PubTables-1M test page, this found 5 candidate table-like rows.

The selected rows correspond to the visual Y positions of the example table:

- row containing `ASCA better 19457 ...`
- row containing `equal 1158 ...`
- row containing `worse 3755 ...`
- row containing `Sum 24370 ...`
- one continuation/OCR-fragment row containing `(32.7) (31.0)`

### Limitation

The candidate row bounding boxes still span both:

- paragraph text on the left side of the page,
- table values on the right side of the page.

This happens because the PDF is a multi-column/figure layout where paragraph lines and table rows share similar Y coordinates.

Conclusion:

- Numeric-ratio row detection is a useful signal for locating table Y bands.
- It is not enough to isolate the table horizontally.
- We need X-position clustering or table-like word-block detection to remove unrelated paragraph text.

### Next Target

Next implementation should infer the table's horizontal region:

1. Use candidate table-like rows.
2. Inspect word X positions inside those rows.
3. Prefer numeric/table tokens and nearby labels.
4. Detect compact right-side cluster.
5. Produce an initial candidate table bounding box.

The next heuristic should focus on excluding left-side paragraph text from candidate rows.

## 2026-06-29 - Focused X-Alignment Detection

### Implementation Direction

Added geometry-based X-position clustering:

- collect word `left` positions,
- group nearby X positions using tolerance,
- keep X clusters that repeat across multiple rows.

Initial whole-page result:

- Too noisy because normal paragraph columns also create repeated X positions.

Refined approach:

- First detect candidate table-like rows using row summaries.
- Then run repeated-X detection only on those candidate rows.

### Observed Result

Focused repeated-X detection produced meaningful table-side clusters.

Observed table-like X clusters included:

- row-label/header region around `x=909` and `x=964`
- first numeric column around `x=1028`
- second numeric column around `x=1145`
- third numeric column around `x=1249`
- sum/final column around `x=1349` and `x=1408`

### Limitation

Some left-side paragraph clusters still remain because candidate rows were originally whole-page visual rows.

Conclusion:

- Repeated X alignment is a stronger general signal than numeric-only detection.
- It still needs clustering/refinement to separate table-side aligned groups from paragraph-side text.

Next target:

- Select the densest repeated-X cluster band.
- Use that X band plus candidate row Y range to propose an automatic table bounding box.

## 2026-06-29 - Automatic Candidate Box From X Band

### Implementation Direction

Added helpers to move from candidate rows to an automatic table box:

- `find_strongest_x_band()` groups repeated X clusters into horizontal bands and selects the strongest band.
- `build_bounding_box_from_x_band()` combines the selected X band with the Y range of candidate table-like rows.

### Observed Result

On the current PubTables-1M page, the strongest X band selected:

```text
left=909
right=1460
cluster_count=11
score=41
```

With padding, the proposed candidate table box was:

```text
left=889
top=740
right=1480
bottom=915
row_count=5
```

### Interpretation

This is much closer to the actual right-side table region than the earlier whole-row candidate box:

```text
left=139
top=760
right=1466
bottom=895
```

The X-band method successfully reduces contamination from left-side paragraph text.

### Limitation

The candidate box is still heuristic:

- It depends on earlier candidate-row detection.
- It may fail for very text-heavy tables with weak repeated X clusters.
- It needs visual validation by drawing the proposed box on the page.

Next target:

- Wire the automatic candidate box into the Streamlit app.
- Display the detected X band and candidate table box.
- Draw the automatic box on the original image for visual inspection.

## 2026-06-29 - Visual Validation Of Automatic Candidate Box

### Observed Result

The Streamlit app drew a green automatic candidate table-region rectangle around the small table on the PubTables-1M page.

What worked:

- The detected region correctly localized the table on the right side of the page.
- It avoided most of the abstract/body paragraph text on the left.
- It captured the table body rows and most numeric columns.
- It worked on a partially ruled/borderless presentation table where pure line detection was insufficient.

What still needs improvement:

- The detected region starts slightly below the top semantic header area.
- The `ASDM` label/header above the table is not fully included.
- The region has some padding/extra whitespace, which is acceptable for now but should be tuned later.

Research value:

- This validates that combining OCR row grouping with repeated X-alignment can localize a table-like region in a multi-column scientific document.
- It supports the hybrid approach: line cues are helpful but insufficient; OCR layout geometry adds robustness for partially ruled/borderless tables.

Next target:

- Use the automatically detected table box to filter OCR words.
- Re-run row grouping only inside the detected table box.
- Compare automatic-region row groups against manual-region row groups.

## 2026-06-29 - First Mapped Table Preview

### Implementation Direction

Added first-pass table mapping:

1. Use the automatically detected table region.
2. Filter OCR words inside that region.
3. Group filtered words into visual rows.
4. Infer repeated X anchors.
5. Merge nearby anchors into column groups.
6. Assign row words to nearest column group.
7. Display the mapped output as a Streamlit dataframe.

### Observed Result

The app produced a rough structured table preview.

What worked:

- Core table rows were mostly isolated.
- Rows corresponding to `ASCA better`, `equal`, `worse`, and `Sum` appeared in the preview.
- The output is now table-like rather than plain OCR text.

What failed or needs improvement:

- OCR noise remains, such as `t equal` instead of `equal` and `r Sum` instead of `Sum`.
- Some multi-token cells were split across neighboring columns.
- Example error:

```text
14654 | (21.8) 34,123 | (50.8)
```

Expected:

```text
14654 (21.8) | 34,123 (50.8)
```

- Extra noise rows appeared, including rows like `=.` and continuation fragments such as `(32.7) (31.0) 4`.

### Interpretation

The current method assigns each word to the nearest column center. This is too naive for multi-token cells because values and percentages have different X positions.

Research value:

- This demonstrates a common table-structure problem: cell content may contain multiple tokens spread horizontally inside one cell.
- Column assignment should use inferred column boundaries or cell regions, not only nearest anchor centers.
- Row cleanup and multi-line cell merging will also be needed.

Next target:

- Replace nearest-column-center assignment with column-boundary assignment.
- Then add simple cleanup for low-content/noise rows.

## 2026-06-30 - Direction Change To Table-Aware AI Pipeline

### Current Baseline Status

The current system is a classical/custom baseline:

- Tesseract OCR for text and word boxes.
- OpenCV/Pillow for image preprocessing and line detection.
- Custom Python heuristics for:
  - OCR row grouping,
  - table-like row detection,
  - repeated X-alignment detection,
  - automatic candidate table box detection,
  - rough column grouping,
  - rough table preview.

This baseline is useful and should be preserved for comparison.

### Key Limitations Observed

- Tesseract/OpenCV heuristics can localize a table-like region, but structure extraction remains brittle.
- Borderless and partially ruled tables require layout reasoning beyond simple line detection.
- Multi-token cells can be split across inferred columns.
- OCR noise and multi-line cells need cleanup.
- Fully robust structure recognition likely needs a table-aware model.

### Architecture Decision

Proceed with a stronger table-aware AI pipeline:

```text
Input PDF/image
-> render page as image
-> PaddleOCR for OCR/text boxes and broader document parsing
-> Table Transformer for table detection and structure recognition
-> align OCR words into predicted table cells
-> export CSV/Excel/JSON
```

Rationale:

- PaddleOCR is useful beyond tables because it supports OCR and document parsing workflows.
- Table Transformer is more directly aligned with table detection and structure recognition research.
- Table Transformer still needs OCR text, so PaddleOCR can supply the text/word boxes.
- This combination is more research-aligned than spending time on `img2table`.

### Decision About `img2table`

`img2table` is not the main next step.

Reason:

- It is a practical table-extraction library, but the research direction is better served by comparing:
  1. classical OCR/CV heuristic baseline,
  2. PaddleOCR document-AI pipeline,
  3. Table Transformer table-structure model.

`img2table` may be revisited later only as an additional practical baseline if needed.

### Next Implementation Target

Add a separate AI/document-AI path without deleting existing code:

```text
src/table_engines/
  __init__.py
  paddle_ocr_engine.py
  table_transformer_engine.py  # later
```

First step:

- Add PaddleOCR as an alternate OCR engine.
- Run it on the same test images/PDF pages.
- Compare PaddleOCR OCR boxes/text against Tesseract output.

## 2026-06-30 - PaddleOCR Engine Integration

### Implementation Direction

Added a separate PaddleOCR path without removing the existing Tesseract baseline.

New files:

```text
src/table_engines/__init__.py
src/table_engines/paddle_ocr_engine.py
```

App changes:

- Added an OCR engine selector:
  - `Tesseract`
  - `PaddleOCR`
- Tesseract remains the default baseline.
- PaddleOCR is used only when selected.

### Technical Notes

Installed PaddleOCR version observed:

```text
3.7.0
```

PaddleOCR output in this version provides recognized text lines with bounding boxes:

```text
rec_texts
rec_scores
rec_boxes
```

The wrapper normalizes PaddleOCR output into the same shape used by the existing pipeline:

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

Important detail:

- The current PaddleOCR wrapper returns line-level OCR boxes, not individual word boxes.
- This may affect row/column grouping differently from Tesseract, which returns word-level boxes.

### Next Research Comparison

Run the same documents with:

```text
Tesseract
PaddleOCR
```

Compare:

- OCR text quality,
- bounding box quality,
- table-region detection behavior,
- mapped table preview quality,
- degraded image performance.

## 2026-06-30 - PaddleOCR Table Recognition Integration

### Implementation Direction

Added a separate PaddleOCR table-aware engine:

```text
src/table_engines/paddle_table_engine.py
```

App changes:

- Added a `Table Engine` selector:
  - `None`
  - `PaddleOCR Table Recognition`
- This table-aware path is independent from the Tesseract/custom heuristic baseline.
- The app now displays:
  - Paddle detected table boxes,
  - table-box overlay,
  - parsed table dataframe from Paddle HTML,
  - raw Paddle table HTML,
  - OCR texts used inside the table.

### Observed Result On PubTables-1M Test Page

PaddleOCR table recognition detected the table region:

```text
left=905
top=687
right=1466
bottom=895
score=0.93
```

This is very close to the visual table area and slightly better at including the header area than the custom heuristic candidate box.

Paddle also returned:

```text
1 table
28 detected cells
HTML table output
parsed dataframe shape: 6 x 5
```

### Limitation

The detected table region is strong, but the table structure/content reconstruction is still imperfect.

Observed issues:

- Some cells are misplaced.
- Some text is merged into the wrong cell.
- The row label `ASCA better` was not reconstructed cleanly.
- Some table values were combined incorrectly, for example multiple values appearing in a single final column.

Research value:

- PaddleOCR table recognition performs automatic table localization well on this sample.
- Structure reconstruction still has errors on a partially ruled/borderless presentation table.
- This supports our research motivation: automatic table detection is not enough; robust table structure recognition remains difficult.

Next target:

- Visually compare:
  1. custom heuristic table box,
  2. PaddleOCR table box,
  3. PaddleOCR table dataframe,
  4. later Table Transformer output.
- Use Paddle's detected table box as a stronger table-localization baseline.

### Baseline Comparison Observation

When using:

```text
OCR Engine: PaddleOCR
Table Engine: PaddleOCR Table Recognition
```

the comparison section showed:

- custom heuristic candidate box: not detected,
- PaddleOCR table-recognition box: detected successfully.

Interpretation:

- The custom heuristic depends on word-level OCR boxes.
- The current PaddleOCR OCR wrapper returns line-level boxes, so the heuristic row/X-clustering logic does not detect a table candidate.
- PaddleOCR's table-aware model does not depend on our heuristic and directly detects the table region.

Research value:

- This clearly separates OCR-only behavior from table-aware document AI behavior.
- It supports using PaddleOCR Table Recognition as a stronger table-localization baseline.
- It also shows that a plain OCR engine, even a stronger one, is not enough for robust table structure extraction unless table-specific modeling is added.

### Updated Fair Baseline Comparison

The comparison was revised so the custom heuristic baseline always uses Tesseract word-level boxes, even when the selected OCR engine is PaddleOCR.

Fair comparison:

```text
Custom heuristic + Tesseract word boxes
vs
PaddleOCR Table Recognition
```

Observed result:

- Both methods detected the same general table region.
- The custom heuristic box was slightly tighter/lower.
- PaddleOCR Table Recognition included slightly more of the top/header area.
- Both avoided most surrounding paragraph text.

Research value:

- A classical OCR/layout heuristic can localize this table reasonably well when word-level OCR is available.
- The table-aware PaddleOCR model also localizes it well and may capture header context better.
- The next meaningful comparison is not just table localization; it is table structure quality and cell reconstruction quality.

Next target:

- Compare the custom mapped table preview against PaddleOCR's parsed table dataframe.
- Record which method better reconstructs rows, columns, headers, and multi-token cells.

## 2026-06-30 - Structure Comparison Result

### Setup

Compared:

```text
Custom heuristic mapped table
vs
PaddleOCR parsed table
```

Both methods used the same source page and detected approximately the same table region.

### Observed Table Localization

- Both methods detected the table region well.
- The custom heuristic box was tighter around the table body.
- PaddleOCR Table Recognition included more of the top/header area.

### Observed Structure Output

Custom heuristic output:

```text
6 rows x 6 columns
```

PaddleOCR parsed table:

```text
6 rows x 5 columns
```

### Custom Heuristic Strengths

- Body rows were more readable and naturally aligned.
- Rows like `ASCA better`, `equal`, `worse`, and `Sum` were recognizable.
- Numeric values were mostly placed in sensible row order.

### Custom Heuristic Weaknesses

- OCR noise remained, such as:
  - `t equal`
  - `r Sum`
  - `=.`
  - stray quote-like characters
- Some final value/percentage pairs were split across separate columns, e.g.:

```text
34,123 | (50.8)
```

instead of:

```text
34,123 (50.8)
```

### PaddleOCR Table Recognition Strengths

- Captured the header region better.
- Identified header-like structure such as:
  - `ASDM`
  - `better`
  - `equal`
  - `Worse`
  - `Sum`
- Uses a table-aware model rather than only handcrafted heuristics.

### PaddleOCR Table Recognition Weaknesses

- Body rows were not reconstructed cleanly.
- Some values were misplaced or merged.
- `ASCA better` was split incorrectly.
- Some final-column values were combined or truncated in the parsed dataframe.

### Research Interpretation

This is a useful comparison:

```text
Custom heuristics produced more readable body rows.
PaddleOCR Table Recognition captured header/table structure better.
Both still made structure errors.
```

This supports the research motivation:

```text
Automatic table localization is not enough.
Robust table structure recognition requires accurate cell reconstruction, header handling, OCR grounding, and cleanup.
```

Next target:

- Save/export both structure outputs for experiment comparison.
- Later compare against Table Transformer.

## Open Questions

- What preprocessing variant improves degraded OCR without destroying table lines?
- Should the OCR pipeline use the preprocessed image, original image, or multiple variants?
- Can table structure be detected even when text OCR quality is poor?
- How should we score OCR and table-structure quality during experiments?
- How should the pipeline decide between ruled-table extraction and borderless-table extraction?
- Can line detection and OCR-layout grouping be combined instead of choosing only one?
- How can we separate table-region words from nearby paragraph/caption text on multi-column pages?
- Should early PoC support manual table-region selection to speed up structure-recognition experiments?
- What heuristic best detects candidate table regions before using ML models?

## 2026-06-30 - Table Transformer Detection Integration

### Implementation Direction

Added Microsoft Table Transformer as a separate table engine without removing existing code:

```text
src/table_engines/table_transformer_engine.py
```

Current model:

```text
microsoft/table-transformer-detection
```

Current scope:

```text
detection only
```

The app now includes a table engine option:

```text
Table Transformer Detection
```

When selected, it:

- runs the Table Transformer detector,
- returns detected table boxes,
- draws the strongest detected box on the original page,
- includes the result in Baseline Comparison.

### PubTables-1M Test Page Result

Direct test on `input/test_table_document.pdf` page 1 at 200 DPI:

```text
image size: 1700 x 2200
detected box:
left=909
top=717
right=1465
bottom=893
score=0.9994
label=table
```

### Interpretation

This is very close to the PaddleOCR table-recognition detection result:

```text
PaddleOCR:
left=905
top=687
right=1466
bottom=895
score=0.93
```

Important finding:

```text
Table Transformer provides a strong independent table-region detection baseline.
```

Compared with custom heuristic detection:

- custom heuristic is transparent and useful for learning,
- PaddleOCR and Table Transformer are stronger table-aware AI baselines,
- all methods still need structure/cell reconstruction evaluation.

### Current Limitation

Table Transformer detection only finds the table box. It does not yet give the final table dataframe.

The next technical step is:

```text
Table Transformer detected table box
-> crop table region
-> run Table Transformer structure-recognition model
-> detect rows/columns/cells
-> align OCR words into predicted cells
```

### Deferred Work

Export/save support is intentionally deferred for now. We will later save:

- detected boxes,
- cropped tables,
- parsed tables/dataframes,
- comparison summaries,
- overlay images.

## 2026-07-01 - Table Transformer Structure Recognition Integration

### Implementation Direction

Extended the Table Transformer engine from detection-only to detection plus structure recognition:

```text
src/table_engines/table_transformer_engine.py
```

Models now used:

```text
microsoft/table-transformer-detection
microsoft/table-transformer-structure-recognition
```

Pipeline:

```text
page image
-> detect table box
-> expand/crop table region with padding
-> detect table rows and columns
-> map Tesseract word boxes into row/column intersections
-> display first-pass dataframe in Streamlit
```

The padding around the detected table box is important. Without padding, the detected table region was too tight and missed some header context. With padding, the structure model recovered a better row/column layout.

### PubTables-1M Test Page Result

Direct test on `input/test_table_document.pdf` page 1 at 200 DPI:

```text
Table Transformer detection:
left=909
top=717
right=1465
bottom=893
score=0.9994

Structure result:
rows=5
columns=6
mapped grid=5 x 6
```

Example mapped rows:

```text
® |  | better | equal | Worse | Sum *
ASCA | better | 19457 (28.9) | 12 (0.02) | 14654 (21.8) | 34,123 (50.8)
t | equal | 1158 (1.7) | 21989 (32.7) | 1024 (15) | 24,171 (36.0}
worse |  | 3755 (5.6) | 2 (0.003) | 5183 (7.7) | 8,940 (13.2)
r | Sum | =. 24370 (36.2} | 22003 (32.7) | 20861 (31.0) | 67,234 (100.0)
```

### What Worked

- Table Transformer detected the table region accurately.
- Structure recognition recovered a 5 row x 6 column grid, which is close to the visual table.
- Multi-token value cells such as `19457 (28.9)` and `34,123 (50.8)` mapped better than the earlier custom heuristic split.
- The app can now compare custom heuristic, PaddleOCR parsed table, and Table Transformer mapped table.

### What Still Fails

- OCR noise remains:
  - `t` before `equal`
  - `r` before `Sum`
  - `=.`
  - `}` instead of `)`
  - stray symbol in the header row
- The structure model gives geometry, but final extraction still depends on OCR quality.
- Header/spanning-cell semantics are not fully used yet.
- Current mapping is simple intersection of row boxes and column boxes.

### Research Interpretation

This is an important milestone:

```text
Table-aware structure prediction improves row/column geometry,
but robust table extraction still requires OCR cleanup, cell-text alignment,
header handling, and degradation-aware preprocessing.
```

Next target:

- clean OCR fragments inside detected cells,
- use detected header/spanning-cell boxes more intelligently,
- compare outputs from custom heuristic, PaddleOCR, and Table Transformer side by side.

## 2026-07-01 - Run-All Table Engine Comparison Fix

### Issue Observed

When `Table Transformer Structure Recognition` was selected, the app displayed the Structure Comparison section but PaddleOCR had not run. The PaddleOCR parsed-table column tried to read:

```text
paddle_table_result["tables"]
```

while `paddle_table_result` was still `None`, causing:

```text
TypeError: 'NoneType' object is not subscriptable
```

### Fix

Added a new table-engine mode:

```text
All Table Engines
```

This mode runs:

- PaddleOCR Table Recognition,
- Table Transformer Detection,
- Table Transformer Structure Recognition,
- custom heuristic baseline remains available through Tesseract word boxes.

Also added a guard so PaddleOCR data is only read when PaddleOCR actually ran.

### Research Value

This makes the comparison screen more useful:

```text
custom heuristic mapped table
vs
PaddleOCR parsed table
vs
Table Transformer mapped table
```

Now all three can be viewed in one run for fair visual comparison.

## 2026-07-01 - First All-Engines Comparison Result

### Setup

Selected:

```text
Table Engine: All Table Engines
OCR baseline for custom heuristic: Tesseract word boxes
Input: PubTables-1M sample page from input/test_table_document.pdf
```

The app displayed both:

- Baseline Comparison for table-region localization,
- Structure Comparison for extracted/mapped table content.

### Table Region Detection Results

Custom heuristic candidate box:

```text
left=889
top=740
right=1480
bottom=915
row_count=5
x_cluster_count=11
score=41
```

PaddleOCR table-recognition box:

```text
label=table
score=0.9328
left=905
top=687
right=1466
bottom=895
```

Table Transformer detection box:

```text
label=table
score=0.9994
left=909
top=717
right=1465
bottom=893
```

### Localization Interpretation

All three approaches detected the same table region.

Important observations:

- The custom heuristic is slightly lower/tighter and misses some top header context.
- PaddleOCR includes more of the upper table/header region.
- Table Transformer is tight and high-confidence.
- PaddleOCR and Table Transformer are very close horizontally and vertically.

This confirms that for the clean PubTables sample, table localization is mostly solved by the model-based approaches.

### Structure Comparison Results

Custom heuristic mapped table:

```text
Rows: 6
Columns: 6
```

PaddleOCR parsed table:

```text
Rows: 6
Columns: 5
```

Table Transformer mapped table:

```text
Rows: 5
Columns: 6
```

### Structure Interpretation

Custom heuristic:

- body rows are readable,
- row labels and numeric values are mostly preserved,
- still has OCR noise such as `t equal`, `r Sum`, and stray rows,
- final value/percentage cells can be split.

PaddleOCR:

- captures some header information,
- parsed table has fewer columns than expected,
- body values are more misplaced/merged in the visible result.

Table Transformer:

- recovers a 5 x 6 grid close to the visual table,
- keeps multi-token value cells together better, such as value plus percentage,
- still has OCR noise because the current text filling uses Tesseract word boxes,
- header/spanning-cell semantics are not fully used yet.

### Research Interpretation

This is a strong paper result:

```text
Model-based detectors localize the table well,
but structure reconstruction quality still varies across engines.
```

For this sample:

- localization is good across all three,
- structure extraction remains imperfect,
- OCR cleanup and cell-alignment logic are still necessary,
- no single baseline fully solves the table.

This supports the research thesis that robust table extraction needs:

```text
table detection
+ structure recognition
+ OCR grounding
+ post-processing/cleanup
+ degradation-aware evaluation
```

### Next Target

Improve the Table Transformer mapped table by:

- removing isolated OCR noise tokens such as `t`, `r`, `=.`, and stray symbols,
- using column header and spanning-cell boxes instead of ignoring them,
- considering PaddleOCR text as an alternate OCR source for filling Table Transformer cells,
- later saving all comparison outputs for repeatable experiments.

## 2026-07-01 - Degraded Table All-Engines Observation

### Setup

Input appears to be the degraded spreadsheet/table image:

```text
input/degraded_table_01.png
```

The image is low-resolution and blurry. It contains a spreadsheet-like table.

### Baseline Comparison Result

Custom heuristic:

```text
No custom heuristic candidate box was detected.
```

PaddleOCR table recognition:

```text
Returned an error in the app:
"src property must be a valid json object"
```

This means the current PaddleOCR table-recognition display path did not produce a clean usable table box/result for this image.

Table Transformer detection:

```text
label=table
score=0.9705
left=28
top=38
right=505
bottom=282
```

### Structure Comparison Result

Custom heuristic mapped table:

```text
Rows: 0
Columns: 0
No custom mapped table was produced.
```

PaddleOCR parsed table:

```text
Not selected / not usable in this run because of the PaddleOCR result issue above.
```

Table Transformer mapped table:

```text
Rows: 11
Columns: 8
```

However, the mapped table text was mostly incorrect. Many cells contained wrong fragments such as:

```text
arb
Time
hm
MOR
owe?
meres
baie
rm
twee
vem
```

### OCR Row Group Observation

The OCR row grouping section produced useful row-level evidence even though text recognition was degraded.

Examples:

```text
Row 4: 1 Stona Sales § NcRagion Goal Nof Goal
Row 5: 2 Store:1 511.309 A 15,000 14.00%
Row 6: 1 Store-2 $15.567 A 20,000 77.82%
Row 7: Stone-3 $16,872 C 10,000 108.72%
Row 8: Store-4 512.566 8 12.000 104.72%
Row 9: Store-5 $14,307 C 15,000 95.30%
Row 10: 3 5toro-6 513.129C 12,000 109.41%
Row 11: Store-7 $25,456 8 15.000 111.04%
Row 12: 3 Stowe-B $11.214 A 10,000 132.34%
Row 13: 10 Total 554,377
```

### Interpretation

This degraded case shows an important split:

```text
Table localization can work even when cell text extraction fails.
```

Table Transformer detected the table area with high confidence, but the final mapped table was wrong because OCR/text quality inside cells was poor.

At the same time, OCR row grouping preserved useful row-level patterns:

- many store rows were detected,
- numeric values and percentages were partially readable,
- row order was mostly visible,
- OCR text was noisy but not completely useless.

### Research Value

This is a strong degraded-document result for the paper:

```text
Detection success does not imply structure extraction success.
```

For degraded tables, we may need:

- better image enhancement before OCR,
- stronger OCR for cropped table regions,
- OCR confidence filtering,
- row-level fallback extraction when cell mapping fails,
- a way to compare row-group quality separately from final cell-grid quality.

### Next Ideas

For degraded inputs, test:

- upscale before OCR,
- sharpen/contrast enhancement,
- run OCR on the detected Table Transformer crop instead of the whole image,
- compare Tesseract vs PaddleOCR OCR text for the cropped table,
- use OCR row groups as a fallback when cell-level mapping is unreliable.

## 2026-07-01 - Batch Detection Evaluation Layer

### Motivation

For large datasets, Streamlit should not be the main evaluation runner. Streamlit remains useful for visual inspection, but metrics need a repeatable non-UI flow.

Added a separate batch-evaluation layer:

```text
src/detection_service.py
src/detection_metrics.py
evaluation/run_detection_eval.py
evaluation/annotations.example.csv
evaluation/README.md
```

### What The Batch Flow Does

The CLI script can:

- scan an input file or directory,
- render PDF pages,
- run selected detectors,
- normalize all detector outputs into one prediction format,
- save `predictions.csv`,
- compare predictions against ground-truth boxes if annotations are supplied,
- compute:
  - Precision@IoU50,
  - Recall@IoU50,
  - F1@IoU50,
  - mean IoU for matched true positives,
  - mean best IoU per ground-truth table,
  - AP50,
  - average runtime.

Current detector methods:

```text
custom_heuristic
paddleocr_table
table_transformer
```

### Smoke Test Command

```bash
python evaluation/run_detection_eval.py \
  --input input \
  --annotations evaluation/annotations.example.csv \
  --output-dir evaluation/results_smoke_all_methods \
  --methods all \
  --only-annotated-pages
```

The `--only-annotated-pages` flag is important for small/manual annotation subsets. Otherwise predictions on unannotated pages may be counted as false positives.

### Smoke Test Result

Using the example annotations:

```text
custom_heuristic:
precision=1.0
recall=0.5
f1=0.6667
mean_best_iou_per_gt=0.3265
average_runtime=2.145s

paddleocr_table:
precision=1.0
recall=0.5
f1=0.6667
mean_best_iou_per_gt=0.5
average_runtime=18.347s

table_transformer:
precision=1.0
recall=1.0
f1=1.0
mean_best_iou_per_gt=0.9193
average_runtime=1.541s
```

Predictions from the smoke run:

```text
degraded_table_01.png page 1:
table_transformer detected box left=28 top=38 right=505 bottom=282 score=0.9705

test_table_document.pdf page 1:
custom_heuristic detected box left=889 top=740 right=1480 bottom=915 score=41
paddleocr_table detected box left=905 top=687 right=1466 bottom=895 score=0.9328
table_transformer detected box left=909 top=717 right=1465 bottom=893 score=0.9994
```

### Important Caveat

`annotations.example.csv` is only a smoke-test annotation file. Its boxes are based on our current visual/model observations, not a formal ground-truth dataset.

For publishable metrics, use:

- dataset-provided annotations, or
- manually verified ground-truth table boxes,
- complete annotations for all evaluated pages.

### Next Evaluation Work

- Add more manually verified pages.
- Decide whether to evaluate all pages or only annotated pages for each experiment.
- Add dataset-specific annotation converters for PubTables-1M, ICDAR, Marmot, FinTabNet, or TableBank.
- Keep Streamlit as visual inspection only; use the batch script for metrics.

## 2026-07-01 - Clarification: PaddleOCR OCR vs PaddleOCR Table Detection

### Question

The batch metric row for PaddleOCR table detection was:

```text
paddleocr_table,0.5,2,1,1,0,1,1.0,0.5,0.6667,1.0,0.5,0.5,17.55
```

At first glance this looked confusing because in the Streamlit degraded-table test, OCR row groups showed useful row text.

### Important Distinction

There are three different capabilities being discussed:

```text
PaddleOCR/plain OCR or OCR row grouping
-> reads text/lines/rows

PaddleOCR table recognition
-> detects table regions and tries to reconstruct table structure/HTML

Table Transformer detection
-> detects table bounding boxes
```

The batch method named:

```text
paddleocr_table
```

means PaddleOCR/PaddleX table-recognition detection, not plain OCR text extraction.

### What Happened On The Degraded Spreadsheet

In Streamlit, the degraded spreadsheet showed useful OCR row groups, for example:

```text
Store-2 ... 20,000 77.82%
Store-5 ... 15,000 95.30%
Total ...
```

But the PaddleOCR table-recognition section did not produce a clean usable table box. It showed an app/result error:

```text
"src property must be a valid json object"
```

In the batch run, PaddleOCR table recognition similarly did not return a table prediction for `degraded_table_01.png`.

The degraded spreadsheet was detected by:

```text
table_transformer
```

with:

```text
left=28
top=38
right=505
bottom=282
score=0.9705
```

### Why PaddleOCR Metrics Look Like This

The example annotations contain two real tables:

```text
1. test_table_document.pdf page 1
2. degraded_table_01.png page 1
```

PaddleOCR table recognition detected:

```text
clean PubTables table: yes
degraded spreadsheet table: no
```

Therefore:

```text
ground_truth_count = 2
prediction_count = 1
true_positives = 1
false_positives = 0
false_negatives = 1
precision = 1.0
recall = 0.5
f1 = 0.6667
```

### Research Interpretation

This is an important research finding:

```text
OCR readability and table detection are different tasks.
```

On degraded images:

- OCR row grouping can preserve useful row-level text patterns,
- PaddleOCR table-recognition detection may still miss the table,
- Table Transformer can detect the table region,
- detecting a table does not guarantee correct table data extraction.

This supports evaluating table detection separately from:

- OCR text quality,
- row grouping quality,
- table structure recognition,
- final cell-value accuracy.
