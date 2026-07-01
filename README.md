# Table Structure PoC

Proof of concept for table detection in scanned/degraded documents.

Current focus:

```text
table detection only
```

We are not evaluating table cell values yet.

## 1. Batch Detection Metrics

Use this for repeatable evaluation and datasets.

Activate environment:

```bash
cd /Users/chaitanya/Documents/table-structure
source .venv/bin/activate
```

Run all detectors on the current annotated sample files:

```bash
python evaluation/run_detection_eval.py \
  --input input \
  --annotations evaluation/annotations.example.csv \
  --output-dir evaluation/results \
  --methods all \
  --only-annotated-pages
```

View metrics:

```bash
cat evaluation/results/metrics_summary.csv
```

View predictions:

```bash
cat evaluation/results/predictions.csv
```

Current detectors:

```text
custom_heuristic    = Tesseract OCR boxes + our table-region heuristic
paddleocr_table     = PaddleOCR/PaddleX table detection
table_transformer   = Microsoft Table Transformer detection
```

Metrics calculated when annotations are provided:

```text
Precision@IoU50
Recall@IoU50
F1@IoU50
Mean IoU
AP50
Runtime
```

Annotation format:

```csv
file_name,page_number,gt_left,gt_top,gt_right,gt_bottom
```

Example annotations are in:

```text
evaluation/annotations.example.csv
```

## 2. Streamlit Visual App

Use this for visual debugging and manual inspection.

Run:

```bash
cd /Users/chaitanya/Documents/table-structure
source .venv/bin/activate
streamlit run app.py
```

In the app, use:

```text
Table Engine -> All Table Engines
```

This shows:

```text
custom heuristic box
PaddleOCR table box
Table Transformer box
```

It also shows structure previews, but those are not the current evaluation target.

## Notes

- Batch evaluation and Streamlit are separate flows.
- Streamlit is for visual checking.
- Batch evaluation is for metrics and larger datasets.
- `annotations.example.csv` is only for smoke testing. Use verified ground-truth boxes for real research results.
