# Batch Table Detection Evaluation

This folder is for non-Streamlit evaluation.

The batch flow runs table detectors on PDFs/images, saves normalized predictions, and optionally computes detection metrics when ground-truth boxes are available.

## Detectors

Current methods:

```text
custom_heuristic
paddleocr_table
table_transformer
```

## Annotation Format

Ground-truth CSV columns:

```text
file_name,page_number,gt_left,gt_top,gt_right,gt_bottom
```

Example:

```csv
file_name,page_number,gt_left,gt_top,gt_right,gt_bottom
test_table_document.pdf,1,905,687,1466,895
degraded_table_01.png,1,28,38,505,282
```

When `--input` is a directory, `file_name` should be relative to that directory.

## Run Without Ground Truth

This only saves predictions and runtime:

```bash
python evaluation/run_detection_eval.py \
  --input input \
  --output-dir evaluation/results
```

## Run With Ground Truth

This saves predictions and metrics:

```bash
python evaluation/run_detection_eval.py \
  --input input \
  --annotations evaluation/annotations.example.csv \
  --output-dir evaluation/results \
  --only-annotated-pages
```

Use `--only-annotated-pages` when your annotation CSV is a small/manual subset. Without it, predictions on unannotated pages will count as false positives.

## Output Files

```text
evaluation/results/predictions.csv
evaluation/results/metrics_summary.csv
evaluation/results/errors.csv
evaluation/results/rendered_pages/
```

## Metrics

When annotations are supplied, the script computes:

```text
Precision@IoU50
Recall@IoU50
F1@IoU50
Mean IoU for matched true positives
Mean best IoU per ground-truth box
AP50
Average detector runtime
```

For the real dataset, replace `annotations.example.csv` with dataset-provided or manually created ground-truth table boxes.
