from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection_metrics import (  # noqa: E402
    add_best_iou_to_predictions,
    build_ground_truth_boxes,
    evaluate_method_predictions,
)
from src.detection_service import (  # noqa: E402
    ALL_METHODS,
    METHOD_CUSTOM,
    METHOD_PADDLE,
    METHOD_TABLE_TRANSFORMER,
    run_detectors,
)
from src.document_loader import SUPPORTED_TYPES, load_document_pages  # noqa: E402


PREDICTION_FIELDS = [
    "file_name",
    "page_number",
    "method",
    "label",
    "score",
    "left",
    "top",
    "right",
    "bottom",
    "runtime_seconds",
    "best_iou",
    "is_true_positive_iou50",
    "error",
]

SUMMARY_FIELDS = [
    "method",
    "iou_threshold",
    "ground_truth_count",
    "prediction_count",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "mean_iou_matched",
    "mean_best_iou_per_gt",
    "ap50",
    "average_runtime_seconds",
]


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered_dir = output_dir / "rendered_pages"
    rendered_dir.mkdir(exist_ok=True)

    methods = _parse_methods(args.methods)
    document_paths = _find_document_paths(input_path)
    annotations = _read_csv(args.annotations) if args.annotations else []
    annotated_page_keys = _annotated_page_keys(annotations)

    predictions = []

    for document_path in document_paths:
        file_name = _file_name_for_csv(document_path, input_path)
        pages = load_document_pages(document_path.read_bytes(), document_path.name, dpi=args.dpi)

        for page_index, page_image in enumerate(pages, start=1):
            if args.only_annotated_pages and (file_name, page_index) not in annotated_page_keys:
                continue

            rendered_path = rendered_dir / _rendered_page_name(file_name, page_index)
            page_image.save(rendered_path)

            print(f"Running detectors for {file_name} page {page_index}...")
            page_predictions = run_detectors(
                image=page_image,
                file_name=file_name,
                page_number=page_index,
                methods=methods,
                image_path=rendered_path,
            )
            predictions.extend(page_predictions)

    valid_predictions = [
        prediction
        for prediction in predictions
        if prediction.get("error") == ""
    ]

    ground_truth_boxes = build_ground_truth_boxes(annotations) if annotations else []

    if ground_truth_boxes:
        predictions_for_csv = add_best_iou_to_predictions(predictions, ground_truth_boxes)
        valid_predictions_for_metrics = [
            prediction
            for prediction in predictions_for_csv
            if prediction.get("error") == ""
        ]
        summary_rows = [
            evaluate_method_predictions(
                method=method,
                predictions=[
                    prediction
                    for prediction in valid_predictions_for_metrics
                    if prediction["method"] == method
                ],
                ground_truth_boxes=ground_truth_boxes,
                iou_threshold=args.iou_threshold,
            )
            for method in methods
        ]
        _write_csv(output_dir / "metrics_summary.csv", SUMMARY_FIELDS, summary_rows)
    else:
        predictions_for_csv = predictions
        print("No annotations provided. Skipping IoU/precision/recall/F1/AP metrics.")

    _write_csv(output_dir / "predictions.csv", PREDICTION_FIELDS, predictions_for_csv)
    _write_csv(output_dir / "errors.csv", PREDICTION_FIELDS, [row for row in predictions_for_csv if row.get("error")])

    print(f"Predictions written to: {output_dir / 'predictions.csv'}")

    if ground_truth_boxes:
        print(f"Metrics written to: {output_dir / 'metrics_summary.csv'}")

    if len(valid_predictions) != len(predictions):
        print(f"Detector errors written to: {output_dir / 'errors.csv'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch table-detection evaluation.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input file or directory containing PDFs/images.",
    )
    parser.add_argument(
        "--annotations",
        help="Optional CSV with ground-truth table boxes.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/results",
        help="Directory for predictions and metrics CSV files.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated methods: custom_heuristic,paddleocr_table,table_transformer or all.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PDF render DPI.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for TP/FP/FN metrics.",
    )
    parser.add_argument(
        "--only-annotated-pages",
        action="store_true",
        help="Only run pages that appear in the annotation CSV.",
    )
    return parser.parse_args()


def _parse_methods(methods_text: str) -> list[str]:
    if methods_text == "all":
        return ALL_METHODS

    aliases = {
        "custom": METHOD_CUSTOM,
        "custom_heuristic": METHOD_CUSTOM,
        "paddle": METHOD_PADDLE,
        "paddleocr": METHOD_PADDLE,
        "paddleocr_table": METHOD_PADDLE,
        "transformer": METHOD_TABLE_TRANSFORMER,
        "table_transformer": METHOD_TABLE_TRANSFORMER,
    }

    methods = []

    for method_text in methods_text.split(","):
        method_key = method_text.strip()

        if method_key not in aliases:
            raise ValueError(f"Unknown method: {method_key}")

        methods.append(aliases[method_key])

    return methods


def _find_document_paths(input_path: Path) -> list[Path]:
    supported_suffixes = {f".{file_type}" for file_type in SUPPORTED_TYPES}

    if input_path.is_file():
        return [input_path]

    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_suffixes
    )


def _file_name_for_csv(document_path: Path, input_path: Path) -> str:
    if input_path.is_dir():
        return str(document_path.relative_to(input_path))

    return document_path.name


def _rendered_page_name(file_name: str, page_number: int) -> str:
    safe_name = file_name.replace("/", "__").replace("\\", "__")
    return f"{Path(safe_name).stem}_page_{page_number}.png"


def _read_csv(path_text: str) -> list[dict]:
    with Path(path_text).expanduser().open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _annotated_page_keys(annotations: list[dict]) -> set[tuple[str, int]]:
    return {
        (row["file_name"], int(row["page_number"]))
        for row in annotations
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
