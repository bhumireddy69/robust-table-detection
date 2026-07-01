from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from src import ocr as tesseract_ocr
from src.image_processing import preprocess_for_ocr
from src.table_engines import paddle_table_engine
from src.table_engines import table_transformer_engine
from src.table_structure import (
    build_bounding_box_from_x_band,
    find_repeated_x_positions,
    find_strongest_x_band,
    find_table_like_rows,
    get_rows_by_indexes,
    group_words_into_rows,
)


METHOD_CUSTOM = "custom_heuristic"
METHOD_PADDLE = "paddleocr_table"
METHOD_TABLE_TRANSFORMER = "table_transformer"
ALL_METHODS = [METHOD_CUSTOM, METHOD_PADDLE, METHOD_TABLE_TRANSFORMER]


def run_detectors(
    image: Image.Image,
    file_name: str,
    page_number: int,
    methods: list[str] | None = None,
    image_path: Path | None = None,
) -> list[dict]:
    selected_methods = methods or ALL_METHODS
    detections = []

    if METHOD_CUSTOM in selected_methods:
        detections.extend(
            detect_with_custom_heuristic(
                image=image,
                file_name=file_name,
                page_number=page_number,
            )
        )

    if METHOD_PADDLE in selected_methods:
        detections.extend(
            detect_with_paddleocr(
                image_path=image_path,
                file_name=file_name,
                page_number=page_number,
            )
        )

    if METHOD_TABLE_TRANSFORMER in selected_methods:
        detections.extend(
            detect_with_table_transformer(
                image=image,
                file_name=file_name,
                page_number=page_number,
            )
        )

    return detections


def detect_with_custom_heuristic(
    image: Image.Image,
    file_name: str,
    page_number: int,
) -> list[dict]:
    started_at = time.perf_counter()
    preprocessed_image = preprocess_for_ocr(image)
    words = tesseract_ocr.extract_words(preprocessed_image)

    grouped_rows = group_words_into_rows(words)
    table_like_rows = find_table_like_rows(grouped_rows)
    table_like_row_indexes = [row["row_index"] for row in table_like_rows]
    focused_rows = get_rows_by_indexes(grouped_rows, table_like_row_indexes)
    x_clusters = find_repeated_x_positions(focused_rows, x_tolerance=14, min_occurrences=2)
    x_band = find_strongest_x_band(x_clusters, max_gap=90, min_clusters=3)
    box = build_bounding_box_from_x_band(table_like_rows, x_band, padding=20)
    runtime_seconds = time.perf_counter() - started_at

    if box is None:
        return []

    return [
        _normalize_detection(
            file_name=file_name,
            page_number=page_number,
            method=METHOD_CUSTOM,
            box=box,
            score=float(box.get("score") or 0),
            runtime_seconds=runtime_seconds,
        )
    ]


def detect_with_paddleocr(
    image_path: Path | None,
    file_name: str,
    page_number: int,
) -> list[dict]:
    if image_path is None:
        return [
            _error_detection(
                file_name=file_name,
                page_number=page_number,
                method=METHOD_PADDLE,
                error="image_path is required for PaddleOCR table detection",
            )
        ]

    started_at = time.perf_counter()

    try:
        result = paddle_table_engine.recognize_tables(str(image_path))
        runtime_seconds = time.perf_counter() - started_at
    except Exception as exc:
        return [
            _error_detection(
                file_name=file_name,
                page_number=page_number,
                method=METHOD_PADDLE,
                error=str(exc),
            )
        ]

    detections = []

    for box in result["table_boxes"]:
        detections.append(
            _normalize_detection(
                file_name=file_name,
                page_number=page_number,
                method=METHOD_PADDLE,
                box=box,
                score=float(box.get("score") or 0),
                runtime_seconds=runtime_seconds,
            )
        )

    return detections


def detect_with_table_transformer(
    image: Image.Image,
    file_name: str,
    page_number: int,
) -> list[dict]:
    started_at = time.perf_counter()

    try:
        boxes = table_transformer_engine.detect_tables(image)
        runtime_seconds = time.perf_counter() - started_at
    except Exception as exc:
        return [
            _error_detection(
                file_name=file_name,
                page_number=page_number,
                method=METHOD_TABLE_TRANSFORMER,
                error=str(exc),
            )
        ]

    detections = []

    for box in boxes:
        detections.append(
            _normalize_detection(
                file_name=file_name,
                page_number=page_number,
                method=METHOD_TABLE_TRANSFORMER,
                box=box,
                score=float(box.get("score") or 0),
                runtime_seconds=runtime_seconds,
            )
        )

    return detections


def _normalize_detection(
    file_name: str,
    page_number: int,
    method: str,
    box: dict,
    score: float,
    runtime_seconds: float,
) -> dict:
    return {
        "file_name": file_name,
        "page_number": page_number,
        "method": method,
        "label": box.get("label", "table"),
        "score": score,
        "left": int(box["left"]),
        "top": int(box["top"]),
        "right": int(box["right"]),
        "bottom": int(box["bottom"]),
        "runtime_seconds": runtime_seconds,
        "error": "",
    }


def _error_detection(
    file_name: str,
    page_number: int,
    method: str,
    error: str,
) -> dict:
    return {
        "file_name": file_name,
        "page_number": page_number,
        "method": method,
        "label": "",
        "score": 0.0,
        "left": "",
        "top": "",
        "right": "",
        "bottom": "",
        "runtime_seconds": 0.0,
        "error": error,
    }
