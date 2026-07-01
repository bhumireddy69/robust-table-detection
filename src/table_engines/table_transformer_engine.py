from __future__ import annotations

from functools import lru_cache

import torch
from PIL import Image


DETECTION_MODEL_NAME = "microsoft/table-transformer-detection"
STRUCTURE_MODEL_NAME = "microsoft/table-transformer-structure-recognition"


@lru_cache(maxsize=1)
def _get_detection_model():
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection

    processor = AutoImageProcessor.from_pretrained(DETECTION_MODEL_NAME)
    model = TableTransformerForObjectDetection.from_pretrained(DETECTION_MODEL_NAME)
    model.eval()
    return processor, model


def detect_tables(image: Image.Image, threshold: float = 0.5) -> list[dict]:
    processor, model = _get_detection_model()
    rgb_image = image.convert("RGB")

    return _detect_objects(
        image=rgb_image,
        processor=processor,
        model=model,
        threshold=threshold,
    )


@lru_cache(maxsize=1)
def _get_structure_model():
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection

    processor = AutoImageProcessor.from_pretrained(STRUCTURE_MODEL_NAME)
    model = TableTransformerForObjectDetection.from_pretrained(STRUCTURE_MODEL_NAME)
    model.eval()
    return processor, model


def recognize_structure(
    image: Image.Image,
    table_box: dict | None,
    threshold: float = 0.6,
    crop_padding: int = 50,
) -> dict:
    processor, model = _get_structure_model()

    if table_box is None:
        crop_left = 0
        crop_top = 0
        crop_right = image.width
        crop_bottom = image.height
    else:
        crop_left = max(0, int(table_box["left"]) - crop_padding)
        crop_top = max(0, int(table_box["top"]) - crop_padding)
        crop_right = min(image.width, int(table_box["right"]) + crop_padding)
        crop_bottom = min(image.height, int(table_box["bottom"]) + crop_padding)

    table_crop = image.crop((crop_left, crop_top, crop_right, crop_bottom)).convert("RGB")
    crop_boxes = _detect_objects(
        image=table_crop,
        processor=processor,
        model=model,
        threshold=threshold,
    )
    page_boxes = [_offset_box(box, crop_left, crop_top) for box in crop_boxes]

    return {
        "table_box": table_box,
        "crop_box": {
            "left": crop_left,
            "top": crop_top,
            "right": crop_right,
            "bottom": crop_bottom,
        },
        "all_boxes": page_boxes,
        "rows": _boxes_by_label(page_boxes, "table row", sort_key="top"),
        "columns": _boxes_by_label(page_boxes, "table column", sort_key="left"),
        "column_headers": _boxes_by_label(page_boxes, "table column header", sort_key="top"),
        "projected_row_headers": _boxes_by_label(page_boxes, "table projected row header", sort_key="top"),
        "spanning_cells": _boxes_by_label(page_boxes, "table spanning cell", sort_key="top"),
        "table_crop": table_crop,
    }


def build_grid_from_words(structure: dict, words: list[dict]) -> list[list[str]]:
    rows = structure["rows"]
    columns = structure["columns"]

    if not rows or not columns:
        return []

    grid = []

    for row in rows:
        grid_row = []

        for column in columns:
            cell_words = [
                word
                for word in words
                if _word_center_inside_box(word, _intersection_box(row, column))
            ]
            cell_words.sort(key=lambda word: (word["left"], word["top"]))
            grid_row.append(" ".join(str(word["text"]) for word in cell_words))

        grid.append(grid_row)

    return grid


def _detect_objects(image: Image.Image, processor, model, threshold: float) -> list[dict]:
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[image.height, image.width]])
    results = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=target_sizes,
    )[0]

    boxes = []
    id_to_label = model.config.id2label

    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        label = id_to_label[int(label_id)]
        left, top, right, bottom = box.tolist()
        boxes.append(
            {
                "label": label,
                "score": float(score),
                "left": round(left),
                "top": round(top),
                "right": round(right),
                "bottom": round(bottom),
            }
        )

    boxes.sort(key=lambda item: item["score"], reverse=True)
    return boxes


def _offset_box(box: dict, offset_left: int, offset_top: int) -> dict:
    return {
        **box,
        "left": box["left"] + offset_left,
        "top": box["top"] + offset_top,
        "right": box["right"] + offset_left,
        "bottom": box["bottom"] + offset_top,
    }


def _boxes_by_label(boxes: list[dict], label: str, sort_key: str) -> list[dict]:
    selected_boxes = [box for box in boxes if box["label"] == label]
    selected_boxes.sort(key=lambda box: box[sort_key])
    return selected_boxes


def _intersection_box(row: dict, column: dict) -> dict:
    return {
        "left": max(row["left"], column["left"]),
        "top": max(row["top"], column["top"]),
        "right": min(row["right"], column["right"]),
        "bottom": min(row["bottom"], column["bottom"]),
    }


def _word_center_inside_box(word: dict, box: dict) -> bool:
    if box["left"] >= box["right"] or box["top"] >= box["bottom"]:
        return False

    center_x = word["left"] + word["width"] / 2
    center_y = word["top"] + word["height"] / 2

    return (
        box["left"] <= center_x <= box["right"]
        and box["top"] <= center_y <= box["bottom"]
    )
