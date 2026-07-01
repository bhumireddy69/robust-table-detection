from __future__ import annotations

from functools import lru_cache
from io import StringIO

import pandas as pd
from PIL import Image


@lru_cache(maxsize=1)
def _get_pipeline():
    from paddleocr import TableRecognitionPipelineV2

    return TableRecognitionPipelineV2(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=True,
        use_ocr_model=True,
    )


def recognize_tables(image_path: str) -> dict:
    pipeline = _get_pipeline()
    results = pipeline.predict(image_path)

    if not results:
        return {
            "layout_boxes": [],
            "table_boxes": [],
            "tables": [],
        }

    result = results[0]
    layout_boxes = _normalize_layout_boxes(result["layout_det_res"]["boxes"])
    table_boxes = [box for box in layout_boxes if box["label"] == "table"]
    tables = [_normalize_table_result(table) for table in result["table_res_list"]]

    return {
        "layout_boxes": layout_boxes,
        "table_boxes": table_boxes,
        "tables": tables,
    }


def _normalize_layout_boxes(boxes: list[dict]) -> list[dict]:
    normalized_boxes = []

    for box in boxes:
        left, top, right, bottom = [float(value) for value in box["coordinate"]]
        normalized_boxes.append(
            {
                "label": box["label"],
                "score": float(box["score"]),
                "left": round(left),
                "top": round(top),
                "right": round(right),
                "bottom": round(bottom),
            }
        )

    return normalized_boxes


def _normalize_table_result(table: dict) -> dict:
    html = table["pred_html"]
    dataframes = _html_to_dataframes(html)

    return {
        "html": html,
        "dataframes": dataframes,
        "cell_count": len(table["cell_box_list"]),
        "ocr_texts": list(table["table_ocr_pred"]["rec_texts"]),
        "ocr_boxes": table["table_ocr_pred"]["rec_boxes"],
    }


def _html_to_dataframes(html: str) -> list[pd.DataFrame]:
    try:
        return pd.read_html(StringIO(html))
    except ValueError:
        return []
