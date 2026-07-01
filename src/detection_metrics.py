from __future__ import annotations


def calculate_iou(box_a: dict, box_b: dict) -> float:
    intersection_left = max(float(box_a["left"]), float(box_b["left"]))
    intersection_top = max(float(box_a["top"]), float(box_b["top"]))
    intersection_right = min(float(box_a["right"]), float(box_b["right"]))
    intersection_bottom = min(float(box_a["bottom"]), float(box_b["bottom"]))

    intersection_width = max(0.0, intersection_right - intersection_left)
    intersection_height = max(0.0, intersection_bottom - intersection_top)
    intersection_area = intersection_width * intersection_height

    area_a = _box_area(box_a)
    area_b = _box_area(box_b)
    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def evaluate_predictions(
    predictions: list[dict],
    ground_truth_boxes: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    methods = sorted({prediction["method"] for prediction in predictions})
    summaries = []

    for method in methods:
        method_predictions = [
            prediction
            for prediction in predictions
            if prediction["method"] == method
        ]
        summary = evaluate_method_predictions(
            method=method,
            predictions=method_predictions,
            ground_truth_boxes=ground_truth_boxes,
            iou_threshold=iou_threshold,
        )
        summaries.append(summary)

    return {
        "iou_threshold": iou_threshold,
        "summaries": summaries,
    }


def evaluate_method_predictions(
    method: str,
    predictions: list[dict],
    ground_truth_boxes: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    matched_gt_ids = set()
    true_positive_count = 0
    false_positive_count = 0
    matched_ious = []

    sorted_predictions = sorted(
        predictions,
        key=lambda prediction: float(prediction.get("score") or 0),
        reverse=True,
    )

    precision_points = []
    recall_points = []

    for prediction in sorted_predictions:
        best_gt, best_iou = _best_unmatched_ground_truth(
            prediction,
            ground_truth_boxes,
            matched_gt_ids,
        )

        if best_gt is not None and best_iou >= iou_threshold:
            true_positive_count += 1
            matched_gt_ids.add(best_gt["_gt_id"])
            matched_ious.append(best_iou)
        else:
            false_positive_count += 1

        precision = _safe_divide(true_positive_count, true_positive_count + false_positive_count)
        recall = _safe_divide(true_positive_count, len(ground_truth_boxes))
        precision_points.append(precision)
        recall_points.append(recall)

    false_negative_count = len(ground_truth_boxes) - true_positive_count
    precision = _safe_divide(true_positive_count, true_positive_count + false_positive_count)
    recall = _safe_divide(true_positive_count, true_positive_count + false_negative_count)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return {
        "method": method,
        "iou_threshold": iou_threshold,
        "ground_truth_count": len(ground_truth_boxes),
        "prediction_count": len(predictions),
        "true_positives": true_positive_count,
        "false_positives": false_positive_count,
        "false_negatives": false_negative_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou_matched": _mean(matched_ious),
        "mean_best_iou_per_gt": _mean_best_iou_per_gt(predictions, ground_truth_boxes),
        "ap50": _average_precision(precision_points, recall_points),
        "average_runtime_seconds": _mean(
            [
                float(prediction["runtime_seconds"])
                for prediction in predictions
                if prediction.get("runtime_seconds") not in {"", None}
            ]
        ),
    }


def build_ground_truth_boxes(rows: list[dict]) -> list[dict]:
    ground_truth_boxes = []

    for index, row in enumerate(rows):
        ground_truth_boxes.append(
            {
                "_gt_id": index,
                "file_name": row["file_name"],
                "page_number": int(row["page_number"]),
                "left": float(row["gt_left"]),
                "top": float(row["gt_top"]),
                "right": float(row["gt_right"]),
                "bottom": float(row["gt_bottom"]),
            }
        )

    return ground_truth_boxes


def add_best_iou_to_predictions(predictions: list[dict], ground_truth_boxes: list[dict]) -> list[dict]:
    enriched_predictions = []

    for prediction in predictions:
        if prediction.get("error"):
            enriched_predictions.append(
                {
                    **prediction,
                    "best_iou": "",
                    "is_true_positive_iou50": "",
                }
            )
            continue

        matching_ground_truth = _same_page_ground_truth(prediction, ground_truth_boxes)
        best_iou = 0.0

        for gt_box in matching_ground_truth:
            best_iou = max(best_iou, calculate_iou(prediction, gt_box))

        enriched_predictions.append(
            {
                **prediction,
                "best_iou": best_iou,
                "is_true_positive_iou50": best_iou >= 0.5,
            }
        )

    return enriched_predictions


def _best_unmatched_ground_truth(
    prediction: dict,
    ground_truth_boxes: list[dict],
    matched_gt_ids: set[int],
) -> tuple[dict | None, float]:
    best_gt = None
    best_iou = 0.0

    for gt_box in _same_page_ground_truth(prediction, ground_truth_boxes):
        if gt_box["_gt_id"] in matched_gt_ids:
            continue

        iou = calculate_iou(prediction, gt_box)

        if iou > best_iou:
            best_gt = gt_box
            best_iou = iou

    return best_gt, best_iou


def _same_page_ground_truth(prediction: dict, ground_truth_boxes: list[dict]) -> list[dict]:
    return [
        gt_box
        for gt_box in ground_truth_boxes
        if gt_box["file_name"] == prediction["file_name"]
        and int(gt_box["page_number"]) == int(prediction["page_number"])
    ]


def _mean_best_iou_per_gt(predictions: list[dict], ground_truth_boxes: list[dict]) -> float:
    best_ious = []

    for gt_box in ground_truth_boxes:
        page_predictions = [
            prediction
            for prediction in predictions
            if prediction["file_name"] == gt_box["file_name"]
            and int(prediction["page_number"]) == int(gt_box["page_number"])
        ]
        best_iou = 0.0

        for prediction in page_predictions:
            best_iou = max(best_iou, calculate_iou(prediction, gt_box))

        best_ious.append(best_iou)

    return _mean(best_ious)


def _average_precision(precision_points: list[float], recall_points: list[float]) -> float:
    if not precision_points or not recall_points:
        return 0.0

    ap = 0.0
    previous_recall = 0.0

    for precision, recall in zip(precision_points, recall_points):
        recall_delta = max(0.0, recall - previous_recall)
        ap += precision * recall_delta
        previous_recall = max(previous_recall, recall)

    return ap


def _box_area(box: dict) -> float:
    width = max(0.0, float(box["right"]) - float(box["left"]))
    height = max(0.0, float(box["bottom"]) - float(box["top"]))
    return width * height


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)
