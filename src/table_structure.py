from __future__ import annotations


def group_words_into_rows(words: list[dict], y_tolerance: int = 10) -> list[list[dict]]:
    rows: list[list[dict]] = []

    sorted_words = sorted(words, key=lambda word: word["top"])

    for word in sorted_words:
        word_center_y = word["top"] + word["height"] / 2

        matching_row = None

        for row in rows:
            row_center_y = _row_center_y(row)

            if abs(word_center_y - row_center_y) <= y_tolerance:
                matching_row = row
                break

        if matching_row is None:
            rows.append([word])
        else:
            matching_row.append(word)

    for row in rows:
        row.sort(key=lambda word: word["left"])

    rows.sort(key=lambda row: _row_center_y(row))

    return rows


def filter_words_by_region(
    words: list[dict],
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> list[dict]:
    filtered_words: list[dict] = []

    for word in words:
        center_x = word["left"] + word["width"] / 2
        center_y = word["top"] + word["height"] / 2

        if left <= center_x <= right and top <= center_y <= bottom:
            filtered_words.append(word)

    return filtered_words


def summarize_row(row: list[dict]) -> dict:
    text_tokens = [str(word["text"]) for word in row]
    numeric_count = sum(1 for token in text_tokens if _is_numeric_like(token))
    word_count = len(row)

    left = min(word["left"] for word in row)
    top = min(word["top"] for word in row)
    right = max(word["left"] + word["width"] for word in row)
    bottom = max(word["top"] + word["height"] for word in row)

    return {
        "text": " ".join(text_tokens),
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "word_count": word_count,
        "numeric_count": numeric_count,
        "numeric_ratio": numeric_count / word_count if word_count else 0,
    }


def find_table_like_rows(
    rows: list[list[dict]],
    min_words: int = 4,
    min_numeric_ratio: float = 0.3,
) -> list[dict]:
    table_like_rows: list[dict] = []

    for index, row in enumerate(rows):
        row_summary = summarize_row(row)
        row_summary["row_index"] = index

        if (
            row_summary["word_count"] >= min_words
            and row_summary["numeric_ratio"] >= min_numeric_ratio
        ):
            table_like_rows.append(row_summary)

    return table_like_rows


def build_bounding_box_from_rows(row_summaries: list[dict]) -> dict | None:
    if not row_summaries:
        return None

    return {
        "left": min(row["left"] for row in row_summaries),
        "top": min(row["top"] for row in row_summaries),
        "right": max(row["right"] for row in row_summaries),
        "bottom": max(row["bottom"] for row in row_summaries),
        "row_count": len(row_summaries),
    }


def find_repeated_x_positions(
    rows: list[list[dict]],
    x_tolerance: int = 12,
    min_occurrences: int = 2,
) -> list[dict]:
    x_positions = []

    for row_index, row in enumerate(rows):
        for word in row:
            x_positions.append(
                {
                    "x": word["left"],
                    "row_index": row_index,
                    "text": str(word["text"]),
                }
            )

    x_positions.sort(key=lambda item: item["x"])

    clusters: list[list[dict]] = []

    for item in x_positions:
        matching_cluster = None

        for cluster in clusters:
            cluster_center = _cluster_center_x(cluster)

            if abs(item["x"] - cluster_center) <= x_tolerance:
                matching_cluster = cluster
                break

        if matching_cluster is None:
            clusters.append([item])
        else:
            matching_cluster.append(item)

    repeated_clusters = []

    for cluster in clusters:
        row_indexes = {item["row_index"] for item in cluster}

        if len(row_indexes) >= min_occurrences:
            repeated_clusters.append(
                {
                    "x": round(_cluster_center_x(cluster)),
                    "occurrences": len(cluster),
                    "row_count": len(row_indexes),
                    "examples": [item["text"] for item in cluster[:5]],
                }
            )

    repeated_clusters.sort(key=lambda cluster: cluster["x"])

    return repeated_clusters


def get_rows_by_indexes(rows: list[list[dict]], row_indexes: list[int]) -> list[list[dict]]:
    selected_rows: list[list[dict]] = []

    for row_index in row_indexes:
        if 0 <= row_index < len(rows):
            selected_rows.append(rows[row_index])

    return selected_rows


def find_strongest_x_band(
    x_clusters: list[dict],
    max_gap: int = 90,
    min_clusters: int = 3,
) -> dict | None:
    if not x_clusters:
        return None

    sorted_clusters = sorted(x_clusters, key=lambda cluster: cluster["x"])
    bands: list[list[dict]] = [[sorted_clusters[0]]]

    for cluster in sorted_clusters[1:]:
        current_band = bands[-1]
        previous_x = current_band[-1]["x"]

        if cluster["x"] - previous_x <= max_gap:
            current_band.append(cluster)
        else:
            bands.append([cluster])

    candidate_bands = [
        band
        for band in bands
        if len(band) >= min_clusters
    ]

    if not candidate_bands:
        return None

    strongest_band = max(
        candidate_bands,
        key=lambda band: (
            sum(cluster["row_count"] for cluster in band),
            len(band),
        ),
    )

    return {
        "left": min(cluster["x"] for cluster in strongest_band),
        "right": max(cluster["x"] for cluster in strongest_band),
        "cluster_count": len(strongest_band),
        "score": sum(cluster["row_count"] for cluster in strongest_band),
        "clusters": strongest_band,
    }


def infer_column_anchors(
    rows: list[list[dict]],
    x_tolerance: int = 20,
    min_occurrences: int = 2,
) -> list[int]:
    x_clusters = find_repeated_x_positions(
        rows,
        x_tolerance=x_tolerance,
        min_occurrences=min_occurrences,
    )

    return [cluster["x"] for cluster in x_clusters]


def merge_column_anchors(anchors: list[int], max_gap: int = 70) -> list[dict]:
    if not anchors:
        return []

    sorted_anchors = sorted(anchors)
    groups: list[list[int]] = [[sorted_anchors[0]]]

    for anchor in sorted_anchors[1:]:
        current_group = groups[-1]

        if anchor - current_group[-1] <= max_gap:
            current_group.append(anchor)
        else:
            groups.append([anchor])

    column_groups = []

    for group in groups:
        column_groups.append(
            {
                "left": min(group),
                "right": max(group),
                "center": round(sum(group) / len(group)),
                "anchor_count": len(group),
                "anchors": group,
            }
        )

    return column_groups


def build_column_boundaries(column_groups: list[dict]) -> list[dict]:
    if not column_groups:
        return []

    sorted_columns = sorted(column_groups, key=lambda column: column["center"])
    boundaries = []

    for index, column in enumerate(sorted_columns):
        if index == 0:
            left_boundary = float("-inf")
        else:
            previous_center = sorted_columns[index - 1]["center"]
            left_boundary = (previous_center + column["center"]) / 2

        if index == len(sorted_columns) - 1:
            right_boundary = float("inf")
        else:
            next_center = sorted_columns[index + 1]["center"]
            right_boundary = (column["center"] + next_center) / 2

        boundaries.append(
            {
                "left": left_boundary,
                "right": right_boundary,
                "center": column["center"],
                "column": column,
            }
        )

    return boundaries


def assign_words_to_columns(row: list[dict], column_groups: list[dict]) -> list[str]:
    columns: list[list[dict]] = [
        []
        for _ in column_groups
    ]

    if not column_groups:
        return []

    column_boundaries = build_column_boundaries(column_groups)

    for word in row:
        column_index = _column_index_for_x(word["left"], column_boundaries)
        columns[column_index].append(word)

    row_values = []

    for column_words in columns:
        column_words.sort(key=lambda word: word["left"])
        row_values.append(" ".join(str(word["text"]) for word in column_words))

    return row_values


def build_table_rows(rows: list[list[dict]], column_groups: list[dict]) -> list[list[str]]:
    return [
        assign_words_to_columns(row, column_groups)
        for row in rows
    ]


def build_bounding_box_from_x_band(
    row_summaries: list[dict],
    x_band: dict | None,
    padding: int = 20,
) -> dict | None:
    if not row_summaries or x_band is None:
        return None

    return {
        "left": max(0, x_band["left"] - padding),
        "top": max(0, min(row["top"] for row in row_summaries) - padding),
        "right": x_band["right"] + padding,
        "bottom": max(row["bottom"] for row in row_summaries) + padding,
        "row_count": len(row_summaries),
        "x_cluster_count": x_band["cluster_count"],
        "score": x_band["score"],
    }


def _row_center_y(row: list[dict]) -> float:
    centers = [
        word["top"] + word["height"] / 2
        for word in row
    ]

    return sum(centers) / len(centers)


def _cluster_center_x(cluster: list[dict]) -> float:
    return sum(item["x"] for item in cluster) / len(cluster)


def _nearest_column_index(center_x: float, column_groups: list[dict]) -> int:
    distances = [
        abs(center_x - column["center"])
        for column in column_groups
    ]

    return distances.index(min(distances))


def _column_index_for_x(x: float, column_boundaries: list[dict]) -> int:
    for index, boundary in enumerate(column_boundaries):
        if boundary["left"] <= x < boundary["right"]:
            return index

    return len(column_boundaries) - 1


def _is_numeric_like(text: str) -> bool:
    cleaned = (
        text.strip()
        .replace(",", "")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
    )

    return cleaned.isdigit()
