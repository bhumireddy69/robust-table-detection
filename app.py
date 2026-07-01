from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.document_loader import SUPPORTED_TYPES, load_document_pages
from src.image_processing import (
    draw_region_box,
    draw_region_boxes,
    draw_word_boxes, 
    preprocess_for_ocr, 
    extract_table_lines, 
    draw_line_overlay
)
from src import ocr as tesseract_ocr
from src.table_engines import paddle_ocr_engine
from src.table_engines import paddle_table_engine
from src.table_engines import table_transformer_engine
from src.table_structure import (
    build_bounding_box_from_x_band,
    build_table_rows,
    filter_words_by_region,
    find_repeated_x_positions,
    find_strongest_x_band,
    find_table_like_rows,
    get_rows_by_indexes,
    group_words_into_rows,
    infer_column_anchors,
    merge_column_anchors,
    summarize_row,
)


OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


st.set_page_config(
    page_title="Table Structure PoC",
    page_icon="TS",
    layout="wide",
)


st.title("Table Structure PoC")

uploaded_file = st.file_uploader(
    "Upload a scanned PDF or image",
    type=SUPPORTED_TYPES,
)

dpi = st.sidebar.slider("PDF render DPI", min_value=100, max_value=400, value=200, step=50)
ocr_engine = st.sidebar.selectbox(
    "OCR Engine",
    options=["Tesseract", "PaddleOCR"],
)
table_engine = st.sidebar.selectbox(
    "Table Engine",
    options=[
        "None",
        "PaddleOCR Table Recognition",
        "Table Transformer Detection",
        "Table Transformer Structure Recognition",
        "All Table Engines",
    ],
)
run_ocr = st.sidebar.button("Run OCR", type="primary", disabled=uploaded_file is None)

if uploaded_file is None:
    st.info("Upload a PDF or image to start.")
    st.stop()

file_bytes = uploaded_file.getvalue()

try:
    pages = load_document_pages(file_bytes, uploaded_file.name, dpi=dpi)
except Exception as exc:
    st.error(f"Could not load document: {exc}")
    st.stop()

page_number = st.sidebar.number_input(
    "Page",
    min_value=1,
    max_value=len(pages),
    value=1,
    step=1,
)

original_image = pages[page_number - 1]

st.sidebar.subheader("Table Region")
region_left = st.sidebar.number_input("Left", min_value=0, max_value=original_image.width, value=0, step=10)
region_top = st.sidebar.number_input("Top", min_value=0, max_value=original_image.height, value=0, step=10)
region_right = st.sidebar.number_input(
    "Right",
    min_value=0,
    max_value=original_image.width,
    value=original_image.width,
    step=10,
)
region_bottom = st.sidebar.number_input(
    "Bottom",
    min_value=0,
    max_value=original_image.height,
    value=original_image.height,
    step=10,
)

preprocessed_image = preprocess_for_ocr(original_image)
horizontal_lines, vertical_lines, combined_lines = extract_table_lines(preprocessed_image)
line_overlay = draw_line_overlay(original_image, combined_lines)

original_path = OUTPUT_DIR / f"page_{page_number}_original.png"
preprocessed_path = OUTPUT_DIR / f"page_{page_number}_preprocessed.png"
horizontal_lines_path = OUTPUT_DIR / f"page_{page_number}_horizontal_lines.png"
vertical_lines_path = OUTPUT_DIR / f"page_{page_number}_vertical_lines.png"
combined_lines_path = OUTPUT_DIR / f"page_{page_number}_combined_lines.png"
line_overlay_path = OUTPUT_DIR / f"page_{page_number}_line_overlay.png"

original_image.save(original_path)
preprocessed_image.save(preprocessed_path)
horizontal_lines.save(horizontal_lines_path)
vertical_lines.save(vertical_lines_path)
combined_lines.save(combined_lines_path)
line_overlay.save(line_overlay_path)

left, right = st.columns(2)

with left:
    st.subheader("Original")
    st.image(original_image, use_container_width=True)

with right:
    st.subheader("Preprocessed for OCR")
    st.image(preprocessed_image, use_container_width=True)

if not run_ocr:
    st.stop()

with st.spinner(f"Running {ocr_engine} OCR..."):
    if ocr_engine == "PaddleOCR":
        text = paddle_ocr_engine.extract_text(original_image)
        words = paddle_ocr_engine.extract_words(original_image)
    else:
        text = tesseract_ocr.extract_text(preprocessed_image)
        words = tesseract_ocr.extract_words(preprocessed_image)

    boxed_image = draw_word_boxes(original_image, words)

if ocr_engine == "Tesseract":
    heuristic_words = words
else:
    with st.spinner("Running Tesseract OCR for custom heuristic baseline..."):
        heuristic_words = tesseract_ocr.extract_words(preprocessed_image)

all_grouped_rows = group_words_into_rows(heuristic_words)
table_like_rows = find_table_like_rows(all_grouped_rows)
table_like_row_indexes = [row["row_index"] for row in table_like_rows]
focused_rows = get_rows_by_indexes(all_grouped_rows, table_like_row_indexes)
x_clusters = find_repeated_x_positions(focused_rows, x_tolerance=14, min_occurrences=2)
x_band = find_strongest_x_band(x_clusters, max_gap=90, min_clusters=3)
auto_table_box = build_bounding_box_from_x_band(table_like_rows, x_band, padding=20)
auto_table_box_image = draw_region_box(original_image, auto_table_box)
custom_table_rows = []
custom_auto_grouped_rows = []
custom_auto_table_words = []
custom_column_anchors = []
custom_column_groups = []

if auto_table_box is not None:
    custom_auto_table_words = filter_words_by_region(
        heuristic_words,
        left=auto_table_box["left"],
        top=auto_table_box["top"],
        right=auto_table_box["right"],
        bottom=auto_table_box["bottom"],
    )
    custom_auto_grouped_rows = group_words_into_rows(custom_auto_table_words)
    custom_column_anchors = infer_column_anchors(custom_auto_grouped_rows, x_tolerance=20, min_occurrences=2)
    custom_column_groups = merge_column_anchors(custom_column_anchors, max_gap=55)
    custom_table_rows = build_table_rows(custom_auto_grouped_rows, custom_column_groups)

paddle_table_result = None
table_transformer_boxes = []
table_transformer_structure = None
table_transformer_grid = []

run_paddle_table = table_engine in ["PaddleOCR Table Recognition", "All Table Engines"]
run_table_transformer_detection = table_engine in [
    "Table Transformer Detection",
    "Table Transformer Structure Recognition",
    "All Table Engines",
]
run_table_transformer_structure = table_engine in [
    "Table Transformer Structure Recognition",
    "All Table Engines",
]

if run_paddle_table:
    with st.spinner("Running PaddleOCR table recognition..."):
        paddle_table_result = paddle_table_engine.recognize_tables(str(original_path))

if run_table_transformer_detection:
    with st.spinner("Running Table Transformer detection..."):
        table_transformer_boxes = table_transformer_engine.detect_tables(original_image)

    if run_table_transformer_structure and table_transformer_boxes:
        with st.spinner("Running Table Transformer structure recognition..."):
            table_transformer_structure = table_transformer_engine.recognize_structure(
                original_image,
                table_transformer_boxes[0],
            )
            table_transformer_grid = table_transformer_engine.build_grid_from_words(
                table_transformer_structure,
                heuristic_words,
            )

st.subheader("Table Line Detection")

line_col1, line_col2 = st.columns(2)

with line_col1:
    st.caption("Horizontal lines")
    st.image(horizontal_lines, use_container_width=True)

with line_col2:
    st.caption("Vertical lines")
    st.image(vertical_lines, use_container_width=True)

st.caption("Detected lines overlay")
st.image(line_overlay, use_container_width=True)  

st.subheader("Automatic Candidate Table Region")
st.caption("Custom heuristic baseline uses Tesseract word-level OCR boxes.")

if auto_table_box is None:
    st.warning("No automatic table candidate was detected.")
else:
    st.json(auto_table_box)
    st.image(auto_table_box_image, use_container_width=True)

st.subheader("PaddleOCR Table Recognition")

if not run_paddle_table:
    st.info("Select PaddleOCR Table Recognition in the sidebar to run the table-aware model.")
elif paddle_table_result is None:
    st.warning("PaddleOCR table recognition did not run.")
else:
    st.write("Detected table boxes:")
    st.json(paddle_table_result["table_boxes"])

    if paddle_table_result["table_boxes"]:
        first_table_box_image = draw_region_box(original_image, paddle_table_result["table_boxes"][0])
        st.image(first_table_box_image, use_container_width=True)

    if not paddle_table_result["tables"]:
        st.warning("No table structures were returned.")
    else:
        for table_index, table in enumerate(paddle_table_result["tables"], start=1):
            st.write(f"Table {table_index}")
            st.caption(f"Detected cells: {table['cell_count']}")

            if table["dataframes"]:
                st.dataframe(table["dataframes"][0], use_container_width=True)
            else:
                st.warning("Could not parse PaddleOCR table HTML into a dataframe.")

            with st.expander("PaddleOCR table HTML"):
                st.code(table["html"], language="html")

            with st.expander("PaddleOCR table OCR texts"):
                st.write(table["ocr_texts"])

st.subheader("Table Transformer Detection")

if not run_table_transformer_detection:
    st.info("Select a Table Transformer option in the sidebar to run Microsoft's table models.")
else:
    if not table_transformer_boxes:
        st.warning("No Table Transformer table boxes were detected.")
    else:
        st.write("Detected table boxes:")
        st.json(table_transformer_boxes)
        st.image(draw_region_box(original_image, table_transformer_boxes[0]), use_container_width=True)

        if table_transformer_structure is not None:
            st.write("Structure box counts:")
            st.json(
                {
                    "rows": len(table_transformer_structure["rows"]),
                    "columns": len(table_transformer_structure["columns"]),
                    "column_headers": len(table_transformer_structure["column_headers"]),
                    "spanning_cells": len(table_transformer_structure["spanning_cells"]),
                    "projected_row_headers": len(table_transformer_structure["projected_row_headers"]),
                }
            )

            structure_view_left, structure_view_right = st.columns(2)

            with structure_view_left:
                st.caption("Detected rows")
                st.image(
                    draw_region_boxes(original_image, table_transformer_structure["rows"], color=(255, 0, 0)),
                    use_container_width=True,
                )

            with structure_view_right:
                st.caption("Detected columns")
                st.image(
                    draw_region_boxes(original_image, table_transformer_structure["columns"], color=(0, 255, 0)),
                    use_container_width=True,
                )

            st.write("Table Transformer mapped table preview:")
            st.dataframe(pd.DataFrame(table_transformer_grid), use_container_width=True)

            with st.expander("Table Transformer structure boxes"):
                structure_boxes_for_display = {
                    key: table_transformer_structure[key]
                    for key in [
                        "crop_box",
                        "rows",
                        "columns",
                        "column_headers",
                        "spanning_cells",
                        "projected_row_headers",
                    ]
                }
                st.json(structure_boxes_for_display)

st.subheader("Baseline Comparison")

if table_engine == "None":
    st.info("Enable a table engine to compare table-region detection.")
else:
    comparison_columns = st.columns(3)

    with comparison_columns[0]:
        st.caption("Custom heuristic candidate box (Tesseract word boxes)")
        if auto_table_box is None:
            st.warning("No custom heuristic candidate box was detected.")
        else:
            st.json(auto_table_box)
            st.image(auto_table_box_image, use_container_width=True)

    with comparison_columns[1]:
        st.caption("PaddleOCR table-recognition box")
        if paddle_table_result is None:
            st.info("Not selected for this run.")
        else:
            paddle_box = paddle_table_result["table_boxes"][0] if paddle_table_result["table_boxes"] else None
            st.json(paddle_box)
            st.image(draw_region_box(original_image, paddle_box), use_container_width=True)

    with comparison_columns[2]:
        st.caption("Table Transformer detection box")
        transformer_box = table_transformer_boxes[0] if table_transformer_boxes else None
        if transformer_box is None:
            st.info("Not selected for this run.")
        else:
            st.json(transformer_box)
            st.image(draw_region_box(original_image, transformer_box), use_container_width=True)

st.subheader("Structure Comparison")

if paddle_table_result is None and table_transformer_structure is None:
    st.info("Enable PaddleOCR Table Recognition or Table Transformer Structure Recognition to compare table structure outputs.")
else:
    structure_columns = st.columns(3)

    with structure_columns[0]:
        st.caption("Custom heuristic mapped table")
        custom_df = pd.DataFrame(custom_table_rows)
        st.write(f"Rows: {custom_df.shape[0]}, Columns: {custom_df.shape[1]}")

        if custom_df.empty:
            st.warning("No custom mapped table was produced.")
        else:
            st.dataframe(custom_df, use_container_width=True)

    with structure_columns[1]:
        st.caption("PaddleOCR parsed table")
        paddle_df = None

        if (
            paddle_table_result is not None
            and paddle_table_result["tables"]
            and paddle_table_result["tables"][0]["dataframes"]
        ):
            paddle_df = paddle_table_result["tables"][0]["dataframes"][0]

        if paddle_df is None:
            st.info("Not selected for this run.")
        else:
            st.write(f"Rows: {paddle_df.shape[0]}, Columns: {paddle_df.shape[1]}")
            st.dataframe(paddle_df, use_container_width=True)

    with structure_columns[2]:
        st.caption("Table Transformer mapped table")
        transformer_df = pd.DataFrame(table_transformer_grid)

        if transformer_df.empty:
            st.info("Not selected for this run.")
        else:
            st.write(f"Rows: {transformer_df.shape[0]}, Columns: {transformer_df.shape[1]}")
            st.dataframe(transformer_df, use_container_width=True)

boxed_path = OUTPUT_DIR / f"page_{page_number}_word_boxes.png"
text_path = OUTPUT_DIR / f"page_{page_number}_ocr.txt"
csv_path = OUTPUT_DIR / f"page_{page_number}_words.csv"
auto_table_box_path = OUTPUT_DIR / f"page_{page_number}_auto_table_box.png"
ocr_engine_path = OUTPUT_DIR / f"page_{page_number}_ocr_engine.txt"

boxed_image.save(boxed_path)
auto_table_box_image.save(auto_table_box_path)
text_path.write_text(text, encoding="utf-8")
ocr_engine_path.write_text(ocr_engine, encoding="utf-8")
pd.DataFrame(words).to_csv(csv_path, index=False)

st.subheader("OCR Word Boxes")
st.image(boxed_image, use_container_width=True)

st.subheader("OCR Row Groups")

table_words = filter_words_by_region(
    words,
    left=region_left,
    top=region_top,
    right=region_right,
    bottom=region_bottom,
)
grouped_rows = group_words_into_rows(table_words)

st.caption(f"Manual region filtered words: {len(table_words)} of {len(words)}")

for index, row in enumerate(grouped_rows, start=1):
    row_summary = summarize_row(row)
    st.write(
        f"Row {index}: "
        f"words={row_summary['word_count']} "
        f"numeric_ratio={row_summary['numeric_ratio']:.2f} "
        f"| {row_summary['text']}"
    )

st.subheader("Automatic Table Row Groups")

if auto_table_box is None:
    st.warning("No automatic table candidate was detected.")
else:
    st.caption(f"Automatic region filtered Tesseract words: {len(custom_auto_table_words)} of {len(heuristic_words)}")
    st.write(f"Column anchors: `{custom_column_anchors}`")
    st.write("Column groups:")
    st.json(custom_column_groups)

    st.write("Mapped table preview:")
    st.dataframe(pd.DataFrame(custom_table_rows), use_container_width=True)

    for index, row in enumerate(custom_auto_grouped_rows, start=1):
        row_summary = summarize_row(row)
        st.write(
            f"Row {index}: "
            f"words={row_summary['word_count']} "
            f"numeric_ratio={row_summary['numeric_ratio']:.2f} "
            f"| {row_summary['text']}"
        )

text_tab, words_tab, files_tab = st.tabs(["Extracted Text", "Word Data", "Generated Files"])

with text_tab:
    st.text_area("Text", value=text, height=300)

with words_tab:
    st.dataframe(pd.DataFrame(words), use_container_width=True)

with files_tab:
    st.write(f"Original image: `{original_path}`")
    st.write(f"Preprocessed image: `{preprocessed_path}`")
    st.write(f"Automatic table box image: `{auto_table_box_path}`")
    st.write(f"Word boxes image: `{boxed_path}`")
    st.write(f"OCR text: `{text_path}`")
    st.write(f"OCR word CSV: `{csv_path}`")
    st.write(f"OCR engine: `{ocr_engine}`")
