from __future__ import annotations

import pandas as pd
import pytesseract
from PIL import Image


def extract_text(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)


def extract_words(image: Image.Image) -> list[dict]:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
    data = data.dropna(subset=["text"])
    data = data[data["text"].astype(str).str.strip() != ""]

    columns = ["text", "conf", "left", "top", "width", "height", "block_num", "par_num", "line_num", "word_num"]
    words = data[columns].copy()
    words["conf"] = pd.to_numeric(words["conf"], errors="coerce").fillna(-1)

    return words.to_dict(orient="records")
