from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image


@lru_cache(maxsize=1)
def _get_ocr():
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def extract_text(image: Image.Image) -> str:
    words = extract_words(image)
    return "\n".join(word["text"] for word in words)


def extract_words(image: Image.Image) -> list[dict]:
    ocr = _get_ocr()
    image_array = np.array(image.convert("RGB"))
    results = ocr.predict(image_array)

    if not results:
        return []

    result = results[0]
    texts = result["rec_texts"]
    scores = result["rec_scores"]
    boxes = result["rec_boxes"]

    words: list[dict] = []

    for index, (text, score, box) in enumerate(zip(texts, scores, boxes), start=1):
        cleaned_text = str(text).strip()

        if not cleaned_text:
            continue

        left, top, right, bottom = [int(value) for value in box.tolist()]

        words.append(
            {
                "text": cleaned_text,
                "conf": float(score) * 100,
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top,
                "block_num": 1,
                "par_num": 1,
                "line_num": index,
                "word_num": 1,
            }
        )

    return words
