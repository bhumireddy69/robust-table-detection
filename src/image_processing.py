from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def pil_to_cv(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv_to_pil(image: np.ndarray) -> Image.Image:
    if len(image.shape) == 2:
        return Image.fromarray(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    cv_image = pil_to_cv(image)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=18)
    thresholded = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )
    return cv_to_pil(thresholded)


def draw_word_boxes(image: Image.Image, words: list[dict]) -> Image.Image:
    boxed = pil_to_cv(image)

    for word in words:
        if word["text"].strip() == "":
            continue

        x, y, width, height = word["left"], word["top"], word["width"], word["height"]
        cv2.rectangle(boxed, (x, y), (x + width, y + height), (0, 120, 255), 2)

    return cv_to_pil(boxed)


def extract_table_lines(preprocessed_image: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
    # Though we are passing preprocessed gray image, we are again doing it by with 'L' for defensive purpose
    gray = np.array(preprocessed_image.convert("L"))
    inverted = cv2.bitwise_not(gray)

    image_width = inverted.shape[1]
    horizontal_size = max(10, image_width // 30)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal_lines = cv2.erode(inverted, horizontal_kernel)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel)

    image_height = inverted.shape[0]
    vertical_size = max(10, image_height // 30)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical_lines = cv2.erode(inverted, vertical_kernel)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel)

    combined_lines = cv2.add(horizontal_lines, vertical_lines)
    
    return cv_to_pil(horizontal_lines), cv_to_pil(vertical_lines), cv_to_pil(combined_lines)

def draw_line_overlay(original_image: Image.Image, combined_lines: Image.Image) ->  Image.Image:
    original_cv = pil_to_cv(original_image)
    lines_gray = np.array(combined_lines.convert("L"))
    overlay = original_cv.copy()
    overlay[lines_gray > 0] = (0, 0, 255)

    return cv_to_pil(overlay)


def draw_region_box(image: Image.Image, box: dict | None, color: tuple[int, int, int] = (0, 255, 0)) -> Image.Image:
    boxed = pil_to_cv(image)

    if box is None:
        return cv_to_pil(boxed)

    left = int(box["left"])
    top = int(box["top"])
    right = int(box["right"])
    bottom = int(box["bottom"])

    cv2.rectangle(boxed, (left, top), (right, bottom), color, 3)

    return cv_to_pil(boxed)


def draw_region_boxes(
    image: Image.Image,
    boxes: list[dict],
    color: tuple[int, int, int] = (0, 255, 0),
) -> Image.Image:
    boxed = pil_to_cv(image)

    for box in boxes:
        left = int(box["left"])
        top = int(box["top"])
        right = int(box["right"])
        bottom = int(box["bottom"])
        cv2.rectangle(boxed, (left, top), (right, bottom), color, 2)

    return cv_to_pil(boxed)
