from __future__ import annotations

from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image


SUPPORTED_TYPES = ["pdf", "png", "jpg", "jpeg", "tif", "tiff"]


def load_document_pages(file_bytes: bytes, file_name: str, dpi: int = 200) -> list[Image.Image]:
    suffix = Path(file_name).suffix.lower().replace(".", "")

    if suffix == "pdf":
        return _render_pdf_pages(file_bytes, dpi=dpi)

    if suffix in {"png", "jpg", "jpeg", "tif", "tiff"}:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        return [image]

    raise ValueError(f"Unsupported file type: {suffix}")


def _render_pdf_pages(file_bytes: bytes, dpi: int) -> list[Image.Image]:
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[Image.Image] = []

    with fitz.open(stream=file_bytes, filetype="pdf") as document:
        for page in document:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            pages.append(image)

    return pages
