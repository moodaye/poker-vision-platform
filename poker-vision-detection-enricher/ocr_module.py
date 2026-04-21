"""
OCR module — uses EasyOCR to extract text from a pre-cropped image region.

The EasyOCR reader is initialised lazily on first call so that importing this
module does not trigger a heavyweight model load.
"""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)

_reader = None


def _get_reader() -> Any:
    global _reader
    if _reader is None:
        import easyocr  # deferred import — avoids slow load at module import time

        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def _preprocess(image_crop: Image.Image) -> Image.Image:
    img = image_crop.convert("L")  # greyscale

    # Upscale if the shorter dimension is below 32px for better OCR accuracy
    w, h = img.size
    short = min(w, h)
    if short < 32:
        scale = 32 / short
        img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)

    # Mild contrast boost
    img = ImageEnhance.Contrast(img).enhance(1.5)

    return img


def run_ocr(image_crop: Image.Image) -> str:
    import numpy as np  # deferred — numpy is an easyocr transitive dep

    try:
        img = _preprocess(image_crop)
        arr = np.array(img)  # EasyOCR requires a numpy array, not a PIL Image
        reader = _get_reader()
        results = reader.readtext(arr, detail=0, paragraph=True)
        return " ".join(str(r) for r in results).strip()
    except Exception:
        logger.exception("OCR failed")
        return ""
