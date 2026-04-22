"""
OCR module — uses pytesseract to extract text from a pre-cropped image region.

Requires the Tesseract binary to be installed on the host machine.
Windows default path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe
"""

from __future__ import annotations

import logging
import os
from typing import cast

import pytesseract
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)

# Allow overriding the Tesseract binary path via environment variable.
# On Windows the installer does not add Tesseract to PATH by default.
_TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD

# PSM 7 = single text line; whitelist limits recognition to digits and slash
# (covers chip counts like "470" and blind values like "1/2").
_TESSERACT_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789/"


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
    try:
        img = _preprocess(image_crop)
        return cast(
            str, pytesseract.image_to_string(img, config=_TESSERACT_CONFIG)
        ).strip()
    except Exception:
        logger.exception("OCR failed")
        return ""
