"""
OCR module — uses pytesseract to extract text from a pre-cropped image region.

Requires the Tesseract binary to be installed on the host machine.
Windows default path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe
"""

from __future__ import annotations

import logging
import os

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

# Minimum Tesseract word-level confidence (0–100) to include a word in the result.
# Words below this are treated as noise.
_MIN_WORD_CONF = 0


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


def run_ocr(image_crop: Image.Image) -> tuple[str, float]:
    """Return ``(text, confidence)`` where *confidence* is in ``[0.0, 1.0]``.

    Confidence is the mean of Tesseract's per-word confidence scores,
    normalised from the 0–100 integer range Tesseract uses.  Returns
    ``("", 0.0)`` when no text is recognised or OCR fails.
    """
    try:
        img = _preprocess(image_crop)
        data = pytesseract.image_to_data(
            img,
            config=_TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT,
        )
        # Tesseract emits conf=-1 for non-word layout rows; skip those and blanks.
        word_confs = [
            int(c)
            for c, t in zip(data["conf"], data["text"], strict=False)
            if int(c) > _MIN_WORD_CONF and str(t).strip()
        ]
        text_parts = [
            str(t).strip()
            for c, t in zip(data["conf"], data["text"], strict=False)
            if int(c) > _MIN_WORD_CONF and str(t).strip()
        ]
        if not word_confs:
            return "", 0.0
        text = "".join(text_parts)
        conf = round(sum(word_confs) / len(word_confs) / 100.0, 4)
        return text, conf
    except Exception:
        logger.exception("OCR failed")
        return "", 0.0
