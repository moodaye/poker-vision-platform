"""
OCR module — uses pytesseract to extract text from a pre-cropped image region.

Requires the Tesseract binary to be installed on the host machine.
Windows default path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Iterable

import pytesseract
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)

# Allow overriding the Tesseract binary path via environment variable.
# On Windows the installer does not add Tesseract to PATH by default.
_TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD

# OCR profiles by object class/type. Numeric regions prioritize digits and
# separators; player names allow alphanumeric text.
_OCR_PROFILES: dict[str, list[str]] = {
    "numeric": [
        "--psm 7 -c tessedit_char_whitelist=0123456789,./",
        "--psm 6 -c tessedit_char_whitelist=0123456789,./",
    ],
    "blinds": [
        "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/:- ",
        "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/:- ",
    ],
    "total_pot": [
        "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,.:$ ",
        "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,.:$ ",
    ],
    "player_name": [
        "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.",
        "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.",
    ],
}

# Minimum Tesseract word-level confidence (0–100) to include a word in the result.
# Words below this are treated as noise.
_MIN_WORD_CONF = 0


def _preprocess(image_crop: Image.Image, *, strong: bool = False) -> Image.Image:
    img = image_crop.convert("L")  # greyscale

    # Upscale if the shorter dimension is below 32px for better OCR accuracy
    w, h = img.size
    short = min(w, h)
    if short < 32:
        scale = 32 / short
        img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)

    # Mild contrast boost, with an optional stronger pass for difficult crops.
    contrast = 2.0 if strong else 1.5
    img = ImageEnhance.Contrast(img).enhance(contrast)

    if strong:
        # Binary thresholding reduces anti-aliased HUD text noise.
        img = img.point(lambda p: 255 if p >= 140 else 0)

    return img


def _iter_ocr_passes(
    profile: str,
    max_passes: int = 0,
) -> Iterable[tuple[bool, str]]:
    configs = _OCR_PROFILES.get(profile) or _OCR_PROFILES["numeric"]
    count = 0
    for strong in (False, True):
        for config in configs:
            if max_passes > 0 and count >= max_passes:
                return
            yield strong, config
            count += 1


def _clean_text_for_profile(text: str, profile: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    if profile == "numeric":
        return re.sub(r"[^0-9,./]", "", cleaned)

    if profile == "blinds":
        # Preserve labels and separators; parser extracts numeric values later.
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    if profile == "total_pot":
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    return cleaned


# Stop trying further passes once we have at least this confidence.
_EARLY_EXIT_CONF = 0.70


def run_ocr(
    image_crop: Image.Image,
    profile: str = "numeric",
    max_passes: int = 1,
) -> tuple[str, float]:
    """Return ``(text, confidence)`` where *confidence* is in ``[0.0, 1.0]``.

    Confidence is the mean of Tesseract's per-word confidence scores,
    normalised from the 0–100 integer range Tesseract uses.  Returns
    ``("", 0.0)`` when no text is recognised or OCR fails.

    By default this does a single pass using the profile's first PSM config.
    Set ``max_passes`` to a larger value (or 0 for unlimited) to enable the
    legacy multi-pass fallback behaviour.

    Performance: exits early when confidence >= _EARLY_EXIT_CONF to avoid
    spawning redundant Tesseract subprocesses (each call launches tesseract.exe).
    """
    t_ocr_start = time.perf_counter()
    pass_count = 0
    try:
        best_text = ""
        best_conf = 0.0

        for strong, config in _iter_ocr_passes(profile, max_passes=max_passes):
            t_pass = time.perf_counter()
            pass_count += 1
            img = _preprocess(image_crop, strong=strong)

            # Primary attempt: word-level confidences via image_to_data.
            data = pytesseract.image_to_data(
                img,
                config=config,
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
                text = ""
                conf = 0.0
            else:
                text = "".join(text_parts)
                text = _clean_text_for_profile(text, profile)
                conf = round(sum(word_confs) / len(word_confs) / 100.0, 4)
                if text and (
                    conf > best_conf
                    or (conf == best_conf and len(text) > len(best_text))
                ):
                    best_text = text
                    best_conf = conf

            # Fallback: image_to_string recovers mixed label+numeric fields
            # where image_to_data emits sparse/low-confidence words.
            # Only run if primary pass produced nothing useful.
            if not text:
                raw_text = pytesseract.image_to_string(img, config=config)
                raw_text = _clean_text_for_profile(raw_text, profile)
                if raw_text:
                    digit_count = sum(1 for ch in raw_text if ch.isdigit())
                    heuristic_conf = round(0.50 + min(0.35, digit_count * 0.03), 4)
                    if heuristic_conf > best_conf or (
                        heuristic_conf == best_conf and len(raw_text) > len(best_text)
                    ):
                        best_text = raw_text
                        best_conf = heuristic_conf

            logger.debug(
                "  ocr pass %d (strong=%s) %.3fs  → %r conf=%.2f",
                pass_count,
                strong,
                time.perf_counter() - t_pass,
                text,
                conf,
            )

            # Early exit: no point running more passes if we already have a
            # confident result — avoids spawning extra tesseract.exe processes.
            if best_conf >= _EARLY_EXIT_CONF:
                break

        logger.info(
            "[timing] ocr profile=%-12s passes=%d total=%.3fs  result=%r conf=%.2f",
            profile,
            pass_count,
            time.perf_counter() - t_ocr_start,
            best_text,
            best_conf,
        )

        if not best_text:
            return "", 0.0
        return best_text, best_conf
    except Exception:
        logger.exception("OCR failed")
        return "", 0.0
