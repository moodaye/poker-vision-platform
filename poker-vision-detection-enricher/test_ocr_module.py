"""
Integration tests for ocr_module.run_ocr.

These tests load the real EasyOCR model and are slow on first run (model
download + initialisation). Run explicitly with:

    uv run pytest poker-vision-detection-enricher/test_ocr_module.py -v -m integration

Excluded from the default test run via the 'integration' marker.
"""

import pytest
from ocr_module import run_ocr
from PIL import Image, ImageDraw


def _make_text_image(text: str, size: tuple[int, int] = (120, 40)) -> Image.Image:
    """Render white text on a black background — simulates a poker UI chip/pot label."""
    img = Image.new("RGB", size, color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Use default PIL bitmap font — no system font required
    draw.text((4, 8), text, fill=(255, 255, 255))
    return img


@pytest.mark.integration
def test_run_ocr_returns_string() -> None:
    """run_ocr always returns a str and never raises."""
    img = _make_text_image("450")
    result = run_ocr(img)
    assert isinstance(result, str)


@pytest.mark.integration
def test_run_ocr_reads_simple_number() -> None:
    """run_ocr extracts a clearly rendered number from a synthetic crop."""
    img = _make_text_image("1200")
    result = run_ocr(img)
    # Exact match isn't guaranteed with a tiny bitmap font, but digits should appear
    assert any(ch.isdigit() for ch in result), (
        f"Expected digits in OCR output, got: {result!r}"
    )


@pytest.mark.integration
def test_run_ocr_empty_image_returns_string() -> None:
    """run_ocr on a blank image returns an empty string without raising."""
    img = Image.new("RGB", (80, 30), color=(0, 0, 0))
    result = run_ocr(img)
    assert isinstance(result, str)
    assert result == ""


@pytest.mark.integration
def test_run_ocr_tiny_crop_upscales_without_error() -> None:
    """run_ocr handles very small crops (< 32px) without raising."""
    img = _make_text_image("99", size=(20, 12))
    result = run_ocr(img)
    assert isinstance(result, str)
