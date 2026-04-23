"""
Tests for ocr_module.run_ocr (pytesseract-backed).

pytesseract calls the Tesseract binary synchronously — no model load, so
these tests run in the default test suite without any special marker.

Requires the Tesseract binary installed on the host:
    Windows: https://github.com/UB-Mannheim/tesseract/wiki

run_ocr returns a (text: str, confidence: float) tuple where confidence
is in [0.0, 1.0] — the mean of Tesseract's per-word confidence scores.
"""

from ocr_module import run_ocr
from PIL import Image, ImageDraw


def _make_text_image(text: str, size: tuple[int, int] = (120, 40)) -> Image.Image:
    """Render white text on a black background — simulates a poker UI chip/pot label."""
    img = Image.new("RGB", size, color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Use default PIL bitmap font — no system font required
    draw.text((4, 8), text, fill=(255, 255, 255))
    return img


def test_run_ocr_returns_tuple() -> None:
    """run_ocr returns (str, float) and never raises."""
    img = _make_text_image("450")
    result = run_ocr(img)
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2
    text, conf = result
    assert isinstance(text, str)
    assert isinstance(conf, float)


def test_run_ocr_confidence_in_range() -> None:
    """Confidence value is always in [0.0, 1.0]."""
    img = _make_text_image("1200")
    _, conf = run_ocr(img)
    assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"


def test_run_ocr_reads_simple_number() -> None:
    """run_ocr extracts a clearly rendered number from a synthetic crop."""
    img = _make_text_image("1200")
    text, _ = run_ocr(img)
    # Exact match isn't guaranteed with a tiny bitmap font, but digits should appear
    assert any(ch.isdigit() for ch in text), (
        f"Expected digits in OCR output, got: {text!r}"
    )


def test_run_ocr_empty_image_returns_empty_string_and_zero_conf() -> None:
    """run_ocr on a blank image returns ('', 0.0) without raising."""
    img = Image.new("RGB", (80, 30), color=(0, 0, 0))
    text, conf = run_ocr(img)
    assert isinstance(text, str)
    assert text == ""
    assert conf == 0.0


def test_run_ocr_tiny_crop_upscales_without_error() -> None:
    """run_ocr handles very small crops (< 32px) without raising."""
    img = _make_text_image("99", size=(20, 12))
    text, conf = run_ocr(img)
    assert isinstance(text, str)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0
