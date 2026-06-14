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


# ---------------------------------------------------------------------------
# Internal helpers — profile routing and text cleaning
# ---------------------------------------------------------------------------

from ocr_module import (  # noqa: E402
    _OCR_PROFILES,
    _clean_text_for_profile,
    _iter_ocr_passes,
    _preprocess,
)


def test_iter_ocr_passes_known_profile_yields_four_passes() -> None:
    """Each known profile has 2 configs; combined with 2 strong values → 4 passes."""
    passes = list(_iter_ocr_passes("numeric"))
    assert len(passes) == 4
    # First two passes: strong=False; last two: strong=True
    strong_values = [p[0] for p in passes]
    assert strong_values == [False, False, True, True]
    # Config strings alternate through the profile's two entries
    configs = [p[1] for p in passes]
    assert configs[0] == _OCR_PROFILES["numeric"][0]
    assert configs[1] == _OCR_PROFILES["numeric"][1]


def test_iter_ocr_passes_unknown_profile_falls_back_to_numeric() -> None:
    """An unrecognised profile name silently falls back to the 'numeric' profile."""
    passes_unknown = list(_iter_ocr_passes("totally_unknown_profile", max_passes=1))
    passes_numeric = list(_iter_ocr_passes("numeric", max_passes=1))
    assert passes_unknown == passes_numeric


def test_iter_ocr_passes_max_passes_limits_iterations() -> None:
    passes = list(_iter_ocr_passes("numeric", max_passes=1))
    assert len(passes) == 1
    assert passes[0] == (False, _OCR_PROFILES["numeric"][0])


def test_iter_ocr_passes_zero_max_passes_allows_unlimited() -> None:
    passes = list(_iter_ocr_passes("numeric", max_passes=0))
    assert len(passes) == 4


def test_iter_ocr_passes_player_name_uses_player_name_configs() -> None:
    passes = list(_iter_ocr_passes("player_name"))
    assert len(passes) == 4
    configs = [p[1] for p in passes]
    assert configs[0] == _OCR_PROFILES["player_name"][0]
    assert configs[1] == _OCR_PROFILES["player_name"][1]


def test_clean_text_for_profile_numeric_keeps_digits_and_separators() -> None:
    # Numeric whitelist is [0-9,./] — commas, dots, and slashes are preserved
    assert _clean_text_for_profile("  1,500  ", "numeric") == "1,500"
    assert _clean_text_for_profile("50/100", "numeric") == "50/100"
    assert _clean_text_for_profile("1.5", "numeric") == "1.5"
    # Non-whitelisted characters are stripped
    assert _clean_text_for_profile("$4.50", "numeric") == "4.50"
    assert _clean_text_for_profile("abc", "numeric") == ""


def test_clean_text_for_profile_blinds_collapses_whitespace() -> None:
    result = _clean_text_for_profile("  50 /  100  ", "blinds")
    assert "  " not in result  # no double spaces
    assert "50" in result
    assert "100" in result


def test_clean_text_for_profile_total_pot_collapses_whitespace() -> None:
    result = _clean_text_for_profile("Total  Pot:  300", "total_pot")
    assert "  " not in result
    assert "300" in result


def test_clean_text_for_profile_empty_string_returns_empty() -> None:
    for profile in ("numeric", "blinds", "total_pot", "player_name"):
        assert _clean_text_for_profile("", profile) == ""
        assert _clean_text_for_profile("   ", profile) == ""


def test_preprocess_upscales_image_below_32px() -> None:
    """Images with a short dimension < 32px should be upscaled."""
    small = Image.new("RGB", (60, 20), color=(128, 128, 128))
    result = _preprocess(small)
    w, h = result.size
    assert min(w, h) >= 32


def test_preprocess_does_not_upscale_image_above_32px() -> None:
    """Images already at or above 32px short dimension should not be upscaled."""
    img = Image.new("RGB", (100, 40), color=(100, 100, 100))
    result = _preprocess(img)
    # Size should not change when already large enough
    assert result.size == (100, 40)


def test_preprocess_returns_greyscale() -> None:
    img = Image.new("RGB", (80, 40), color=(200, 100, 50))
    result = _preprocess(img)
    assert result.mode == "L"


def test_run_ocr_player_name_profile_returns_valid_tuple() -> None:
    """run_ocr with the player_name profile should not crash and return (str, float)."""
    img = _make_text_image("Player1", size=(150, 40))
    text, conf = run_ocr(img, profile="player_name")
    assert isinstance(text, str)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


def test_run_ocr_unknown_profile_does_not_raise() -> None:
    """An unknown profile name falls back silently — run_ocr should not raise."""
    img = _make_text_image("500")
    text, conf = run_ocr(img, profile="nonexistent_profile")
    assert isinstance(text, str)
    assert isinstance(conf, float)
