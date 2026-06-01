"""
tests/test_bet_entry.py — Unit tests for bet_entry.py.

pyautogui and Win32 calls are patched so no real mouse/keyboard events are
fired and no real windows are required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── _is_numeric_text ───────────────────────────────────────────────────────────


def test_is_numeric_text_empty_string() -> None:
    from bet_entry import _is_numeric_text

    assert _is_numeric_text("") is True


def test_is_numeric_text_digits_only() -> None:
    from bet_entry import _is_numeric_text

    assert _is_numeric_text("300") is True


def test_is_numeric_text_with_comma_separator() -> None:
    from bet_entry import _is_numeric_text

    assert _is_numeric_text("1,500") is True


def test_is_numeric_text_with_decimal() -> None:
    from bet_entry import _is_numeric_text

    assert _is_numeric_text("12.5") is True


def test_is_numeric_text_rejects_alpha() -> None:
    from bet_entry import _is_numeric_text

    assert _is_numeric_text("Call") is False


def test_is_numeric_text_rejects_mixed() -> None:
    from bet_entry import _is_numeric_text

    assert _is_numeric_text("300abc") is False


# ── find_bet_input ─────────────────────────────────────────────────────────────


def test_find_bet_input_returns_first_numeric_edit() -> None:
    from bet_entry import find_bet_input

    with patch(
        "bet_entry.list_child_edits", return_value=[(10, "0"), (20, "chat box")]
    ):
        result = find_bet_input(99)

    assert result == 10


def test_find_bet_input_skips_non_numeric_edits() -> None:
    from bet_entry import find_bet_input

    with patch(
        "bet_entry.list_child_edits", return_value=[(10, "Enter chat"), (20, "300")]
    ):
        result = find_bet_input(99)

    assert result == 20


def test_find_bet_input_returns_none_when_no_numeric_edit() -> None:
    from bet_entry import find_bet_input

    with patch("bet_entry.list_child_edits", return_value=[(10, "Enter chat")]):
        result = find_bet_input(99)

    assert result is None


def test_find_bet_input_returns_none_when_no_edits() -> None:
    from bet_entry import find_bet_input

    with patch("bet_entry.list_child_edits", return_value=[]):
        result = find_bet_input(99)

    assert result is None


# ── enter_bet_amount ───────────────────────────────────────────────────────────


def test_enter_bet_amount_raises_on_zero() -> None:
    from bet_entry import enter_bet_amount

    with pytest.raises(ValueError, match="positive integer"):
        enter_bet_amount(99, 0)


def test_enter_bet_amount_raises_on_negative() -> None:
    from bet_entry import enter_bet_amount

    with pytest.raises(ValueError, match="positive integer"):
        enter_bet_amount(99, -50)


def test_enter_bet_amount_returns_false_when_no_edit() -> None:
    from bet_entry import enter_bet_amount

    with patch("bet_entry.find_bet_input", return_value=None):
        result = enter_bet_amount(99, 300)

    assert result is False


def test_enter_bet_amount_types_correct_value() -> None:
    """Should focus, click, Ctrl+A, then write the amount string."""
    from bet_entry import enter_bet_amount

    with (
        patch("bet_entry.find_bet_input", return_value=55),
        patch("bet_entry.get_window_rect", return_value=(100, 200, 180, 232)),
        patch("bet_entry.focus_window") as mock_focus,
        patch("bet_entry.pyautogui") as mock_pg,
        patch("bet_entry.time") as mock_time,
    ):
        mock_time.sleep = MagicMock()
        result = enter_bet_amount(99, 300)

    assert result is True
    mock_focus.assert_called_once_with(99)
    # click at centre of rect (100+180)//2=140, (200+232)//2=216
    mock_pg.click.assert_called_once_with(140, 216)
    mock_pg.hotkey.assert_called_once_with("ctrl", "a")
    mock_pg.write.assert_called_once_with("300", interval=0.02)


def test_enter_bet_amount_correct_screen_centre() -> None:
    """Click coordinates should be the centre of the edit control rect."""
    from bet_entry import enter_bet_amount

    with (
        patch("bet_entry.find_bet_input", return_value=55),
        patch("bet_entry.get_window_rect", return_value=(200, 300, 280, 332)),
        patch("bet_entry.focus_window"),
        patch("bet_entry.pyautogui") as mock_pg,
        patch("bet_entry.time"),
    ):
        enter_bet_amount(99, 150)

    mock_pg.click.assert_called_once_with(240, 316)
