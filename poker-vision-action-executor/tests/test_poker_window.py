"""
tests/test_poker_window.py — Unit tests for poker_window.py.

All ctypes/Win32 calls are patched out so these tests run without needing real
windows to be present.  We patch the private helper functions (_get_window_text,
_get_class_name, _enum_child_windows) rather than ctypes windll attributes to
keep the tests portable and robust.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# ── list_child_buttons ─────────────────────────────────────────────────────────


def test_list_child_buttons_filters_by_configured_class_names() -> None:
    """Only controls matching the configured class names should be returned."""
    import poker_window as pw

    children = [10, 20, 30, 40]
    classes = {
        10: "AfxWnd140u",
        20: "Edit",
        30: "button",
        40: "AfxWnd140u",
    }
    texts = {10: "Fold", 20: "amount", 30: "Call", 40: "Check"}

    with (
        patch.object(pw, "_enum_child_windows", return_value=children),
        patch.object(pw, "_get_class_name", side_effect=lambda h: classes[h]),
        patch.object(pw, "_get_window_text", side_effect=lambda h: texts[h]),
    ):
        result = pw.list_child_buttons(99, class_names=["AfxWnd140u"])

    hwnds = [h for h, _ in result]
    assert 10 in hwnds
    assert 40 in hwnds
    assert 20 not in hwnds
    assert 30 not in hwnds


def test_list_child_buttons_strips_text_whitespace() -> None:
    import poker_window as pw

    with (
        patch.object(pw, "_enum_child_windows", return_value=[10]),
        patch.object(pw, "_get_class_name", return_value="Button"),
        patch.object(pw, "_get_window_text", return_value="  Fold  "),
    ):
        result = pw.list_child_buttons(99)

    assert result[0][1] == "Fold"


# ── list_child_edits ───────────────────────────────────────────────────────────


def test_list_child_edits_returns_edit_class_only() -> None:
    import poker_window as pw

    children = [10, 20]
    classes = {10: "Edit", 20: "Button"}
    texts = {10: "300", 20: "Raise"}

    with (
        patch.object(pw, "_enum_child_windows", return_value=children),
        patch.object(pw, "_get_class_name", side_effect=lambda h: classes[h]),
        patch.object(pw, "_get_window_text", side_effect=lambda h: texts[h]),
    ):
        result = pw.list_child_edits(99)

    assert len(result) == 1
    assert result[0] == (10, "300")


# ── find_poker_window ──────────────────────────────────────────────────────────


def test_find_poker_window_returns_none_when_no_hints() -> None:
    import poker_window as pw

    result = pw.find_poker_window(None)
    assert result is None

    result = pw.find_poker_window([])
    assert result is None


def test_find_poker_window_returns_none_when_no_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EnumWindows finds windows but none match the hints."""
    import poker_window as pw

    def fake_enum(callback: object, l_param: int) -> None:
        # Simulate one visible window with a non-matching title.
        assert callable(callback)
        callback(9999, l_param)

    monkeypatch.setattr(pw.user32, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(pw.user32, "EnumWindows", fake_enum)
    monkeypatch.setattr(pw, "_get_window_text", lambda h: "Notepad")

    result = pw.find_poker_window(["PokerStars"])
    assert result is None


def test_find_poker_window_returns_hwnd_on_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import poker_window as pw

    FAKE_HWND = 9999

    def fake_enum(callback: object, l_param: int) -> bool:
        assert callable(callback)
        callback(FAKE_HWND, l_param)
        return True

    monkeypatch.setattr(pw.user32, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(pw.user32, "EnumWindows", fake_enum)
    monkeypatch.setattr(pw, "_get_window_text", lambda h: "PokerStars Lobby")

    result = pw.find_poker_window(["PokerStars"])
    assert result == FAKE_HWND


def test_find_poker_window_case_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import poker_window as pw

    def fake_enum(callback: object, l_param: int) -> bool:
        assert callable(callback)
        callback(42, l_param)
        return True

    monkeypatch.setattr(pw.user32, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(pw.user32, "EnumWindows", fake_enum)
    monkeypatch.setattr(pw, "_get_window_text", lambda h: "POKERSTARS TABLE")

    result = pw.find_poker_window(["pokerstars"])
    assert result == 42
