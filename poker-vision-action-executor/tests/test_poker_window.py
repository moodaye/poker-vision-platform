"""
tests/test_poker_window.py — Unit tests for poker_window.py.

All ctypes/Win32 calls are patched out so these tests run without needing real
windows to be present.  We patch the private helper functions (_get_window_text,
_get_class_name, _enum_child_windows) rather than ctypes windll attributes to
keep the tests portable and robust.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ── list_child_buttons ─────────────────────────────────────────────────────────


def test_list_child_buttons_returns_button_class_only():
    """Only controls with class 'Button' should be returned."""
    import poker_window as pw

    children = [10, 20, 30]
    classes = {10: "Button", 20: "Edit", 30: "button"}  # case-insensitive
    texts = {10: "Fold", 20: "amount", 30: "Call"}

    with patch.object(pw, "_enum_child_windows", return_value=children), \
         patch.object(pw, "_get_class_name", side_effect=lambda h: classes[h]), \
         patch.object(pw, "_get_window_text", side_effect=lambda h: texts[h]):
        result = pw.list_child_buttons(99)

    hwnds = [h for h, _ in result]
    assert 10 in hwnds  # "Button"
    assert 20 not in hwnds  # "Edit" — excluded
    assert 30 in hwnds  # "button" — case-insensitive match


def test_list_child_buttons_strips_text_whitespace():
    import poker_window as pw

    with patch.object(pw, "_enum_child_windows", return_value=[10]), \
         patch.object(pw, "_get_class_name", return_value="Button"), \
         patch.object(pw, "_get_window_text", return_value="  Fold  "):
        result = pw.list_child_buttons(99)

    assert result[0][1] == "Fold"


# ── list_child_edits ───────────────────────────────────────────────────────────


def test_list_child_edits_returns_edit_class_only():
    import poker_window as pw

    children = [10, 20]
    classes = {10: "Edit", 20: "Button"}
    texts = {10: "300", 20: "Raise"}

    with patch.object(pw, "_enum_child_windows", return_value=children), \
         patch.object(pw, "_get_class_name", side_effect=lambda h: classes[h]), \
         patch.object(pw, "_get_window_text", side_effect=lambda h: texts[h]):
        result = pw.list_child_edits(99)

    assert len(result) == 1
    assert result[0] == (10, "300")


# ── find_poker_window ──────────────────────────────────────────────────────────


def test_find_poker_window_returns_none_when_no_hints():
    import poker_window as pw

    result = pw.find_poker_window(None)
    assert result is None

    result = pw.find_poker_window([])
    assert result is None


def test_find_poker_window_returns_none_when_no_match(monkeypatch):
    """EnumWindows finds windows but none match the hints."""
    import poker_window as pw

    def fake_enum(callback, lParam):
        # Simulate one visible window with a non-matching title.
        callback(9999, 0)

    monkeypatch.setattr(pw.user32, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(pw.user32, "EnumWindows", fake_enum)
    monkeypatch.setattr(pw, "_get_window_text", lambda h: "Notepad")

    result = pw.find_poker_window(["PokerStars"])
    assert result is None


def test_find_poker_window_returns_hwnd_on_match(monkeypatch):
    import poker_window as pw

    FAKE_HWND = 9999

    def fake_enum(callback, lParam):
        callback(FAKE_HWND, 0)
        return True

    monkeypatch.setattr(pw.user32, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(pw.user32, "EnumWindows", fake_enum)
    monkeypatch.setattr(pw, "_get_window_text", lambda h: "PokerStars Lobby")

    result = pw.find_poker_window(["PokerStars"])
    assert result == FAKE_HWND


def test_find_poker_window_case_insensitive(monkeypatch):
    import poker_window as pw

    def fake_enum(callback, lParam):
        callback(42, 0)
        return True

    monkeypatch.setattr(pw.user32, "IsWindowVisible", lambda h: True)
    monkeypatch.setattr(pw.user32, "EnumWindows", fake_enum)
    monkeypatch.setattr(pw, "_get_window_text", lambda h: "POKERSTARS TABLE")

    result = pw.find_poker_window(["pokerstars"])
    assert result == 42
