"""
poker_window.py — Locate and focus the poker client window using the Windows UI API.

All window and control discovery is done through ctypes calls to user32.dll so
that no external packages are required (pywin32 is not a dependency).

Public API
----------
    find_poker_window(title_hints)  → hwnd (int) or None
    focus_window(hwnd)              → None
    list_child_buttons(hwnd)        → list[tuple[int, str]]
    list_child_edits(hwnd)          → list[tuple[int, str]]
    get_window_rect(hwnd)           → tuple[int, int, int, int]
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging

logger = logging.getLogger(__name__)

user32 = ctypes.windll.user32

# ── Low-level Win32 helpers ────────────────────────────────────────────────────


def _get_window_text(hwnd: int) -> str:
    """Return the text/caption of a window handle."""
    length = user32.GetWindowTextLengthW(hwnd) + 1
    buf = ctypes.create_unicode_buffer(length)
    user32.GetWindowTextW(hwnd, buf, length)
    return buf.value


def _get_class_name(hwnd: int) -> str:
    """Return the Win32 class name of a window handle."""
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def _enum_child_windows(parent_hwnd: int) -> list[int]:
    """Return the hwnd of every direct child window of *parent_hwnd*."""
    children: list[int] = []

    EnumChildProc = ctypes.WINFUNCTYPE(
        ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
    )

    @EnumChildProc
    def callback(hwnd: int, _lParam: int) -> bool:
        children.append(hwnd)
        return True

    user32.EnumChildWindows(parent_hwnd, callback, 0)
    return children


def _walk_child_windows(parent_hwnd: int) -> list[int]:
    """Return the hwnd of every descendant window of *parent_hwnd*."""
    children: list[int] = []
    for child in _enum_child_windows(parent_hwnd):
        children.append(child)
        children.extend(_walk_child_windows(child))
    return children


# ── Public helpers ─────────────────────────────────────────────────────────────


def get_window_rect(hwnd: int) -> tuple[int, int, int, int]:
    """Return the bounding rectangle of a window in screen coordinates.

    Returns:
        (left, top, right, bottom)
    """
    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.left, rect.top, rect.right, rect.bottom


def list_child_buttons(
    parent_hwnd: int, class_names: list[str] | None = None
) -> list[tuple[int, str]]:
    """Return all descendant controls matching the configured button class names.

    Args:
        parent_hwnd: hwnd of the poker client window.
        class_names: Optional list of Win32 class names to include.
                     If None, defaults to ["button"].

    Returns:
        List of ``(hwnd, button_text)`` tuples.
    """
    if class_names is None:
        class_names = ["button"]
    class_names_lower = [c.lower() for c in class_names]

    results: list[tuple[int, str]] = []
    for child in _walk_child_windows(parent_hwnd):
        if _get_class_name(child).lower() in class_names_lower:
            text = _get_window_text(child).strip()
            results.append((child, text))
    return results


def list_child_edits(parent_hwnd: int) -> list[tuple[int, str]]:
    """Return all descendant controls with Win32 class ``Edit``.

    Returns:
        List of ``(hwnd, current_text)`` tuples.
    """
    results: list[tuple[int, str]] = []
    for child in _walk_child_windows(parent_hwnd):
        if _get_class_name(child).lower() == "edit":
            text = _get_window_text(child).strip()
            results.append((child, text))
    return results


def focus_window(hwnd: int) -> None:
    """Restore and bring *hwnd* to the foreground."""
    SW_RESTORE = 9
    user32.ShowWindow(hwnd, SW_RESTORE)
    user32.SetForegroundWindow(hwnd)


def find_poker_window(title_hints: list[str] | None = None) -> int | None:
    """Find the first visible top-level window whose title contains any hint.

    Performs a case-insensitive substring match against each hint in order.
    Stops at the first match.

    Args:
        title_hints: List of substrings to match against window titles.
                     If empty or None, returns None immediately.

    Returns:
        hwnd of the matched window, or None if no match is found.
    """
    if not title_hints:
        return None

    found: list[int] = []

    EnumWindowsProc = ctypes.WINFUNCTYPE(
        ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
    )

    @EnumWindowsProc
    def callback(hwnd: int, _lParam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        title = _get_window_text(hwnd)
        for hint in title_hints:
            if hint.lower() in title.lower():
                found.append(hwnd)
                logger.info("Poker window found: hwnd=%d title=%r", hwnd, title)
                return False  # stop — first match wins
        return True

    user32.EnumWindows(callback, 0)
    return found[0] if found else None
