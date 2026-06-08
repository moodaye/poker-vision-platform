"""
bet_entry.py — Locate the bet-size input field inside the poker client window
and type a numeric amount into it.

Strategy
--------
1. Find all Win32 ``Edit`` child controls inside the poker window.
2. Keep only those whose current text is numeric (or empty) — this filters
   out chat boxes and other text fields.
3. Focus the parent window, click the edit control, Ctrl+A to clear, then
   type the desired amount using pyautogui.

pyautogui is used for keyboard simulation so the keystrokes pass through
the standard Windows message pump and work with both native Win32 apps and
any poker client that renders its own input field.

Public API
----------
    find_bet_input(parent_hwnd)             → hwnd (int) or None
    enter_bet_amount(parent_hwnd, amount)   → bool
"""

from __future__ import annotations

import logging
import time

import pyautogui
from poker_window import focus_window, get_window_rect, list_child_edits

logger = logging.getLogger(__name__)

# Brief settle delay (seconds) between UI interactions to avoid races.
_SETTLE = 0.05


def _is_numeric_text(text: str) -> bool:
    """Return True if *text* is empty or contains only digits and separators."""
    cleaned = text.replace(",", "").replace(".", "").strip()
    return cleaned == "" or cleaned.isdigit()


def find_bet_input(parent_hwnd: int) -> int | None:
    """Return the hwnd of the bet-size ``Edit`` control, or None.

    Selects the first child ``Edit`` control whose current text is numeric or
    empty, which distinguishes bet-size boxes from chat / search fields.

    Args:
        parent_hwnd: hwnd of the poker client window.

    Returns:
        hwnd of the edit control, or None if not found.
    """
    edits = list_child_edits(parent_hwnd)
    numeric_edits = [(h, t) for h, t in edits if _is_numeric_text(t)]
    if not numeric_edits:
        logger.warning("No numeric Edit control found in hwnd=%d", parent_hwnd)
        return None
    hwnd, text = numeric_edits[0]
    logger.info("Bet input located: hwnd=%d current_value=%r", hwnd, text)
    return hwnd


def enter_bet_amount(parent_hwnd: int, amount: int) -> bool:
    """Enter *amount* into the bet-size input box of *parent_hwnd*.

    Steps:
        1. Locate the Edit control.
        2. Bring the parent window to the foreground.
        3. Click the edit control to give it focus.
        4. Ctrl+A to select any existing content.
        5. Type the new amount (digits only).

    Args:
        parent_hwnd: hwnd of the poker client window.
        amount:      Positive integer bet amount.

    Returns:
        True if the amount was entered successfully; False if the edit control
        could not be found.

    Raises:
        ValueError: if *amount* is not a positive integer.
    """
    if amount <= 0:
        raise ValueError(f"Bet amount must be a positive integer, got {amount!r}")

    edit_hwnd = find_bet_input(parent_hwnd)
    if edit_hwnd is None:
        return False

    left, top, right, bottom = get_window_rect(edit_hwnd)
    cx = (left + right) // 2
    cy = (top + bottom) // 2

    focus_window(parent_hwnd)
    time.sleep(_SETTLE)

    pyautogui.click(cx, cy)
    time.sleep(_SETTLE)

    pyautogui.hotkey("ctrl", "a")
    time.sleep(_SETTLE)

    pyautogui.write(str(amount), interval=0.02)

    logger.info("Entered bet amount=%d at screen pos (%d, %d)", amount, cx, cy)
    return True
