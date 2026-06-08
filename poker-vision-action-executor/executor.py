"""
executor.py — Core action-execution logic for the poker action executor.

Finds the poker client window, locates the appropriate button (and the
bet-size input box for raise/bet actions), then performs the interaction.

Flow
----
    1. Normalise the requested action string.
    2. Look up configured button-label variants for that action.
    3. Find the poker client window via the Windows UI API.
    4. Scan the window's child ``Button`` controls for a label match.
    5. (raise/bet only) Enter the amount into the bet-size ``Edit`` control.
    6. Click the matched button using pyautogui screen coordinates.

Public API
----------
    execute(action, amount, dry_run, window_title_hint)  →  ActionResult
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pyautogui
import yaml
from bet_entry import enter_bet_amount
from models import ActionResult
from poker_window import (
    find_poker_window,
    focus_window,
    get_window_rect,
    list_child_buttons,
)

logger = logging.getLogger(__name__)

# ── Load config ────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

with _CONFIG_PATH.open() as _f:
    _CFG: dict = yaml.safe_load(_f)

_WINDOW_TITLE_HINTS: list[str] = _CFG.get("window_title_hints", [])
_BUTTON_LABELS: dict[str, list[str]] = _CFG.get("button_labels", {})
_PRE_DELAY: float = _CFG.get("pre_action_delay_ms", 200) / 1000.0
_POST_DELAY: float = _CFG.get("post_action_delay_ms", 100) / 1000.0

# Actions that must be accompanied by a numeric bet amount.
_RAISE_ACTIONS: frozenset[str] = frozenset({"raise", "bet"})


# ── Internal helpers ───────────────────────────────────────────────────────────


def _click_at_hwnd(hwnd: int) -> None:
    """Click the centre of *hwnd* using pyautogui screen coordinates."""
    left, top, right, bottom = get_window_rect(hwnd)
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    pyautogui.click(cx, cy)
    logger.info("Clicked hwnd=%d at (%d, %d)", hwnd, cx, cy)


def _find_action_button(parent_hwnd: int, action: str) -> int | None:
    """Search child ``Button`` controls for one matching *action*'s variants.

    Performs a case-insensitive prefix match so that dynamic labels such as
    "Call 75" are matched by the variant "Call".

    Args:
        parent_hwnd: hwnd of the poker client window.
        action:      Normalised action string (e.g. "fold", "raise").

    Returns:
        hwnd of the matched button, or None.
    """
    variants = [v.lower() for v in _BUTTON_LABELS.get(action, [])]
    if not variants:
        logger.warning("No button label variants configured for action %r", action)
        return None

    for hwnd, text in list_child_buttons(parent_hwnd):
        t = text.lower().strip()
        for variant in variants:
            if t == variant or t.startswith(variant):
                logger.info(
                    "Button matched: hwnd=%d text=%r variant=%r", hwnd, text, variant
                )
                return hwnd

    logger.warning(
        "No button found for action %r in window hwnd=%d", action, parent_hwnd
    )
    return None


# ── Public API ─────────────────────────────────────────────────────────────────


def execute(
    action: str,
    amount: int | None = None,
    dry_run: bool = False,
    window_title_hint: str | None = None,
) -> ActionResult:
    """Execute a poker action against the foreground poker client window.

    Args:
        action:             One of "fold", "call", "check", "raise", "bet".
                            "watching" is silently accepted and returns
                            success=True without doing anything.
        amount:             Required for "raise" and "bet" actions.
        dry_run:            If True, locate the button but do not click.
                            Useful for verifying detection without live impact.
        window_title_hint:  Override the configured window title hints with a
                            single hint (used by the test harness).

    Returns:
        ActionResult describing what happened.
    """
    action = action.lower().strip()

    # "watching" means the hero is not acting — nothing to do.
    if action == "watching":
        return ActionResult(
            success=True,
            action=action,
            amount=None,
            method="none",
            message="Action is 'watching' — nothing to execute",
        )

    if action not in _BUTTON_LABELS:
        return ActionResult(
            success=False,
            action=action,
            amount=None,
            method="none",
            message=(
                f"Unknown action {action!r}. Valid actions: {sorted(_BUTTON_LABELS)}"
            ),
        )

    if action in _RAISE_ACTIONS and amount is None:
        return ActionResult(
            success=False,
            action=action,
            amount=None,
            method="none",
            message=f"Action {action!r} requires an 'amount' to be supplied",
        )

    # ── Find window ────────────────────────────────────────────────────────────
    hints = [window_title_hint] if window_title_hint else _WINDOW_TITLE_HINTS
    poker_hwnd = find_poker_window(hints)
    if poker_hwnd is None:
        return ActionResult(
            success=False,
            action=action,
            amount=amount,
            method="none",
            message=f"No poker client window found matching hints: {hints}",
        )

    # ── Find action button ─────────────────────────────────────────────────────
    button_hwnd = _find_action_button(poker_hwnd, action)
    if button_hwnd is None:
        return ActionResult(
            success=False,
            action=action,
            amount=amount,
            method="none",
            message=f"Button for action {action!r} not found in window hwnd={poker_hwnd}",
        )

    if dry_run:
        left, top, right, bottom = get_window_rect(button_hwnd)
        cx = (left + right) // 2
        cy = (top + bottom) // 2
        return ActionResult(
            success=True,
            action=action,
            amount=amount,
            method="dry_run",
            message=(
                f"DRY RUN: would click {action!r} button at screen pos ({cx}, {cy})"
            ),
        )

    # ── Bring window to foreground ─────────────────────────────────────────────
    focus_window(poker_hwnd)
    time.sleep(_PRE_DELAY)

    # ── Enter bet amount (raise/bet only) ──────────────────────────────────────
    if action in _RAISE_ACTIONS:
        if not enter_bet_amount(poker_hwnd, amount):  # type: ignore[arg-type]
            return ActionResult(
                success=False,
                action=action,
                amount=amount,
                method="none",
                message="Could not locate the bet-size input box in the window",
            )

    # ── Click the action button ────────────────────────────────────────────────
    _click_at_hwnd(button_hwnd)
    time.sleep(_POST_DELAY)

    logger.info("Executed action=%r amount=%r", action, amount)
    return ActionResult(
        success=True,
        action=action,
        amount=amount,
        method="windows_api",
        message=f"Action {action!r} executed successfully",
    )
