"""
models.py — Shared data models for the action executor.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionResult:
    """Result of an action execution attempt.

    Attributes:
        success:  True if the action was carried out (or dry-run succeeded).
        action:   Normalised action string that was requested.
        amount:   Bet amount for raise/bet actions; None otherwise.
        method:   How the action was executed:
                    "windows_api" — button clicked via Win32 screen coordinates
                    "dry_run"     — button located but not clicked
                    "none"        — action was not executed (watching / error)
        message:  Human-readable outcome description.
    """

    success: bool
    action: str
    amount: int | None
    method: str
    message: str
