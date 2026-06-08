"""
tests/test_executor.py — Unit tests for executor.py.

All Win32 and pyautogui interactions are patched.  No real windows need to be
present and no mouse/keyboard events are fired.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

# ── Helpers ────────────────────────────────────────────────────────────────────


def _patch_executor(
    poker_hwnd: int | None = 99,
    button_hwnd: int | None = 55,
    button_rect: tuple[int, int, int, int] = (100, 200, 200, 230),
    enter_bet_ok: bool = True,
) -> tuple[Any, ...]:
    """Return a context manager stack that patches all executor dependencies."""
    return (
        patch("executor.find_poker_window", return_value=poker_hwnd),
        patch(
            "executor.list_child_buttons",
            return_value=[
                (55, "Fold"),
                (56, "Call 50"),
                (57, "Check"),
                (58, "Raise To"),
            ],
        ),
        patch("executor.get_window_rect", return_value=button_rect),
        patch("executor.focus_window"),
        patch("executor._click_at_hwnd"),
        patch("executor.enter_bet_amount", return_value=enter_bet_ok),
        patch("executor.time"),
    )


# ── "watching" action ──────────────────────────────────────────────────────────


def test_watching_returns_success_without_window_search() -> None:
    from executor import execute

    with patch("executor.find_poker_window") as mock_find:
        result = execute("watching")

    assert result.success is True
    assert result.method == "none"
    mock_find.assert_not_called()


def test_watching_case_insensitive() -> None:
    from executor import execute

    result = execute("Watching")
    assert result.success is True


# ── Unknown action ─────────────────────────────────────────────────────────────


def test_unknown_action_returns_failure() -> None:
    from executor import execute

    result = execute("shove")
    assert result.success is False
    assert "shove" in result.message


# ── Missing amount for raise/bet ───────────────────────────────────────────────


def test_raise_without_amount_returns_failure() -> None:
    from executor import execute

    result = execute("raise")
    assert result.success is False
    assert "amount" in result.message.lower()


def test_bet_without_amount_returns_failure() -> None:
    from executor import execute

    result = execute("bet")
    assert result.success is False


def test_raise_with_amount_does_not_fail_on_missing_amount_check() -> None:
    from executor import execute

    patches = _patch_executor()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
    ):
        result = execute("raise", amount=300)

    assert result.success is True


# ── Window not found ───────────────────────────────────────────────────────────


def test_fold_fails_when_no_window_found() -> None:
    from executor import execute

    with patch("executor.find_poker_window", return_value=None):
        result = execute("fold")

    assert result.success is False
    assert "window" in result.message.lower()


def test_fold_uses_configured_title_hints_by_default() -> None:
    from executor import _WINDOW_TITLE_HINTS, execute

    with patch("executor.find_poker_window", return_value=None) as mock_find:
        execute("fold")

    mock_find.assert_called_once_with(_WINDOW_TITLE_HINTS)


def test_window_title_hint_overrides_config() -> None:
    from executor import execute

    with patch("executor.find_poker_window", return_value=None) as mock_find:
        execute("fold", window_title_hint="MyPokerApp")

    mock_find.assert_called_once_with(["MyPokerApp"])


# ── Button not found ───────────────────────────────────────────────────────────


def test_fold_fails_when_no_button_found() -> None:
    from executor import execute

    with (
        patch("executor.find_poker_window", return_value=99),
        patch(
            "executor.list_child_buttons", return_value=[(10, "Call"), (11, "Raise")]
        ),
    ):
        result = execute("fold")

    assert result.success is False
    assert "fold" in result.message.lower()


# ── Simple actions (fold / call / check) ───────────────────────────────────────


def test_fold_success() -> None:
    from executor import execute

    patches = _patch_executor()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4] as mock_click,
        patches[5],
        patches[6],
    ):
        result = execute("fold")

    assert result.success is True
    assert result.action == "fold"
    assert result.method == "windows_api"
    mock_click.assert_called_once()


def test_call_matches_dynamic_label() -> None:
    """'Call 50' button should be matched by the 'Call' variant (prefix match)."""
    from executor import execute

    patches = _patch_executor()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4] as mock_click,
        patches[5],
        patches[6],
    ):
        result = execute("call")

    assert result.success is True
    mock_click.assert_called_once()


def test_check_success() -> None:
    from executor import execute

    patches = _patch_executor()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4] as mock_click,
        patches[5],
        patches[6],
    ):
        result = execute("check")

    assert result.success is True
    mock_click.assert_called_once()


# ── Raise / bet ────────────────────────────────────────────────────────────────


def test_raise_calls_enter_bet_amount_then_clicks() -> None:
    from executor import execute

    patches = _patch_executor()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4] as mock_click,
        patches[5] as mock_bet,
        patches[6],
    ):
        result = execute("raise", amount=300)

    assert result.success is True
    mock_bet.assert_called_once_with(99, 300)
    mock_click.assert_called_once()


def test_raise_fails_when_bet_entry_not_found() -> None:
    from executor import execute

    patches = _patch_executor(enter_bet_ok=False)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4] as mock_click,
        patches[5],
        patches[6],
    ):
        result = execute("raise", amount=300)

    assert result.success is False
    assert "bet" in result.message.lower()
    mock_click.assert_not_called()


def test_bet_success() -> None:
    from executor import execute

    with (
        patch("executor.find_poker_window", return_value=99),
        patch("executor.list_child_buttons", return_value=[(58, "Bet")]),
        patch("executor.get_window_rect", return_value=(0, 0, 100, 30)),
        patch("executor.focus_window"),
        patch("executor._click_at_hwnd") as mock_click,
        patch("executor.enter_bet_amount", return_value=True),
        patch("executor.time"),
    ):
        result = execute("bet", amount=100)

    assert result.success is True
    mock_click.assert_called_once()


# ── Dry run ────────────────────────────────────────────────────────────────────


def test_dry_run_does_not_click() -> None:
    from executor import execute

    patches = _patch_executor()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4] as mock_click,
        patches[5],
        patches[6],
    ):
        result = execute("fold", dry_run=True)

    assert result.success is True
    assert result.method == "dry_run"
    assert "DRY RUN" in result.message
    mock_click.assert_not_called()


def test_dry_run_does_not_enter_bet_amount() -> None:
    from executor import execute

    patches = _patch_executor()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5] as mock_bet,
        patches[6],
    ):
        execute("raise", amount=300, dry_run=True)

    mock_bet.assert_not_called()


def test_dry_run_includes_screen_position_in_message() -> None:
    from executor import execute

    with (
        patch("executor.find_poker_window", return_value=99),
        patch("executor.list_child_buttons", return_value=[(55, "Fold")]),
        patch("executor.get_window_rect", return_value=(100, 200, 200, 230)),
        patch("executor.focus_window"),
        patch("executor._click_at_hwnd"),
        patch("executor.enter_bet_amount", return_value=True),
        patch("executor.time"),
    ):
        result = execute("fold", dry_run=True)

    assert "150" in result.message  # cx = (100+200)//2
    assert "215" in result.message  # cy = (200+230)//2


# ── action case normalisation ──────────────────────────────────────────────────


def test_action_normalised_to_lowercase() -> None:
    from executor import execute

    patches = _patch_executor()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
    ):
        result = execute("FOLD")

    assert result.action == "fold"
    assert result.success is True
