"""
test_harness/test_harness_integration.py — Integration tests for the action executor.

These tests spin up the Win32 test harness (which creates a real native window
with real Button and Edit controls), then call ``executor.execute()`` directly
against it.  Because the harness window is a real Win32 window, the executor's
EnumWindows / EnumChildWindows discovery and pyautogui click coordinates work
exactly as they would against a real poker client.

Run only with:
    uv run pytest test_harness/ -v -m integration

The tests are excluded from the default (unit) test run because they:
  - create real OS windows
  - fire real mouse events via pyautogui
  - require a graphical desktop (no headless CI)
"""

from __future__ import annotations

import time

import pytest

from test_harness.harness import run_harness_in_thread


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def harness():
    """Start the Win32 harness, yield state, wait for cleanup."""
    thread, state = run_harness_in_thread(auto_close_after=8.0)
    ready = state.ready.wait(timeout=5.0)
    if not ready:
        pytest.skip("Win32 harness window did not become ready in time")
    yield state
    state.done.wait(timeout=3.0)


# ── Helper ─────────────────────────────────────────────────────────────────────


def _exec(action: str, amount: int | None = None, dry_run: bool = False):
    from executor import execute

    return execute(action, amount=amount, dry_run=dry_run, window_title_hint="PokerTestHarness")


# ── Tests ──────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_integration_fold(harness):
    """Executor finds and clicks the Fold button; harness records the click."""
    result = _exec("fold")

    assert result.success is True, result.message
    assert result.method == "windows_api"
    assert "Fold" in harness.clicked, f"clicked={harness.clicked}"


@pytest.mark.integration
def test_integration_call(harness):
    """'Call 50' button is matched via prefix by the 'Call' variant."""
    result = _exec("call")

    assert result.success is True, result.message
    # The button caption in the harness is "Call 50"
    assert any(c.startswith("Call") for c in harness.clicked), f"clicked={harness.clicked}"


@pytest.mark.integration
def test_integration_check(harness):
    result = _exec("check")

    assert result.success is True, result.message
    assert any(c.startswith("Check") for c in harness.clicked), f"clicked={harness.clicked}"


@pytest.mark.integration
def test_integration_raise(harness):
    """Raise action should type the amount then click the Raise To button."""
    result = _exec("raise", amount=250)

    assert result.success is True, result.message
    # The bet-size Edit control should have been updated.
    assert harness.last_bet_value == "250", f"last_bet_value={harness.last_bet_value!r}"
    assert any(c.startswith("Raise") for c in harness.clicked), f"clicked={harness.clicked}"


@pytest.mark.integration
def test_integration_dry_run_does_not_click(harness):
    """dry_run=True should return method='dry_run' and not record any click."""
    initial_clicks = list(harness.clicked)
    result = _exec("fold", dry_run=True)

    assert result.success is True, result.message
    assert result.method == "dry_run"
    assert harness.clicked == initial_clicks, "dry_run should not fire a click"


@pytest.mark.integration
def test_integration_watching_skips_window_search():
    """'watching' never touches the UI — passes even with no harness running."""
    from executor import execute

    result = execute("watching")
    assert result.success is True
    assert result.method == "none"


@pytest.mark.integration
def test_integration_unknown_action_fails():
    from executor import execute

    result = execute("shove", window_title_hint="PokerTestHarness")
    assert result.success is False


@pytest.mark.integration
def test_integration_raise_without_amount_fails():
    from executor import execute

    result = execute("raise", window_title_hint="PokerTestHarness")
    assert result.success is False
    assert "amount" in result.message.lower()
