"""
tests/test_api.py — Unit tests for the Flask API (api.py).

Uses Flask's test client; all executor logic is patched so no real windows
or mouse events are involved.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from models import ActionResult


@pytest.fixture()
def client():
    """Return a Flask test client for the action-executor API."""
    from api import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── /health ────────────────────────────────────────────────────────────────────


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


# ── /execute — request validation ─────────────────────────────────────────────


def test_execute_requires_json_body(client):
    resp = client.post("/execute", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_execute_requires_action_field(client):
    resp = client.post("/execute", json={})
    assert resp.status_code == 400
    assert "action" in resp.get_json()["error"].lower()


def test_execute_rejects_non_string_action(client):
    resp = client.post("/execute", json={"action": 42})
    assert resp.status_code == 400


def test_execute_rejects_non_integer_amount(client):
    resp = client.post("/execute", json={"action": "fold", "amount": "lots"})
    assert resp.status_code == 400


# ── /execute — successful responses ───────────────────────────────────────────


def _ok_result(action="fold", amount=None):
    return ActionResult(
        success=True,
        action=action,
        amount=amount,
        method="windows_api",
        message="ok",
    )


def test_execute_fold_returns_200(client):
    with patch("api.execute", return_value=_ok_result("fold")):
        resp = client.post("/execute", json={"action": "fold"})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["action"] == "fold"
    assert body["method"] == "windows_api"


def test_execute_raise_passes_amount_to_executor(client):
    with patch("api.execute") as mock_exec:
        mock_exec.return_value = _ok_result("raise", 300)
        resp = client.post("/execute", json={"action": "raise", "amount": 300})

    assert resp.status_code == 200
    mock_exec.assert_called_once_with(
        "raise", amount=300, dry_run=False, window_title_hint=None
    )


def test_execute_dry_run_flag_forwarded(client):
    with patch("api.execute") as mock_exec:
        mock_exec.return_value = ActionResult(
            success=True, action="fold", amount=None, method="dry_run", message="ok"
        )
        client.post("/execute", json={"action": "fold", "dry_run": True})

    mock_exec.assert_called_once_with(
        "fold", amount=None, dry_run=True, window_title_hint=None
    )


def test_execute_window_title_hint_forwarded(client):
    with patch("api.execute") as mock_exec:
        mock_exec.return_value = _ok_result()
        client.post(
            "/execute",
            json={"action": "fold", "window_title_hint": "MyPokerApp"},
        )

    mock_exec.assert_called_once_with(
        "fold", amount=None, dry_run=False, window_title_hint="MyPokerApp"
    )


def test_execute_amount_coerced_to_int(client):
    """Amount supplied as a string should be cast to int."""
    with patch("api.execute") as mock_exec:
        mock_exec.return_value = _ok_result("raise", 150)
        client.post("/execute", json={"action": "raise", "amount": "150"})

    _, kwargs = mock_exec.call_args
    assert kwargs["amount"] == 150


# ── /execute — failure responses ──────────────────────────────────────────────


def test_execute_returns_422_on_executor_failure(client):
    fail_result = ActionResult(
        success=False,
        action="fold",
        amount=None,
        method="none",
        message="Window not found",
    )
    with patch("api.execute", return_value=fail_result):
        resp = client.post("/execute", json={"action": "fold"})

    assert resp.status_code == 422
    body = resp.get_json()
    assert body["success"] is False
    assert "Window not found" in body["message"]


def test_execute_returns_500_on_unexpected_exception(client):
    with patch("api.execute", side_effect=RuntimeError("unexpected")):
        resp = client.post("/execute", json={"action": "fold"})

    assert resp.status_code == 500
    assert "error" in resp.get_json()


# ── /execute — response shape ──────────────────────────────────────────────────


def test_execute_response_contains_all_fields(client):
    with patch("api.execute", return_value=_ok_result("call")):
        resp = client.post("/execute", json={"action": "call"})

    body = resp.get_json()
    for key in ("success", "action", "amount", "method", "message"):
        assert key in body, f"Missing field: {key}"


def test_execute_amount_null_in_response_for_non_raise(client):
    with patch("api.execute", return_value=_ok_result("check")):
        resp = client.post("/execute", json={"action": "check"})

    assert resp.get_json()["amount"] is None


def test_execute_amount_present_in_response_for_raise(client):
    with patch("api.execute", return_value=_ok_result("raise", 250)):
        resp = client.post("/execute", json={"action": "raise", "amount": 250})

    assert resp.get_json()["amount"] == 250
