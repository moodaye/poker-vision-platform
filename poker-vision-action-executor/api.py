"""
api.py — Action Executor service.

Receives a poker decision from the orchestrator and enacts it against the
live poker client window using the Windows UI API.

Run:
    uv run python api.py

Endpoints:
    POST /execute
        Request body (JSON):
            {
                "action":             "fold" | "call" | "check" | "raise" | "bet",
                "amount":             <int | null>,        # required for raise/bet
                "dry_run":            <bool, default false>,
                "window_title_hint":  <str | null>         # overrides config hints
            }
        Response (JSON):
            {
                "success": <bool>,
                "action":  <str>,
                "amount":  <int | null>,
                "method":  "windows_api" | "dry_run" | "none",
                "message": <str>
            }
        HTTP 200  on success
        HTTP 422  when the action could not be executed (window/button not found)
        HTTP 400  for malformed requests

    GET /health
        Returns {"status": "ok"}

Examples:
    {"action": "fold"}
    {"action": "call"}
    {"action": "raise", "amount": 300}
    {"action": "raise", "amount": 300, "dry_run": true}
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from executor import execute
from flask import Flask, Response, jsonify, request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.route("/health")
def health() -> Response:
    return jsonify({"status": "ok"})


@app.route("/execute", methods=["POST"])
def execute_action() -> tuple[Response, int] | Response:
    """Execute a poker action against the live client window."""
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    action = data.get("action")
    if not action or not isinstance(action, str):
        return jsonify(
            {"error": "'action' field is required and must be a string"}
        ), 400

    raw_amount = data.get("amount")
    amount: int | None = None
    if raw_amount is not None:
        try:
            amount = int(raw_amount)
        except (TypeError, ValueError):
            return jsonify({"error": "'amount' must be an integer"}), 400

    dry_run: bool = bool(data.get("dry_run", False))
    window_title_hint: str | None = data.get("window_title_hint") or None

    try:
        result = execute(
            action,
            amount=amount,
            dry_run=dry_run,
            window_title_hint=window_title_hint,
        )
    except Exception as exc:
        logger.exception("Unexpected error during action execution")
        return jsonify({"error": f"Execution error: {exc}"}), 500

    status = 200 if result.success else 422
    return jsonify(
        {
            "success": result.success,
            "action": result.action,
            "amount": result.amount,
            "method": result.method,
            "message": result.message,
        }
    ), status


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _config_path = Path(__file__).parent / "config.yaml"
    with _config_path.open() as _f:
        _cfg = yaml.safe_load(_f)
    port: int = int(_cfg.get("port", 5005))
    app.run(host="0.0.0.0", port=port, debug=False)
