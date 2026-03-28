"""
api.py — Decision engine inference service.

Stateless: no model to load. Accepts a HandState as JSON and returns a Decision.

Run:
    uv run python api.py

Endpoints:
    POST /decide   body: HandState as JSON (see HandState dataclass in models.py)
                   returns: {"action": "...", "amount": ..., "reason": "..."}

    GET  /health   returns: {"status": "ok"}

Example request body:
    {
        "hero_cards": ["Ah", "Kd"],
        "position": "BTN",
        "big_blind": 100,
        "small_blind": 50,
        "hero_stack": 3000,
        "pot": 150,
        "amount_to_call": 0,
        "action_history": [],
        "is_hero_turn": true,
        "hero_folded": false
    }
"""

from __future__ import annotations

import dataclasses
import logging

from decision_engine.controller import decide_next_action
from decision_engine.models import ActionEntry, HandState
from flask import Flask, Response, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Required fields and their expected types ─────────────────────────────────

_REQUIRED_FIELDS: dict[str, type] = {
    "hero_cards": list,
    "position": str,
    "big_blind": int,
    "small_blind": int,
    "hero_stack": int,
    "pot": int,
    "amount_to_call": int,
}

_VALID_POSITIONS = {"BTN", "SB", "BB"}


# ── Routes ────────────────────────────────────────────────────────────────────


@app.route("/health")
def health() -> Response:
    return jsonify({"status": "ok"})


@app.route("/decide", methods=["POST"])
def decide() -> tuple[Response, int] | Response:
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    # Validate required fields are present and have correct types
    for field, expected_type in _REQUIRED_FIELDS.items():
        if field not in data:
            return jsonify({"error": f"Missing required field: '{field}'"}), 400
        if not isinstance(data[field], expected_type):
            return jsonify(
                {"error": f"Field '{field}' must be {expected_type.__name__}"}
            ), 400

    if data["position"] not in _VALID_POSITIONS:
        return jsonify(
            {"error": f"'position' must be one of {sorted(_VALID_POSITIONS)}"}
        ), 400

    if len(data["hero_cards"]) != 2:
        return jsonify({"error": "'hero_cards' must contain exactly 2 cards"}), 400

    # Parse action_history entries
    raw_history = data.get("action_history", [])
    try:
        action_history = [
            ActionEntry(
                player=entry["player"],
                action=entry["action"],
                amount=entry.get("amount"),
            )
            for entry in raw_history
        ]
    except (KeyError, TypeError) as e:
        return jsonify(
            {
                "error": f"Invalid action_history entry: {e}. Each entry needs 'player' and 'action'."
            }
        ), 400

    state = HandState(
        hero_cards=data["hero_cards"],
        position=data["position"],
        big_blind=data["big_blind"],
        small_blind=data["small_blind"],
        hero_stack=data["hero_stack"],
        pot=data["pot"],
        amount_to_call=data["amount_to_call"],
        action_history=action_history,
        is_hero_turn=data.get("is_hero_turn", True),
        hero_folded=data.get("hero_folded", False),
    )

    try:
        decision = decide_next_action(state)
    except Exception as e:
        logger.exception("Error in decide_next_action")
        return jsonify({"error": f"Internal error: {e}"}), 500

    logger.info(f"Decision: {decision.action} | {decision.reason}")
    return jsonify(dataclasses.asdict(decision))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)
