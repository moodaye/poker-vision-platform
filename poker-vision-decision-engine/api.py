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
        "schema_version": "2.0.0",
        "hero_cards": ["Ah", "Kd"],
        "position": "BTN",
        "hero_seat": "BTN",
        "action_on": "BTN",
        "big_blind": 100,
        "small_blind": 50,
        "hero_stack": 3000,
        "pot": 150,
        "amount_to_call": 0,
        "tournament_status": {
            "current_blind_level": 12,
            "small_blind_amount": 50,
            "big_blind_amount": 100,
            "ante_amount": 10,
            "seconds_until_next_level": 90
        },
        "action_history": [],
        "is_hero_turn": true,
        "hero_folded": false
    }

`position` remains accepted for backward compatibility; `hero_seat` is preferred.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from decision_engine.controller import decide_next_action
from decision_engine.models import ActionEntry, HandState, SeatState, TournamentStatus
from flask import Flask, Response, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Required fields and their expected types ─────────────────────────────────

_REQUIRED_FIELDS: dict[str, type] = {
    "hero_cards": list,
    "big_blind": int,
    "small_blind": int,
    "hero_stack": int,
    "pot": int,
    "amount_to_call": int,
}

_VALID_POSITIONS = {"BTN", "SB", "BB"}
_VALID_ACTION_ON = {"BTN", "SB", "BB", "unknown", "none"}
_VALID_HAND_PHASES = {"preflop", "postflop"}


def _to_int_or_default(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_tournament_status(
    raw: Any,
    *,
    small_blind: int,
    big_blind: int,
) -> TournamentStatus:
    if not isinstance(raw, dict):
        return TournamentStatus(
            small_blind_amount=small_blind,
            big_blind_amount=big_blind,
        )

    return TournamentStatus(
        current_blind_level=_to_int_or_none(raw.get("current_blind_level")),
        small_blind_amount=_to_int_or_default(
            raw.get("small_blind_amount"),
            small_blind,
        ),
        big_blind_amount=_to_int_or_default(raw.get("big_blind_amount"), big_blind),
        ante_amount=_to_int_or_default(raw.get("ante_amount"), 0),
        seconds_until_next_level=_to_int_or_none(raw.get("seconds_until_next_level")),
    )


def _default_seats(
    *,
    hero_seat: str,
    hero_stack: int,
    hero_folded: bool,
    hero_has_cards: bool,
    is_hero_turn: bool,
) -> list[SeatState]:
    seats: list[SeatState] = []
    for seat in ("BTN", "SB", "BB"):
        is_hero = seat == hero_seat
        seats.append(
            SeatState(
                seat=seat,  # type: ignore[arg-type]
                is_hero=is_hero,
                status="deciding" if (is_hero and is_hero_turn) else "unknown",
                stack=hero_stack if is_hero else None,
                is_folded=hero_folded if is_hero else None,
                is_all_in=None,
                has_cards=hero_has_cards if is_hero else None,
            )
        )
    return seats


def _parse_seats(
    raw: Any,
    hero_seat: str,
    hero_stack: int,
    hero_folded: bool,
    is_hero_turn: bool,
) -> list[SeatState]:
    if not isinstance(raw, list):
        return _default_seats(
            hero_seat=hero_seat,
            hero_stack=hero_stack,
            hero_folded=hero_folded,
            hero_has_cards=True,
            is_hero_turn=is_hero_turn,
        )

    parsed: list[SeatState] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        seat = item.get("seat")
        if seat not in _VALID_POSITIONS:
            continue
        parsed.append(
            SeatState(
                seat=seat,
                is_hero=bool(item.get("is_hero", seat == hero_seat)),
                status=item.get("status", "unknown"),
                stack=item.get("stack"),
                is_folded=item.get("is_folded"),
                is_all_in=item.get("is_all_in"),
                has_cards=item.get("has_cards"),
                player_name=item.get("player_name"),
            )
        )

    if parsed:
        return parsed

    return _default_seats(
        hero_seat=hero_seat,
        hero_stack=hero_stack,
        hero_folded=hero_folded,
        hero_has_cards=True,
        is_hero_turn=is_hero_turn,
    )


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

    position = data.get("position")
    hero_seat = data.get("hero_seat", position)

    if position is not None and position not in _VALID_POSITIONS:
        return jsonify(
            {"error": f"'position' must be one of {sorted(_VALID_POSITIONS)}"}
        ), 400

    if hero_seat not in _VALID_POSITIONS:
        return jsonify(
            {"error": f"'hero_seat' must be one of {sorted(_VALID_POSITIONS)}"}
        ), 400

    action_on = data.get("action_on", "unknown")
    if action_on not in _VALID_ACTION_ON:
        return jsonify(
            {"error": f"'action_on' must be one of {sorted(_VALID_ACTION_ON)}"}
        ), 400

    hero_folded = bool(data.get("hero_folded", False))
    hand_phase = str(data.get("hand_phase", "preflop")).lower()
    if hand_phase not in _VALID_HAND_PHASES:
        return jsonify(
            {"error": f"'hand_phase' must be one of {sorted(_VALID_HAND_PHASES)}"}
        ), 400

    if len(data["hero_cards"]) == 0:
        if hero_folded:
            return jsonify(
                {
                    "action": "watching",
                    "amount": None,
                    "reason": "Hero has already folded",
                }
            )
        return jsonify(
            {
                "action": "watching",
                "amount": None,
                "reason": "Hero cards not exposed",
            }
        )

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

    is_hero_turn = data.get("is_hero_turn", True)

    state = HandState(
        hero_cards=data["hero_cards"],
        big_blind=data["big_blind"],
        small_blind=data["small_blind"],
        hero_stack=data["hero_stack"],
        pot=data["pot"],
        amount_to_call=data["amount_to_call"],
        hand_phase=hand_phase,
        schema_version=str(data.get("schema_version", "2.0.0")),
        hero_cards_visibility=data.get("hero_cards_visibility", "exposed"),
        position=position or hero_seat,
        hero_seat=hero_seat,
        action_on=action_on,
        seats=_parse_seats(
            data.get("seats"),
            hero_seat=hero_seat,
            hero_stack=data["hero_stack"],
            hero_folded=bool(hero_folded),
            is_hero_turn=bool(is_hero_turn),
        ),
        tournament_status=_parse_tournament_status(
            data.get("tournament_status"),
            small_blind=data["small_blind"],
            big_blind=data["big_blind"],
        ),
        action_history=action_history,
        is_hero_turn=is_hero_turn,
        hero_folded=hero_folded,
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
