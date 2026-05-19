"""End-to-end integration tests: enricher payload → hand_state_parser → decision engine.

These tests exercise the full in-process pipeline without any running services.
They import directly from both sub-packages (paths added by conftest.py).

Each scenario represents a realistic preflop situation and asserts both:
  - the hand state parsed from the enriched payload is correct, and
  - the decision engine returns a sensible action for that state.
"""

from __future__ import annotations

from typing import Any

from decision_engine.controller import decide_next_action
from decision_engine.models import ActionEntry, HandState
from hand_state_parser import build_hand_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hand_state_from_enriched(payload: dict[str, Any]) -> HandState:
    """Convert an enriched payload all the way to a HandState dataclass.

    This replicates the logic in poker-vision-decision-engine/api.py so that
    the integration test exercises the same conversion path used in production.
    """
    data = build_hand_state(payload)

    action_history = [
        ActionEntry(
            player=entry["player"],
            action=entry["action"],
            amount=entry.get("amount"),
        )
        for entry in data.get("action_history", [])
    ]

    return HandState(
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


# ---------------------------------------------------------------------------
# Scenario 1: BTN opens with AK in an unopened pot
# ---------------------------------------------------------------------------


def test_btn_ak_unopened_raises() -> None:
    """BTN with AK facing no previous action should open-raise."""
    payload = {
        "objects": [
            {
                "class_name": "holecard",
                "classification": "Ah",
                "confidence": 0.95,
                "classification_conf": 0.96,
            },
            {
                "class_name": "holecard",
                "classification": "Kd",
                "confidence": 0.93,
                "classification_conf": 0.95,
            },
            {"class_name": "blinds", "ocr_text": "50/100", "ocr_conf": 0.90},
            {"class_name": "chip_stack", "ocr_text": "3000", "ocr_conf": 0.91},
            {"class_name": "pot", "ocr_text": "150", "ocr_conf": 0.88},
            # no bet object → amount_to_call = 0
            {
                "class_name": "dealer_button",
                "spatial_info": {"hero_position": "BTN"},
                "spatial_conf": 0.85,
                "confidence": 0.92,
            },
            {"class_name": "bet_box", "confidence": 0.90},
        ]
    }

    state = _hand_state_from_enriched(payload)

    # Parser assertions
    assert state.hero_cards == ["Ah", "Kd"]
    assert state.position == "BTN"
    assert state.big_blind == 100
    assert state.small_blind == 50
    assert state.hero_stack == 3000
    assert state.amount_to_call == 0
    assert state.is_hero_turn is True
    assert state.hero_folded is False

    # Decision engine assertions
    decision = decide_next_action(state)
    assert decision.action == "raise", (
        f"Expected raise, got {decision.action!r}: {decision.reason}"
    )
    assert decision.amount is not None
    assert decision.amount > 0


# ---------------------------------------------------------------------------
# Scenario 2: Hero has already folded → watching
# ---------------------------------------------------------------------------


def test_hero_folded_returns_watching() -> None:
    """When the fold button was already pressed the engine should return 'watching'."""
    payload = {
        "objects": [
            {
                "class_name": "holecard",
                "classification": "Tc",
                "confidence": 0.92,
                "classification_conf": 0.91,
            },
            {
                "class_name": "holecard",
                "classification": "7h",
                "confidence": 0.90,
                "classification_conf": 0.88,
            },
            {"class_name": "blinds", "ocr_text": "50/100", "ocr_conf": 0.88},
            {"class_name": "chip_stack", "ocr_text": "2800", "ocr_conf": 0.87},
            {"class_name": "pot", "ocr_text": "300", "ocr_conf": 0.85},
            # action_history shows BTN folded
            {
                "class_name": "action",
                "player": "BTN",
                "action": "fold",
                "confidence": 0.85,
            },
            {
                "class_name": "dealer_button",
                "spatial_info": {"hero_position": "BTN"},
                "spatial_conf": 0.80,
                "confidence": 0.90,
            },
        ]
    }

    state = _hand_state_from_enriched(payload)
    assert state.hero_folded is True

    decision = decide_next_action(state)
    assert decision.action == "watching"
    assert decision.amount is None


# ---------------------------------------------------------------------------
# Scenario 3: Not hero's turn → watching
# ---------------------------------------------------------------------------


def test_not_hero_turn_returns_watching() -> None:
    """No hero action controls detected → parser defaults is_hero_turn to True,
    but when we override it in the state the engine must return 'watching'."""
    payload = {
        "objects": [
            {
                "class_name": "holecard",
                "classification": "Qs",
                "confidence": 0.91,
                "classification_conf": 0.90,
            },
            {
                "class_name": "holecard",
                "classification": "Qh",
                "confidence": 0.93,
                "classification_conf": 0.92,
            },
            {"class_name": "blinds", "ocr_text": "50/100", "ocr_conf": 0.89},
            {"class_name": "chip_stack", "ocr_text": "4000", "ocr_conf": 0.90},
            {"class_name": "pot", "ocr_text": "200", "ocr_conf": 0.87},
            # No raise_button / fold_button → parser defaults is_hero_turn=True
            # We manually set it False to simulate waiting
        ]
    }

    data = build_hand_state(payload)
    state = HandState(
        hero_cards=data["hero_cards"],
        position=data["position"],
        big_blind=data["big_blind"],
        small_blind=data["small_blind"],
        hero_stack=data["hero_stack"],
        pot=data["pot"],
        amount_to_call=data["amount_to_call"],
        action_history=[],
        is_hero_turn=False,  # explicitly not our turn
        hero_folded=False,
    )

    decision = decide_next_action(state)
    assert decision.action == "watching"
    assert decision.amount is None


# ---------------------------------------------------------------------------
# Scenario 4: Short-stack shove with a premium hand
# ---------------------------------------------------------------------------


def test_short_stack_premium_shoves() -> None:
    """Hero has < 10 BB and holds AA → engine should shove regardless of position."""
    payload = {
        "objects": [
            {
                "class_name": "holecard",
                "classification": "As",
                "confidence": 0.96,
                "classification_conf": 0.97,
            },
            {
                "class_name": "holecard",
                "classification": "Ac",
                "confidence": 0.95,
                "classification_conf": 0.96,
            },
            {"class_name": "blinds", "ocr_text": "100/200", "ocr_conf": 0.91},
            # hero_stack = 1500 → 7.5 BB → short stack
            {"class_name": "chip_stack", "ocr_text": "1500", "ocr_conf": 0.90},
            {"class_name": "pot", "ocr_text": "300", "ocr_conf": 0.86},
            {"class_name": "bet_box", "confidence": 0.88},
        ]
    }

    state = _hand_state_from_enriched(payload)

    assert state.hero_stack == 1500
    assert state.big_blind == 200

    decision = decide_next_action(state)
    assert decision.action == "raise", (
        f"Expected shove (raise), got {decision.action!r}: {decision.reason}"
    )
    assert decision.amount == float(state.hero_stack), (
        "Short-stack shove should be for full stack"
    )


# ---------------------------------------------------------------------------
# Scenario 5: Facing a raise with a weak hand → fold
# ---------------------------------------------------------------------------


def test_facing_raise_with_weak_hand_folds() -> None:
    """BB facing a BTN open-raise with 72o should fold."""
    payload = {
        "objects": [
            {
                "class_name": "holecard",
                "classification": "7c",
                "confidence": 0.91,
                "classification_conf": 0.90,
            },
            {
                "class_name": "holecard",
                "classification": "2h",
                "confidence": 0.90,
                "classification_conf": 0.89,
            },
            {"class_name": "blinds", "ocr_text": "50/100", "ocr_conf": 0.90},
            {"class_name": "chip_stack", "ocr_text": "3000", "ocr_conf": 0.88},
            {"class_name": "pot", "ocr_text": "350", "ocr_conf": 0.85},
            # BTN raised to 250 → amount_to_call = 250 > BB (100)
            {"class_name": "bet", "ocr_text": "250", "ocr_conf": 0.87},
            {
                "class_name": "dealer_button",
                "spatial_info": {"hero_position": "BB"},
                "spatial_conf": 0.82,
                "confidence": 0.91,
            },
            {"class_name": "bet_box", "confidence": 0.89},
        ]
    }

    state = _hand_state_from_enriched(payload)

    assert state.position == "BB"
    assert state.amount_to_call == 250

    decision = decide_next_action(state)
    assert decision.action == "fold", (
        f"Expected fold, got {decision.action!r}: {decision.reason}"
    )


# ---------------------------------------------------------------------------
# Scenario 6: Degraded vision — all objects below confidence thresholds
# ---------------------------------------------------------------------------


def test_degraded_vision_falls_back_and_still_returns_decision() -> None:
    """When confidence is degraded the parser may use relaxed card candidates
    while still falling back on OCR-derived numeric fields as needed."""
    payload = {
        "objects": [
            # Cards below strict gate are accepted by relaxed card selection.
            {
                "class_name": "holecard",
                "classification": "Tc",
                "confidence": 0.40,
                "classification_conf": 0.91,
            },
            {
                "class_name": "holecard",
                "classification": "7h",
                "confidence": 0.40,
                "classification_conf": 0.88,
            },
            # Blinds below usable threshold → fallback 50/100
            {"class_name": "blinds", "ocr_text": "25/50", "ocr_conf": 0.30},
            # Stack below usable threshold → fallback 3000
            {"class_name": "chip_stack", "ocr_text": "2000", "ocr_conf": 0.25},
        ]
    }

    state = _hand_state_from_enriched(payload)

    # Card values should come from relaxed card selection, not hard fallback.
    assert state.hero_cards == ["Tc", "7h"]
    assert state.big_blind == 100
    assert state.small_blind == 50
    assert state.hero_stack == 3000

    decision = decide_next_action(state)
    assert decision.action in {"watching", "call", "fold", "check", "raise"}
    assert isinstance(decision.reason, str) and len(decision.reason) > 0
