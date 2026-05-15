from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from decision_engine.controller import decide_next_action
from decision_engine.models import ActionEntry, HandState

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "tests" / "fixtures" / "preflop_scenarios" / "manifest.json"


def _load_manifest() -> list[dict[str, Any]]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        scenarios = json.load(f)
    return [scenario for scenario in scenarios if scenario.get("enabled", True)]


def _load_hand_state_from_fixture(path_str: str) -> HandState:
    fixture_path = ROOT / path_str
    with fixture_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

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
        big_blind=data["big_blind"],
        small_blind=data["small_blind"],
        hero_stack=data["hero_stack"],
        pot=data["pot"],
        amount_to_call=data["amount_to_call"],
        schema_version=str(data.get("schema_version", "2.1.0")),
        hero_cards_visibility=data.get("hero_cards_visibility", "exposed"),
        position=data.get("position", data.get("hero_seat", "BTN")),
        hero_seat=data.get("hero_seat", data.get("position", "BTN")),
        action_on=data.get("action_on", "unknown"),
        action_history=action_history,
        is_hero_turn=bool(data.get("is_hero_turn", True)),
        hero_folded=bool(data.get("hero_folded", False)),
    )


@pytest.mark.parametrize("scenario", _load_manifest(), ids=lambda s: str(s["id"]))
def test_preflop_scenarios_from_manifest(scenario: dict[str, Any]) -> None:
    state = _load_hand_state_from_fixture(scenario["hand_state_fixture"])
    decision = decide_next_action(state)

    assert decision.action == scenario["expected_action"]

    expected_reason_contains = scenario.get("expected_reason_contains")
    if isinstance(expected_reason_contains, str) and expected_reason_contains:
        assert expected_reason_contains.lower() in decision.reason.lower()

    if decision.action in {"watching", "check", "fold"}:
        assert decision.amount is None

    if decision.action in {"call", "raise"}:
        assert decision.amount is not None
        assert decision.amount > 0
