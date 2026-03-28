from typing import Any

from decision_engine.controller import decide_next_action
from decision_engine.models import HandState


def _base_state(**overrides: Any) -> HandState:
    defaults = dict(
        hero_cards=["Ah", "Kd"],
        position="BTN",
        big_blind=100,
        small_blind=50,
        hero_stack=3000,
        pot=150,
        amount_to_call=0,
        action_history=[],
        is_hero_turn=True,
        hero_folded=False,
    )
    defaults.update(overrides)
    return HandState(**defaults)  # type: ignore[arg-type]


def test_hero_folded_returns_watch() -> None:
    state = _base_state(hero_folded=True)
    decision = decide_next_action(state)
    assert decision.action == "watch"
    assert decision.amount is None


def test_not_hero_turn_returns_wait() -> None:
    state = _base_state(is_hero_turn=False)
    decision = decide_next_action(state)
    assert decision.action == "wait"
    assert decision.amount is None


def test_hero_turn_delegates_to_preflop() -> None:
    # BTN facing unopened pot with AK (premium) → bet
    state = _base_state()
    decision = decide_next_action(state)
    assert decision.action == "bet"
    assert decision.amount is not None
    assert decision.reason != ""
