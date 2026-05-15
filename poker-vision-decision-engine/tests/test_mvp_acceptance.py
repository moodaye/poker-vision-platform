from typing import Any

from decision_engine.controller import decide_next_action
from decision_engine.models import ActionEntry, HandState


def _state(**overrides: Any) -> HandState:
    defaults = dict(
        hero_cards=["Ah", "Kd"],
        hero_cards_visibility="exposed",
        position="BTN",
        hero_seat="BTN",
        action_on="BTN",
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


def test_mvp_watching_when_not_hero_turn() -> None:
    decision = decide_next_action(_state(is_hero_turn=False, action_on="SB"))
    assert decision.action == "watching"


def test_mvp_watching_when_cards_not_exposed() -> None:
    decision = decide_next_action(
        _state(hero_cards=[], hero_cards_visibility="not_exposed")
    )
    assert decision.action == "watching"


def test_mvp_watching_when_hero_folded() -> None:
    decision = decide_next_action(_state(hero_folded=True))
    assert decision.action == "watching"


def test_mvp_check_call_raise_fold_actions_possible() -> None:
    check_decision = decide_next_action(
        _state(position="BB", hero_seat="BB", action_on="BB", hero_cards=["7h", "2d"])
    )
    assert check_decision.action == "check"

    call_decision = decide_next_action(
        _state(
            position="BB",
            hero_seat="BB",
            action_on="BB",
            hero_cards=["9h", "9d"],
            amount_to_call=300,
            action_history=[ActionEntry(player="BTN", action="raise", amount=300)],
        )
    )
    assert call_decision.action == "call"

    raise_decision = decide_next_action(
        _state(
            position="BTN",
            hero_seat="BTN",
            action_on="BTN",
            hero_cards=["Ah", "Ad"],
            amount_to_call=0,
        )
    )
    assert raise_decision.action == "raise"

    fold_decision = decide_next_action(
        _state(
            position="BTN",
            hero_seat="BTN",
            action_on="BTN",
            hero_cards=["7h", "2d"],
            amount_to_call=300,
            action_history=[ActionEntry(player="SB", action="raise", amount=300)],
        )
    )
    assert fold_decision.action == "fold"
