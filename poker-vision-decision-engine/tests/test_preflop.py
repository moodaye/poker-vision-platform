from typing import Any

from decision_engine.models import ActionEntry, HandState
from decision_engine.preflop import decide_preflop


def _state(**kwargs: Any) -> HandState:
    defaults = dict(
        hero_cards=["Ah", "Ad"],  # AA by default
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
    defaults.update(kwargs)
    return HandState(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Unopened pot
# ---------------------------------------------------------------------------


def test_btn_open_premium_bets() -> None:
    state = _state(hero_cards=["Ah", "Ad"], position="BTN", amount_to_call=0)
    decision = decide_preflop(state)
    assert decision.action == "bet"
    assert decision.amount == 250.0  # 2.5 × 100 BB


def test_btn_open_weak_folds() -> None:
    state = _state(hero_cards=["7h", "2d"], position="BTN", amount_to_call=0)
    decision = decide_preflop(state)
    assert decision.action == "fold"
    assert decision.amount is None


def test_bb_no_raise_checks() -> None:
    state = _state(position="BB", amount_to_call=0, hero_cards=["7h", "2d"])
    decision = decide_preflop(state)
    assert decision.action == "check"
    assert decision.amount is None


def test_sb_open_premium_bets() -> None:
    state = _state(hero_cards=["Kh", "Kd"], position="SB", amount_to_call=0)
    decision = decide_preflop(state)
    assert decision.action == "bet"
    assert decision.amount == 250.0


def test_sb_open_weak_folds() -> None:
    state = _state(hero_cards=["7h", "2d"], position="SB", amount_to_call=0)
    decision = decide_preflop(state)
    assert decision.action == "fold"


# ---------------------------------------------------------------------------
# Facing raise
# ---------------------------------------------------------------------------


def test_facing_raise_strong_three_bets() -> None:
    state = _state(
        hero_cards=["Jh", "Jd"],  # JJ = strong
        position="SB",
        amount_to_call=300,
        action_history=[ActionEntry(player="BTN", action="raise", amount=300)],
    )
    decision = decide_preflop(state)
    assert decision.action == "raise"
    assert decision.amount == 900.0  # 3× raise


def test_facing_raise_premium_three_bets() -> None:
    state = _state(
        hero_cards=["Ah", "Ad"],  # AA = premium
        position="BB",
        amount_to_call=300,
        action_history=[ActionEntry(player="BTN", action="raise", amount=300)],
    )
    decision = decide_preflop(state)
    assert decision.action == "raise"
    assert decision.amount == 900.0


def test_facing_raise_medium_calls() -> None:
    state = _state(
        hero_cards=["9h", "9d"],  # 99 = medium
        position="BB",
        amount_to_call=300,
        action_history=[ActionEntry(player="BTN", action="raise", amount=300)],
    )
    decision = decide_preflop(state)
    assert decision.action == "call"
    assert decision.amount == 300.0


def test_facing_raise_weak_folds() -> None:
    state = _state(
        hero_cards=["7h", "2d"],  # 72o = weak
        position="BTN",
        amount_to_call=300,
        action_history=[ActionEntry(player="SB", action="raise", amount=300)],
    )
    decision = decide_preflop(state)
    assert decision.action == "fold"
    assert decision.amount is None


# ---------------------------------------------------------------------------
# Short stack (push/fold)
# ---------------------------------------------------------------------------


def test_short_stack_premium_shoves() -> None:
    state = _state(
        hero_cards=["Ah", "Ad"],  # AA = premium
        hero_stack=800,  # 8 BB
        big_blind=100,
        amount_to_call=0,
    )
    decision = decide_preflop(state)
    assert decision.action == "bet"
    assert decision.amount == 800.0  # all-in


def test_short_stack_strong_shoves() -> None:
    state = _state(
        hero_cards=["Jh", "Jd"],  # JJ = strong
        hero_stack=900,  # 9 BB
        big_blind=100,
        amount_to_call=0,
    )
    decision = decide_preflop(state)
    assert decision.action == "bet"
    assert decision.amount == 900.0


def test_short_stack_weak_folds() -> None:
    state = _state(
        hero_cards=["7h", "2d"],  # 72o = weak
        hero_stack=800,
        big_blind=100,
        amount_to_call=0,
    )
    decision = decide_preflop(state)
    assert decision.action == "fold"
    assert decision.amount is None


# ---------------------------------------------------------------------------
# Facing all-in
# ---------------------------------------------------------------------------


def test_facing_all_in_premium_calls() -> None:
    state = _state(
        hero_cards=["Ah", "Ad"],
        position="BB",
        amount_to_call=2000,
        action_history=[ActionEntry(player="BTN", action="all_in", amount=2000)],
    )
    decision = decide_preflop(state)
    assert decision.action == "call"


def test_facing_all_in_weak_folds() -> None:
    state = _state(
        hero_cards=["7h", "2d"],
        position="BB",
        amount_to_call=2000,
        action_history=[ActionEntry(player="BTN", action="all_in", amount=2000)],
    )
    decision = decide_preflop(state)
    assert decision.action == "fold"
