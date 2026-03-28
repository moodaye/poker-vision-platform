from __future__ import annotations

from enum import Enum

from decision_engine.hand_eval import HandCategory, classify_hand
from decision_engine.models import Decision, HandState
from decision_engine.utils import effective_stack_bb, is_short_stack


class Situation(str, Enum):
    UNOPENED = "unopened"
    FACING_LIMP = "facing_limp"
    FACING_RAISE = "facing_raise"
    FACING_ALL_IN = "facing_all_in"


# ---------------------------------------------------------------------------
# Situation classifier
# ---------------------------------------------------------------------------


def classify_situation(state: HandState) -> Situation:
    """Determine the preflop situation hero is facing."""
    for entry in state.action_history:
        if entry.action == "all_in":
            return Situation.FACING_ALL_IN

    if state.amount_to_call > state.big_blind:
        return Situation.FACING_RAISE

    if state.amount_to_call == state.big_blind:
        # Someone called the BB without raising = limp
        for entry in state.action_history:
            if entry.action == "call":
                return Situation.FACING_LIMP
        return Situation.UNOPENED

    # amount_to_call == 0 (or < BB, e.g. SB completing): free check / first to act
    return Situation.UNOPENED


# ---------------------------------------------------------------------------
# Bet sizing helpers
# ---------------------------------------------------------------------------


def _open_raise_amount(state: HandState) -> float:
    return state.big_blind * 2.5


def _isolation_raise_amount(state: HandState) -> float:
    return state.big_blind * 3.0


def _three_bet_amount(state: HandState) -> float:
    return state.amount_to_call * 3.0


# ---------------------------------------------------------------------------
# Top-level preflop decision
# ---------------------------------------------------------------------------


def decide_preflop(state: HandState) -> Decision:
    """Apply preflop rules to the given HandState and return a Decision."""
    category = classify_hand(state.hero_cards)
    situation = classify_situation(state)
    stack_bb = effective_stack_bb(state.hero_stack, state.big_blind)
    short = is_short_stack(state.hero_stack, state.big_blind)

    # Short-stack override: push/fold mode
    if short:
        if category in (HandCategory.PREMIUM, HandCategory.STRONG):
            return Decision(
                action="bet",
                amount=float(state.hero_stack),
                reason=f"Short stack ({stack_bb:.1f} BB): shoving {category.value} hand",
            )
        return Decision(
            action="fold",
            amount=None,
            reason=f"Short stack ({stack_bb:.1f} BB): folding {category.value} hand",
        )

    if situation == Situation.UNOPENED:
        return _unopened(state, category)
    elif situation == Situation.FACING_LIMP:
        return _facing_limp(state, category)
    elif situation == Situation.FACING_RAISE:
        return _facing_raise(state, category)
    else:  # FACING_ALL_IN
        return _facing_all_in(state, category)


# ---------------------------------------------------------------------------
# Situation handlers
# ---------------------------------------------------------------------------


def _unopened(state: HandState, category: HandCategory) -> Decision:
    pos = state.position
    size = _open_raise_amount(state)

    if pos == "BB":
        return Decision(
            action="check",
            amount=None,
            reason="BB in unopened pot: check",
        )

    if pos == "BTN":
        if category == HandCategory.WEAK:
            return Decision(
                action="fold",
                amount=None,
                reason="BTN: folding weak hand",
            )
        return Decision(
            action="bet",
            amount=size,
            reason=f"BTN open raise {size:.0f} chips with {category.value} hand",
        )

    # SB plays a tighter opening range
    if category in (HandCategory.PREMIUM, HandCategory.STRONG):
        return Decision(
            action="bet",
            amount=size,
            reason=f"SB open raise {size:.0f} chips with {category.value} hand",
        )
    return Decision(
        action="fold",
        amount=None,
        reason=f"SB: folding {category.value} hand",
    )


def _facing_limp(state: HandState, category: HandCategory) -> Decision:
    pos = state.position
    size = _isolation_raise_amount(state)

    if pos == "BB":
        if category in (HandCategory.PREMIUM, HandCategory.STRONG):
            return Decision(
                action="raise",
                amount=size,
                reason=f"BB isolating limp with {category.value} hand ({size:.0f} chips)",
            )
        # BB gets a free look at the flop
        return Decision(
            action="check",
            amount=None,
            reason=f"BB: checking against limp with {category.value} hand",
        )

    if category in (HandCategory.PREMIUM, HandCategory.STRONG):
        return Decision(
            action="raise",
            amount=size,
            reason=f"{pos} isolating limp with {category.value} hand ({size:.0f} chips)",
        )
    if category == HandCategory.MEDIUM:
        return Decision(
            action="call",
            amount=float(state.amount_to_call),
            reason=f"{pos}: calling limp with {category.value} hand",
        )
    return Decision(
        action="fold",
        amount=None,
        reason=f"{pos}: folding {category.value} hand against limp",
    )


def _facing_raise(state: HandState, category: HandCategory) -> Decision:
    size = _three_bet_amount(state)

    if category in (HandCategory.PREMIUM, HandCategory.STRONG):
        return Decision(
            action="raise",
            amount=size,
            reason=f"3-betting {category.value} hand ({size:.0f} chips)",
        )
    if category == HandCategory.MEDIUM:
        return Decision(
            action="call",
            amount=float(state.amount_to_call),
            reason=f"Calling raise with {category.value} hand",
        )
    return Decision(
        action="fold",
        amount=None,
        reason=f"Folding {category.value} hand against raise",
    )


def _facing_all_in(state: HandState, category: HandCategory) -> Decision:
    if category in (HandCategory.PREMIUM, HandCategory.STRONG):
        return Decision(
            action="call",
            amount=float(state.amount_to_call),
            reason=f"Calling all-in with {category.value} hand",
        )
    return Decision(
        action="fold",
        amount=None,
        reason=f"Folding {category.value} hand against all-in",
    )
