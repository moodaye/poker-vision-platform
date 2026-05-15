from __future__ import annotations

from decision_engine.models import Decision, HandState
from decision_engine.preflop import decide_preflop


def decide_next_action(state: HandState) -> Decision:
    """Entry point: given a HandState, return the recommended Decision."""
    hero_seat = state.hero_seat or state.position

    if state.hero_folded:
        return Decision(
            action="watch",
            amount=None,
            reason="Hero has already folded",
        )

    if len(state.hero_cards) == 0 or state.hero_cards_visibility == "not_exposed":
        return Decision(
            action="watch",
            amount=None,
            reason="Hero cards are not exposed",
        )

    if len(state.hero_cards) != 2:
        return Decision(
            action="watch",
            amount=None,
            reason="Hero cards are incomplete",
        )

    if state.action_on in {"BTN", "SB", "BB"} and state.action_on != hero_seat:
        return Decision(
            action="wait",
            amount=None,
            reason=f"Action is on {state.action_on}, waiting for hero seat {hero_seat}",
        )

    if not state.is_hero_turn:
        return Decision(
            action="wait",
            amount=None,
            reason="Waiting for hero's turn",
        )

    return decide_preflop(state)
