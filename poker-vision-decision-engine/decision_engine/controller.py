from __future__ import annotations

from decision_engine.models import Decision, HandState
from decision_engine.preflop import decide_preflop


def decide_next_action(state: HandState) -> Decision:
    """Entry point: given a HandState, return the recommended Decision."""
    if state.hero_folded:
        return Decision(
            action="watch",
            amount=None,
            reason="Hero has already folded",
        )

    if not state.is_hero_turn:
        return Decision(
            action="wait",
            amount=None,
            reason="Waiting for hero's turn",
        )

    return decide_preflop(state)
