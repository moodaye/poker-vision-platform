from __future__ import annotations

_SHORT_STACK_THRESHOLD_BB: float = 12.0


def effective_stack_bb(stack: int, big_blind: int) -> float:
    """Return hero's stack expressed in big blinds."""
    if big_blind <= 0:
        raise ValueError("big_blind must be a positive integer")
    return stack / big_blind


def is_short_stack(
    stack: int,
    big_blind: int,
    threshold_bb: float = _SHORT_STACK_THRESHOLD_BB,
) -> bool:
    """Return True if hero is in push/fold territory (< threshold big blinds)."""
    return effective_stack_bb(stack, big_blind) < threshold_bb
