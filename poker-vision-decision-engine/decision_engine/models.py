from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# --- Type aliases ---

Position = Literal["BTN", "SB", "BB"]
Action = Literal["bet", "call", "fold", "check", "raise", "wait", "watch"]
PlayerAction = Literal["fold", "call", "raise", "bet", "check", "all_in"]


@dataclass
class ActionEntry:
    """A single action taken by a player in the preflop betting round."""

    player: str  # e.g. "BTN", "SB", "BB"
    action: PlayerAction
    amount: int | None = None


@dataclass
class HandState:
    """Normalised game state passed to the decision engine by the vision layer."""

    hero_cards: list[str]  # e.g. ["Ah", "Kd"]
    position: Position  # hero's seat at the table
    big_blind: int  # BB chip value
    small_blind: int  # SB chip value
    hero_stack: int  # hero's chips remaining (includes current bet)
    pot: int  # chips already in the pot
    amount_to_call: int  # chips hero must put in to continue (0 = free check)
    action_history: list[ActionEntry] = field(default_factory=list)
    is_hero_turn: bool = True
    hero_folded: bool = False


@dataclass
class Decision:
    """Recommended action returned by the decision engine."""

    action: Action
    # bet/raise → sizing in chips
    # call      → mirrors amount_to_call from HandState
    # all others → None
    amount: float | None
    reason: str
