from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# --- Type aliases ---

Position = Literal["BTN", "SB", "BB"]
ActionOn = Literal["BTN", "SB", "BB", "unknown", "none"]
CardVisibility = Literal["exposed", "partial", "not_exposed"]
SeatStatus = Literal[
    "deciding",
    "waiting_turn",
    "folded_this_hand",
    "watching_hand",
    "all_in",
    "eliminated_tournament",
    "unknown",
]
Action = Literal["watching", "check", "call", "raise", "fold"]
PlayerAction = Literal["fold", "call", "raise", "bet", "check", "all_in"]


@dataclass
class ActionEntry:
    """A single action taken by a player in the preflop betting round."""

    player: str  # e.g. "BTN", "SB", "BB"
    action: PlayerAction
    amount: int | None = None


@dataclass
class SeatState:
    """Snapshot of one seat at a 3-player table."""

    seat: Position
    is_hero: bool
    status: SeatStatus = "unknown"
    stack: int | None = None
    is_folded: bool | None = None
    is_all_in: bool | None = None
    has_cards: bool | None = None
    player_name: str | None = None


@dataclass
class TournamentStatus:
    """Tournament clock and blind-level context."""

    current_blind_level: int | None = None
    small_blind_amount: int = 50
    big_blind_amount: int = 100
    ante_amount: int = 0
    seconds_until_next_level: int | None = None


@dataclass
class HandState:
    """Normalised game state passed to the decision engine by the vision layer."""

    hero_cards: list[str]  # e.g. ["Ah", "Kd"]
    big_blind: int  # BB chip value
    small_blind: int  # SB chip value
    hero_stack: int  # hero's chips remaining (includes current bet)
    pot: int  # chips already in the pot
    amount_to_call: int  # chips hero must put in to continue (0 = free check)
    schema_version: str = "2.0.0"
    hero_cards_visibility: CardVisibility = "exposed"
    position: Position = "BTN"  # legacy alias for hero_seat
    hero_seat: Position = "BTN"
    action_on: ActionOn = "unknown"
    seats: list[SeatState] = field(default_factory=list)
    tournament_status: TournamentStatus = field(default_factory=TournamentStatus)
    action_history: list[ActionEntry] = field(default_factory=list)
    is_hero_turn: bool = True
    hero_folded: bool = False

    def __post_init__(self) -> None:
        # Backward compatibility: older callers set only `position`.
        if self.position in {"BTN", "SB", "BB"} and self.hero_seat == "BTN":
            self.hero_seat = self.position

        # Keep legacy `position` aligned with canonical seat.
        self.position = self.hero_seat

        # Ensure tournament blind amounts match hand-level blind fields.
        self.tournament_status.small_blind_amount = self.small_blind
        self.tournament_status.big_blind_amount = self.big_blind


@dataclass
class Decision:
    """Recommended action returned by the decision engine."""

    action: Action
    # raise     → sizing in chips
    # call      → mirrors amount_to_call from HandState
    # all others → None
    amount: float | None
    reason: str
