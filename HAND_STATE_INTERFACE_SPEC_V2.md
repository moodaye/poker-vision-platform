# Hand State Interface Specification v2

Status: Draft implementation active
Version: 2.2.0
Scope: Texas Hold'em, tournament, 3 players (BTN/SB/BB)

## 1. Purpose

This document defines the interface between:
- Hand State Parser (producer)
- Decision Engine (consumer)

The goal is deterministic, explicit game-state transfer with backward-compatible support for legacy v1 fields.

## 2. Contract Summary

The parser emits a JSON object with:
- Legacy fields used by current decision logic
- New v2 fields for explicit seat modeling and tournament context

The decision engine accepts both:
- v2 payloads (preferred)
- v1-style payloads (supported)

## 3. Top-Level Fields

Required:
- schema_version: string
- hero_cards: string[]
- big_blind: integer
- small_blind: integer
- hero_stack: integer
- pot: integer
- amount_to_call: integer
- position: "BTN" | "SB" | "BB" (legacy alias)

Required in v2 semantics:
- hero_seat: "BTN" | "SB" | "BB"

Optional:
- hero_cards_visibility: "exposed" | "partial" | "not_exposed"
- action_on: "BTN" | "SB" | "BB" | "unknown" | "none"
- seats: SeatState[]
- tournament_status: TournamentStatus
- action_history: ActionEntry[]
- is_hero_turn: boolean
- hero_folded: boolean

## 4. Type Definitions

### 4.1 ActionEntry

- player: string (seat label expected: BTN/SB/BB)
- action: "fold" | "call" | "raise" | "bet" | "check" | "all_in"
- amount: integer | null

### 4.2 SeatState

- seat: "BTN" | "SB" | "BB"
- is_hero: boolean
- player_name: string | null
- status: "deciding" | "waiting_turn" | "folded_this_hand" | "watching_hand" | "all_in" | "eliminated_tournament" | "unknown"
- stack: integer | null
- is_folded: boolean | null
- is_all_in: boolean | null
- has_cards: boolean | null

Seat status label meanings:
- deciding: this player is active in the hand and is currently expected to act
- waiting_turn: this player is active in the hand but action is on another seat
- folded_this_hand: player folded this hand and is now observing
- watching_hand: seated in tournament but not currently participating in this hand
- all_in: player has no further decisions this hand
- eliminated_tournament: player is out of chips and out of the tournament
- unknown: state cannot be inferred confidently yet

### 4.3 TournamentStatus

- current_blind_level: integer | null
- small_blind_amount: integer
- big_blind_amount: integer
- ante_amount: integer
- seconds_until_next_level: integer | null

### 4.4 Hero Turn Detection

- is_hero_turn: boolean
  - Derived from:
    - Presence of `bet_box` objects in enriched payload (primary signal).
    - Turn-halo signal from the enricher (fallback): the enricher scores each player bbox using a horizontal band brightness comparison that detects the white/silver active-player ring rendered by this poker client. See the Detection Enricher README for full details.
  - Confidence:
    - `1.0` when `bet_box` is detected.
    - Fallback confidence when halo logic is used.

### 4.5 All-In Detection

- `is_all_in` on `SeatState` is populated from `chip_stack` objects whose `ocr_text` matches `"All In"` (case-insensitive; also matches `"All-In"`, `"ALL IN"`, etc.)
- When detected: `stack = 0`, `is_all_in = true`
- When no all-in signal is present: `is_all_in = null`

See the Detection Enricher README for how the all-in badge is recognised at the OCR layer.

## 5. Parser Output Rules

1. hero_seat is set equal to parsed hero position.
2. position is retained as a legacy alias (same value as hero_seat).
3. action_on is derived from turn-halo seat mapping when available; defaults to "none" when no active halo is detected, or "unknown" when halo evidence exists but seat mapping/confidence is insufficient.
4. seats always contains exactly three entries (BTN/SB/BB).
5. tournament_status always includes small_blind_amount and big_blind_amount.
6. ante_amount defaults to 0 when not detected.
7. current_blind_level and seconds_until_next_level default to null when unknown.
8. each seat includes a normalized status label. Hero status is derived from hero fold/card/turn signals. Opponent status is derived from `action_on`: the active seat is `"deciding"`; non-active seats are `"waiting_turn"`.
9. hero_folded is true when either: (a) action_history contains a confident hero fold, or (b) hero cards are not exposed and pot exceeds forced preflop contributions (`small_blind + big_blind + 3 * ante_amount`).
10. seats entries include `player_name` when available from Stage 2 seat/name enrichment; otherwise null.
11. opponent seat `stack` values are populated when `chip_stack.spatial_info.owner_player` can be matched to a seated `player_name`; otherwise null.

## 6. Decision Engine Consumption Rules

1. If hero_folded=true -> action "watching".
2. If hero cards are not exposed or missing -> action "watching".
3. If action_on is a seat and does not equal hero_seat -> action "watching".
4. Else if is_hero_turn=false -> action "watching".
5. Else run preflop strategy logic.

## 7. Validation Constraints

1. hero_cards length must be either:
- 0: decision engine returns watching
- 2: normal decision path
2. position/hero_seat must be one of BTN/SB/BB.
3. action_on must be one of BTN/SB/BB/unknown/none.
4. Numeric chip fields must be integers.
5. If a seat provides status, it must be one of the allowed SeatState status labels.

## 8. Backward Compatibility

1. Legacy position field remains required by existing callers.
2. hero_seat may be omitted by legacy clients; engine falls back to position.
3. Unknown extra fields are ignored.

## 9. Example Payload

```json
{
  "schema_version": "2.2.0",
  "hero_cards": ["Ah", "Kd"],
  "hero_cards_visibility": "exposed",
  "position": "BTN",
  "hero_seat": "BTN",
  "action_on": "BTN",
  "big_blind": 100,
  "small_blind": 50,
  "hero_stack": 3200,
  "pot": 450,
  "amount_to_call": 200,
  "seats": [
    {
      "seat": "BTN",
      "is_hero": true,
      "player_name": "moodaye",
      "status": "deciding",
      "stack": 3200,
      "is_folded": false,
      "is_all_in": false,
      "has_cards": true
    },
    {
      "seat": "SB",
      "is_hero": false,
      "player_name": "Weave",
      "status": "waiting_turn",
      "stack": 2800,
      "is_folded": null,
      "is_all_in": null,
      "has_cards": null
    },
    {
      "seat": "BB",
      "is_hero": false,
      "player_name": "Donna1212",
      "status": "waiting_turn",
      "stack": 3100,
      "is_folded": null,
      "is_all_in": null,
      "has_cards": null
    }
  ],
  "tournament_status": {
    "current_blind_level": 12,
    "small_blind_amount": 50,
    "big_blind_amount": 100,
    "ante_amount": 12,
    "seconds_until_next_level": 95
  },
  "action_history": [
    { "player": "SB", "action": "call", "amount": 100 }
  ],
  "is_hero_turn": true,
  "hero_folded": false
}
```
