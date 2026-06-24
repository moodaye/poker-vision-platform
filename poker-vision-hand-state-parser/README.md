# poker-vision-hand-state-parser

Converts enriched object-detector output into a structured `HandState` dict for consumption by the decision engine.

---

## Purpose

The hand state parser is the bridge between the vision layer and the decision layer. It takes the enriched detections produced by the Detection Enricher and interprets them into the exact fields the Decision Engine needs to make a preflop decision.

It is exposed both as a Python library (`build_hand_state`) and as an HTTP microservice via `api.py`.

---

## Input

A dict produced by the Detection Enricher with an `"objects"` list:

```json
{
  "objects": [
    { "class_name": "holecard",      "classification": "Ah", "classification_conf": 0.93, "confidence": 0.91 },
    { "class_name": "holecard",      "classification": "Kd", "classification_conf": 0.90, "confidence": 0.88 },
    { "class_name": "blinds",        "ocr_text": "50/100",   "ocr_conf": 0.85,            "confidence": 0.90 },
    { "class_name": "chip_stack",    "ocr_text": "3200",     "ocr_conf": 0.80,            "confidence": 0.87 },
    { "class_name": "total_pot",     "ocr_text": "450",      "ocr_conf": 0.78,            "confidence": 0.88 },
    { "class_name": "bet",           "ocr_text": "200",      "ocr_conf": 0.82,            "confidence": 0.85 },
    { "class_name": "dealer_button", "spatial_info": {"hero_position": "BTN"}, "spatial_conf": 0.80, "confidence": 0.95 },
    { "class_name": "fold_button",   "confidence": 0.92 }
  ]
}
```

Each object's enriched fields depend on how the Detection Enricher routed it:
- **classify** path → `classification`, `classification_conf`
- **ocr** path → `ocr_text`, `ocr_conf`
- **spatial** path → `spatial_info`, `spatial_conf`

---

## Output

A `HandState` dict:

```json
{
  "schema_version": "2.2.0",
  "hero_cards":     ["Ah", "Kd"],
  "hero_cards_visibility": "exposed",
  "position":       "BTN",
  "hero_seat":      "BTN",
  "action_on":      "BTN",
  "small_blind":    50,
  "big_blind":      100,
  "hero_stack":     3200,
  "pot":            450,
  "amount_to_call": 200,
  "seats": [
    {"seat": "BTN", "is_hero": true, "player_name": "moodaye", "status": "deciding", "stack": 3200, "is_folded": false, "is_all_in": null, "has_cards": true},
    {"seat": "SB", "is_hero": false, "player_name": "Weave", "status": "waiting_turn", "stack": 2800, "is_folded": null, "is_all_in": null, "has_cards": null},
    {"seat": "BB", "is_hero": false, "player_name": "Donna1212", "status": "waiting_turn", "stack": 3100, "is_folded": null, "is_all_in": null, "has_cards": null}
  ],
  "tournament_status": {
    "current_blind_level": null,
    "small_blind_amount": 50,
    "big_blind_amount": 100,
    "ante_amount": 0,
    "seconds_until_next_level": null
  },
  "action_history": [],
  "is_hero_turn":   true,
  "hero_folded":    false
}
```

`position` is retained as a legacy alias for `hero_seat`.

---

## Field-by-field extraction logic

### `hero_cards`
- Scans objects with `class_name` in `["holecard", "card"]` (holecards preferred)
- Accepts a card if `classification` matches `[2-9TJQKA][cdhs]`, `confidence >= 0.60`, and `classification_conf >= 0.70`
- Orders accepted hero cards deterministically left-to-right using x position from detector/enricher bboxes
- If fewer than two visible hero card candidates exist, emits `hero_cards: []`

### `hero_cards_visibility`
- `"exposed"` when two hero cards are present in `hero_cards`
- `"not_exposed"` when hero cards are not visible (for example pre-deal or occluded)
- This field replaces reporting of synthetic hero-card fallback values

### `position`
- Looks for `player_me` + `dealer_button` objects with `spatial_info` containing `seat`, `position`, or `hero_position` with value `BTN`, `SB`, or `BB`
- Prefers `player_me + dealer_button` combined; falls back to either alone
- **Fallback:** `"BTN"`

### `small_blind` / `big_blind`
- Finds `class_name == "blinds"` object, parses two integers from `ocr_text` (e.g. `"50/100"` or `"50 100"`)
- Requires `sb > 0` and `bb > sb`
- **Fallback:** `50 / 100`

### `hero_stack`
- Collects all `chip_stack` objects with a parseable positive integer in `ocr_text`
- Also accepts `ocr_text == "All In"` (case-insensitive, allows `"All-In"` / `"ALL IN"` etc.) — treated as `stack = 0` with `is_all_in = True`
- Takes the highest-confidence accepted candidate owned by the hero (matched by `spatial_info.owner_player`)
- **Fallback:** `3000`

### `seats`
- Always emits exactly three seat entries (`BTN`, `SB`, `BB`)
- Populates `player_name` from Stage 2 seat-enriched `player_name` objects
- Hero seat stack uses `hero_stack`
- Opponent seat stacks are populated by mapping `chip_stack.spatial_info.owner_player` to seat via `player_name.spatial_info.seat`
- `is_all_in`: set to `true` for a seat when its `chip_stack` carries `ocr_text == "All In"`; `null` when no all-in signal is detected
- Status mapping:
  - Hero: `deciding` / `waiting_turn` / `watching_hand` / `folded_this_hand` / `all_in` / `eliminated_tournament` based on hero signals
  - Opponents: `deciding` when `action_on == seat`, otherwise `waiting_turn`

### `pot`
- Tries `total_pot`, then `pot`, then `pot_bet` — first accepted wins
- Parses first integer from `ocr_text` (handles prefixes like `"Total Pot: 300"`)
- **Fallback:** `150`

### `amount_to_call`
- Tries `bet`, then `max_bet`, then `min_bet` — first accepted wins
- **Fallback:** `0` (no call needed / check)

### `action_history`
- Scans all objects for those with both `action` and `player` string fields
- Currently always empty — object detector does not yet emit action objects
- **Fallback:** `[]`

### `is_hero_turn`
- **MVP0 (primary):** Detects `bet_box` objects in the enriched detections. If any `bet_box` is present, `is_hero_turn = True` with `confidence = 1.0` and `source = "bet_box_detection"`. The bet-box widget is only rendered by the poker client when it is the hero's turn to act, making it a reliable binary signal.
- **Halo fallback (retained for future):** If no `bet_box` is detected, falls back to `turn_active` + `turn_halo_score` candidates from enriched detections. Picks the strongest active candidate, maps it to a seat (`BTN`/`SB`/`BB`) via `spatial_info` or nearest seated `player_name`, then sets `action_on` to that seat and `is_hero_turn` to whether it matches `hero_seat`.
- **Fallback:** `is_hero_turn = False`; `action_on = "none"` when neither signal is present, or `"unknown"` when a halo exists but seat mapping/confidence is insufficient

### `hero_folded`
- Scans `action_history` for `action == "fold"` attributed to the hero's position with `confidence >= 0.70`
- Also infers folded when `hero_cards_visibility == "not_exposed"` and `pot` exceeds forced preflop contributions (`small_blind + big_blind + 3 * ante_amount`)
- **Fallback:** `False` when neither signal is present

---

## Confidence gating

Every field uses:

```
field_conf = min(detection_conf, extraction_conf)
```

| Band | Range | Behaviour |
|---|---|---|
| Trusted | `>= 0.80` | Accepted, no warning |
| Usable | `0.55 – 0.80` | Accepted with warning in diagnostics |
| Rejected | `< 0.55` | Field default used (or `hero_cards_visibility: "not_exposed"` for hero cards) |

Card-specific gates applied before the band check:
- `detection_conf >= 0.60` (`_MIN_DETECTION_FOR_CARDS`)
- `classification_conf >= 0.70` (`_MIN_CLASSIFICATION_FOR_CARDS`)

---

## Defaulting philosophy

The parser **never raises** on missing or low-confidence data. Numeric and position fields have safe defaults so downstream logic remains deterministic. Hero cards are no longer synthesized; if they are not visible, `hero_cards` is empty and `hero_cards_visibility` is set to `"not_exposed"`.

The only exception is a hard contract violation — if the input dict has no `"objects"` key at all, a `ValueError` is raised.

---

## API

```python
from hand_state_parser import build_hand_state, build_hand_state_with_diagnostics

# Simple — returns HandState dict only
hand_state = build_hand_state(enriched_payload)

# With diagnostics — returns (HandState, diagnostics) tuple
# diagnostics maps each field name to its source, field_conf, band, and whether a field default was used
hand_state, diagnostics = build_hand_state_with_diagnostics(enriched_payload)
```

---

## Diagnostics logging configuration

The HTTP service in `api.py` supports optional diagnostics logging controlled by `config.yaml`:
- `log_diagnostics` (default: `false`)
The HTTP service in `api.py` supports optional diagnostics logging controlled by an environment variable:

- `HAND_STATE_PARSER_LOG_DIAGNOSTICS` (default: `false`)
  - Truthy values: `1`, `true`, `yes`, `on` (case-insensitive)
  - Any other value disables diagnostics logging

When enabled:

- `/parse` uses `build_hand_state_with_diagnostics(...)`
- Service logs include:
  - `Parsed hand state: { ... }`
  - `Hand state diagnostics: { ... }`
- Logs are emitted as JSON strings for easier filtering/searching

When disabled:

- `/parse` uses `build_hand_state(...)`
- No additional diagnostics logs are emitted

Response contract is unchanged in both modes: `/parse` still returns only the HandState payload.

---

## Testing

```
uv run pytest poker-vision-hand-state-parser/tests/ -v
```

---

## Notes

- HTTP API is available at `POST /parse` via `api.py` (port 5003 in local service mode).
- Library API is available via direct import of `build_hand_state()`.
- Runtime dependencies include Flask and PyYAML.

---

## Known Limitations

- Opponent `is_folded`, `is_all_in`, and `has_cards` remain unresolved (`null`) because the current detector/enricher does not emit reliable per-opponent signals for these states.
- Opponent `stack` values depend on successful Stage 2 owner matching (`chip_stack -> owner_player`) and seat assignment (`player_name -> seat`). If either link is missing/noisy, stack remains `null` for that seat.
