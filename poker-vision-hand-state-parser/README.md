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
  "schema_version": "2.1.0",
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
    {"seat": "BTN", "is_hero": true, "status": "deciding", "stack": 3200, "is_folded": false, "is_all_in": null, "has_cards": true},
    {"seat": "SB", "is_hero": false, "status": "unknown", "stack": null, "is_folded": null, "is_all_in": null, "has_cards": null},
    {"seat": "BB", "is_hero": false, "status": "unknown", "stack": null, "is_folded": null, "is_all_in": null, "has_cards": null}
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
- Takes the highest-confidence accepted candidate
- **Fallback:** `3000`

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
- Uses `turn_active` candidates and `turn_halo_score` from enriched detections
- Picks the strongest active candidate and maps it to a seat (`BTN`/`SB`/`BB`) using `spatial_info` or nearest seated `player_name`
- Sets `action_on` to that seat and `is_hero_turn` to whether it matches `hero_seat`
- **Fallback:** `is_hero_turn = False`; `action_on = "none"` when no active halo is detected, or `"unknown"` when a halo exists but seat mapping/confidence is insufficient

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

## Testing

```
uv run pytest poker-vision-hand-state-parser/tests/ -v
```

---

## Notes

- HTTP API is available at `POST /parse` via `api.py` (port 5003 in local service mode).
- Library API is available via direct import of `build_hand_state()`.
- No external dependencies — stdlib only.

---

## Known Limitations

### Opponent state is not modelled

`HandState` currently contains only hero fields. There is no representation of the other players at the table. This means the parser does not extract — and the decision engine cannot reason about — the following:

| Missing field | Why it matters |
|---|---|
| Opponent fold status | Knowing whether an opponent has folded changes the situation from 3-way to heads-up, which significantly widens correct opening/3-betting ranges |
| Opponent stack sizes | Relevant for push/fold sizing decisions and commitment thresholds |
| Opponent seat / position | Needed to correctly attribute action history entries (who raised, who limped) |

**Impact on current behaviour:** The preflop engine always assumes it is playing 3-handed. If one opponent has folded, it will use tighter ranges than optimal for the resulting heads-up situation.

**Planned fix:** Add an `opponents` list to `HandState`, each entry containing `position`, `stack`, and `folded`. The parser would populate this from `player_other` detections and the `action_history`. The decision engine would then select the correct range table based on active player count.

### Player names are intentionally omitted

The engine reasons about seats (BTN / SB / BB), not player identities. Extracting names via OCR would add noise without improving decisions.
