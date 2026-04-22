# poker-vision-hand-state-parser

Converts enriched object-detector output into a structured `HandState` dict for consumption by the decision engine.

---

## Purpose

The hand state parser is the bridge between the vision layer and the decision layer. It takes the enriched detections produced by the Detection Enricher and interprets them into the exact fields the Decision Engine needs to make a preflop decision.

It is a **pure Python library** — no HTTP server, no external dependencies. The orchestrator calls it as a direct function call.

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
  "hero_cards":     ["Ah", "Kd"],
  "position":       "BTN",
  "small_blind":    50,
  "big_blind":      100,
  "hero_stack":     3200,
  "pot":            450,
  "amount_to_call": 200,
  "action_history": [],
  "is_hero_turn":   true,
  "hero_folded":    false
}
```

---

## Field-by-field extraction logic

### `hero_cards`
- Scans objects with `class_name` in `["holecard", "card"]` (holecards preferred)
- Accepts a card if `classification` matches `[2-9TJQKA][cdhs]`, `confidence >= 0.60`, and `classification_conf >= 0.70`
- Takes the first two accepted candidates
- **Fallback:** `["Ah", "Kd"]`

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
- Looks for any of `{check_button, check_fold_button, fold_button, raise_button, bet_pot_button}` with `confidence >= 0.55`
- **Fallback:** `True` (assume it's our turn)

### `hero_folded`
- Scans `action_history` for `action == "fold"` attributed to the hero's position with `confidence >= 0.70`
- **Fallback:** `False`

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
| Rejected | `< 0.55` | Fallback used |

Card-specific gates applied before the band check:
- `detection_conf >= 0.60` (`_MIN_DETECTION_FOR_CARDS`)
- `classification_conf >= 0.70` (`_MIN_CLASSIFICATION_FOR_CARDS`)

---

## Fallback philosophy

The parser **never raises** on missing or low-confidence data. Every field has a safe default so the decision engine always receives a valid `HandState` and can operate deterministically, even when the vision layer is imperfect.

The only exception is a hard contract violation — if the input dict has no `"objects"` key at all, a `ValueError` is raised.

---

## API

```python
from hand_state_parser import build_hand_state, build_hand_state_with_diagnostics

# Simple — returns HandState dict only
hand_state = build_hand_state(enriched_payload)

# With diagnostics — returns (HandState, diagnostics) tuple
# diagnostics maps each field name to its source, field_conf, band, and whether a fallback was used
hand_state, diagnostics = build_hand_state_with_diagnostics(enriched_payload)
```

---

## Testing

```
uv run pytest poker-vision-hand-state-parser/tests/ -v
```

---

## Notes

- No HTTP API. The orchestrator imports and calls `build_hand_state()` directly.
- No external dependencies — stdlib only.
- A future phase 2 evolution could wrap this as an HTTP service if dynamic agent orchestration requires it to be independently callable.
