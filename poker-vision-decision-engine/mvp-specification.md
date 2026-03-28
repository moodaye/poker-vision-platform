# MVP: Poker Decision Engine (v1)

## Goal

Given a structured game state, return a **recommended action for Hero**:

* `bet`, `call`, `fold`, `check`, `raise`, `wait`, `watch`

---

## Scope (strict)

* Game: Texas Hold’em
* Players: 3 (fixed)
* Mode: Tournament
* Streets: **Preflop only**
* Strategy: **Rule-based (no GTO / no ML)**

---

## Inputs

### `HandState` (normalized from vision layer)

Must include:

* hero cards (e.g., `"Ah", "Kd"`)
* position (BTN / SB / BB)
* BB / SB amounts
* stack sizes (in chips + derived BB)
* pot size
* amount to call
* action history (preflop only)
* whose turn it is
* hero folded flag

---

## Outputs

### `Decision`

```python
{
  action: "bet" | "call" | "fold" | "check" | "raise" | "wait" | "watch",
  amount: float | None,  # bet/raise: size in chips; call: mirrors amount_to_call; others: None
  reason: str
}
```

---

## Core Logic

### Step 1: Control flow

* if hero folded → `watch`
* if not hero turn → `wait`
* else → evaluate preflop decision

---

### Step 2: Classify situation

* Unopened pot
* Facing limp
* Facing raise
* Facing all-in

---

### Step 3: Derive features

* position (BTN / SB / BB)
* effective stack (in BB)
* hand category:

  * premium (AA–QQ, AK)
  * strong (JJ–TT, AQ, AJ)
  * medium (pairs < TT, suited broadway)
  * speculative (connectors, suited aces)
  * weak

---

### Step 4: Apply rules (examples)

#### Unopened pot

* BTN → open wide → `bet`
* SB → moderate → `bet` or `fold`
* BB → usually `check`

#### Facing raise

* strong → `bet` (3-bet)
* medium → `call`
* weak → `fold`

#### Short stack (<10–12 BB)

* push/fold only:

  * strong → `bet` (all-in)
  * weak → `fold`

---

## Bet sizing (simple)

* open raise: 2–2.5 BB
* 3-bet: ~3x raise
* shove: all-in (if short stack)

---

## Module structure

```text
api.py               # Flask HTTP service (port 5002)
decision_engine/
  models.py          # HandState, Decision
  controller.py      # decide_next_action()
  preflop.py         # rules
  hand_eval.py       # hand classification
  utils.py           # helpers
tests/
  test_controller.py   # control flow (wait, watch)
  test_hand_eval.py    # hand category classification
  test_preflop.py      # rule-based decision scenarios
```

### Module descriptions

In Python these files are called **modules** — a module is a `.py` file that can contain classes, functions, constants, or a mix. Rough Java equivalent: a single `.java` file, except Python modules often contain multiple classes and standalone functions.

**`models.py`** — Data classes (the "shape" of the data)

Defines the input and output structures. Nothing here makes decisions — it just describes what data looks like:
- `HandState` — everything the engine needs to know about the current game situation (cards, position, stacks, pot, etc.)
- `Decision` — the engine's answer (`action`, `amount`, `reason`)
- `ActionEntry` — a single item in the action history (who did what for how much)

**`hand_eval.py`** — Hand classification

Takes two card strings (e.g. `["Ah", "Kd"]`) and returns a `HandCategory` enum value: `PREMIUM`, `STRONG`, `MEDIUM`, `SPECULATIVE`, or `WEAK`. Pure logic, no game state needed.

**`utils.py`** — Arithmetic helpers

Small calculations reused across modules:
- `effective_stack_bb()` — converts chip count to big blinds
- `is_short_stack()` — checks whether hero is in push/fold territory

**`preflop.py`** — The strategy rules

Where the poker logic lives. Takes a `HandState`, classifies the situation (unopened / facing limp / facing raise / facing all-in), then applies the rules to return a `Decision`. Uses `hand_eval` and `utils`.

**`controller.py`** — Entry point / control flow

A thin front door. Handles the two control-flow states first (`hero_folded → watch`, `not hero's turn → wait`), then delegates to `preflop.py` for everything else. The only module callers outside the package need to know about.

**`api.py`** — Flask HTTP service

Wraps `controller.decide_next_action()` as a REST API so other pipeline components can call the decision engine over HTTP without importing Python code directly.

- Run: `uv run python api.py` from the `poker-vision-decision-engine/` directory
- Port: **5002**

Endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns `{"status": "ok"}` — used to confirm the service is running |
| `POST` | `/decide` | Accepts a `HandState` as JSON, returns a `Decision` as JSON |

`POST /decide` request body:

```json
{
  "hero_cards":     ["Ah", "Kd"],
  "position":       "BTN",
  "big_blind":      100,
  "small_blind":    50,
  "hero_stack":     3000,
  "pot":            150,
  "amount_to_call": 0,
  "action_history": [],
  "is_hero_turn":   true,
  "hero_folded":    false
}
```

`POST /decide` response:

```json
{
  "action": "bet",
  "amount": 250.0,
  "reason": "BTN open raise 250 chips with premium hand"
}
```

`is_hero_turn` and `hero_folded` are optional (default `true` / `false`). All other fields are required. Card notation: rank uppercase, suit lowercase — e.g. `"Ah"` = Ace of hearts.

`action_history` entries each require `player` and `action`; `amount` is optional:

```json
{ "player": "BTN", "action": "raise", "amount": 300 }
```

### Call chain

```
POST /decide
  → api.py parses + validates JSON → HandState
  → controller.decide_next_action(HandState)
      → preflop.decide_preflop(HandState)
          → hand_eval.classify_hand(cards)
          → utils.is_short_stack(stack, bb)
      → Decision
  → api.py serialises Decision → JSON response
```

---

## Non-goals (explicitly excluded)

* No postflop logic
* No opponent modeling
* No GTO solver
* No ML / training
* No bluff frequency mixing

---

## Success criteria

* Given mocked inputs → returns correct action + reason
* Handles all control states: `wait`, `watch`
* Produces sensible preflop decisions for:

  * BTN open
  * SB vs BB
  * facing raise
  * short stack shove spots

---

## Tests

Framework: **pytest** (consistent with other modules). Run via `uv run pytest`.

### `test_controller.py` — control flow

| Scenario | Expected action |
|---|---|
| hero folded | `watch` |
| not hero's turn | `wait` |
| hero's turn, valid state | delegates to preflop rules |

### `test_hand_eval.py` — hand classification

| Hand | Expected category |
|---|---|
| `AA`, `KK`, `QQ`, `AKo` | premium |
| `JJ`, `TT`, `AQo`, `AJs` | strong |
| `99`–`22`, suited broadways | medium |
| connectors, suited aces | speculative |
| `72o` | weak |

### `test_preflop.py` — rule scenarios

| Scenario | Inputs | Expected action |
|---|---|---|
| BTN open, premium hand, deep stack | pos=BTN, no prior action, AA | `bet` |
| BB faces no raise | pos=BB, no prior action | `check` |
| Facing raise, strong hand | pos=SB, raise in front, JJ | `raise` |
| Facing raise, weak hand | pos=BTN, raise in front, 72o | `fold` |
| Short stack (<10 BB), premium | stack=8BB, AA | `bet` (shove) |
| Short stack (<10 BB), weak | stack=8BB, 72o | `fold` |

---

## Next milestone (after MVP)

* Add **postflop (flop only)**
* Introduce **check action logic**
* Add **basic hand strength evaluation**