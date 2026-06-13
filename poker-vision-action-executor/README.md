# poker-vision-action-executor

Flask microservice that enacts the decision engine's output against a live poker client window.

The executor uses the **Windows UI API** (via ctypes) to locate the poker client window by title, find the correct action button by its label, optionally type a bet amount into the bet-size input control, then click the button using pyautogui screen coordinates.

Port: **5005**

---

## Endpoints

### `POST /execute`

Execute a poker action in the foreground poker client window.

**Request body (JSON)**

| Field | Type | Required | Description |
|---|---|---|---|
| `action` | string | ✔ | `"fold"`, `"call"`, `"check"`, `"raise"`, `"bet"`, or `"watching"` |
| `amount` | integer | For raise/bet | Chip amount to enter in the bet-size box |
| `dry_run` | boolean | No (default `false`) | Locate button but don't click — for testing |
| `window_title_hint` | string | No | Override configured window title hints with a single hint |

**Response body (JSON)**

| Field | Type | Description |
|---|---|---|
| `success` | boolean | Whether the action was executed |
| `action` | string | Normalised action string |
| `amount` | integer \| null | Amount used (for raise/bet) |
| `method` | string | `"windows_api"`, `"dry_run"`, or `"none"` |
| `message` | string | Human-readable status description |

**HTTP status codes**

| Code | Meaning |
|---|---|
| 200 | Action executed successfully |
| 400 | Malformed request (missing `action`, wrong types) |
| 422 | Action could not be executed (window or button not found) |
| 500 | Unexpected server error |

#### Example requests

```bash
# Fold
curl -s -X POST http://127.0.0.1:5005/execute \
  -H "Content-Type: application/json" \
  -d '{"action": "fold"}'

# Raise to 300
curl -s -X POST http://127.0.0.1:5005/execute \
  -H "Content-Type: application/json" \
  -d '{"action": "raise", "amount": 300}'

# Dry run — detect without clicking
curl -s -X POST http://127.0.0.1:5005/execute \
  -H "Content-Type: application/json" \
  -d '{"action": "call", "dry_run": true}'
```

---

### `GET /health`

Returns `{"status": "ok"}` when the service is running.

---

## Configuration — `config.yaml`

```yaml
port: 5005

# Substrings to match against top-level window titles (case-insensitive).
# "Top-level" means the main outer window of the poker client, not a child
# control or dialog. The executor uses EnumWindows to find a visible top-level
# window whose title contains one of these hints.
#
# Note: the window does not need to be the active foreground window to be
# discovered. If it is visible but behind another window, it will still be
# considered. Minimized windows may also be matched if Windows reports them as
# visible. Hidden or non-visible windows are skipped.
window_title_hints:
  - PokerStars
  - GGPoker
  - 888poker
  - partypoker
  - WSOP
  - PokerTestHarness   # used by the integration test harness

# Button caption variants per action.  Matching is case-insensitive prefix,
# so "Call 75" is matched by the variant "Call".
button_labels:
  fold:  [Fold, "Fold Hand", FOLD]
  call:  [Call, "Call All-In", CALL]
  check: [Check, CHECK]
  raise: ["Raise To", Raise, RAISE]
  bet:   [Bet, BET]

pre_action_delay_ms: 200
post_action_delay_ms: 100
```

---

## Module structure

```
poker-vision-action-executor/
├── api.py              Flask HTTP service
├── config.yaml         Per-client button variants and port config
├── executor.py         Core execution logic
├── bet_entry.py        Locate and populate the bet-size input
├── poker_window.py     Win32 window / child-control discovery
├── models.py           ActionResult dataclass
├── pyproject.toml
├── tests/
│   ├── test_api.py
│   ├── test_executor.py
│   ├── test_bet_entry.py
│   └── test_poker_window.py
└── test_harness/
    ├── harness.py                  Native Win32 test window (Fold/Call/Check/Raise To)
    └── test_harness_integration.py Integration tests (require graphical desktop)
```

---

## Running

```bash
# From the repo root
uv run python poker-vision-action-executor/api.py
```

Or via the service manager:

```bash
uv run python manage_services.py start
```

---

## Tests

```bash
# Unit tests (no real windows needed)
uv run pytest tests/ -v

# Integration tests (requires Windows desktop, spawns a real Win32 window)
uv run pytest test_harness/ -v -m integration
```

---

## How it works

1. The **orchestrator** POSTs the decision engine output to `POST /execute`.
2. The executor looks up button-label variants for the action in `config.yaml`.
3. It calls `EnumWindows` to find a top-level window whose title contains one of the configured hints.
4. It calls `EnumChildWindows` to enumerate all child `Button` controls and finds the one whose caption prefix-matches a configured variant.
5. For `raise` / `bet` actions it also locates the first numeric `Edit` child, clicks it, selects all, and types the amount via pyautogui.
6. It brings the window to the foreground (`SetForegroundWindow`) and clicks the button centre using pyautogui screen coordinates.
