# Poker Vision — Project Overview

End-to-end system for an autonomous poker bot using computer vision and rule-based decision logic.

---

## Architecture

The project has two interconnected pipelines.

### Training pipeline (dataset preparation)

Produces the trained card classifier model from raw screenshots:

```
screenshots → object detector → detection JSON → card snipper → snips → card labeller → labels.csv → trainer → model
```


### Bot pipeline (live play)

Runs the trained model and decision logic against a live game:

```
Screen Monitor (5000)
       ↓  screenshot PNG (webhook POST)
Orchestrator (5100)
       ↓  (receives screenshot, runs full pipeline)
    Object Detector (Roboflow)
       ↓  detections
    Detection Enricher (5004)
       ↓  enriched detections: [{class_name, bbox_xyxy, classification|ocr_text|spatial_info, confidence}]
    Hand State Parser
       ↓  HandState JSON
    Decision Engine (5002)
       ↓  Decision (action + amount + reason)
Screen Monitor — logs decision, speaks action via Windows TTS
```

#### Orchestrator Service Contract

- **Input:**
  - POST `/decide` (multipart/form-data)
  - `image`: screenshot (PNG/JPG)
- **Output:**
  - JSON: `{ "action": "call", "amount": 400, "reason": "Standard preflop call with suited connectors" }`
- The orchestrator handles all downstream calls (detection, enrichment, hand-state parsing, decision) and returns the next action.

### Why each component runs as a separate API

Each stage in the bot pipeline is a **standalone Flask or FastAPI service** with its own port. This architecture was chosen deliberately:

- **Independent development and testing.** Any component can be run and tested in isolation with a `curl` or test script, without the rest of the pipeline running.
- **Explicit contracts.** Each HTTP boundary forces a defined JSON schema, preventing tight coupling between components and making the interface between stages stable and documentable.
- **Failure isolation.** If one service is slow or crashes, others are unaffected. In a monolith, one slow component blocks the whole chain.
- **Replaceability.** Swapping a component (e.g. replacing Roboflow with a locally-trained detector, or upgrading to an ML-based decision engine) only requires matching the JSON contract at its boundary — no changes elsewhere.

The one cost is per-hop latency. For a live poker bot this is acceptable — local Flask services add milliseconds, well within the several-second window available between game events.

---

## Modules

| Module | Folder | Purpose |
|---|---|---|
| Screen Monitor | `poker-vision-screen-monitor/` | Captures screenshots of the live game, POSTs them to the orchestrator via a configured webhook URL, and voices the returned decision using Windows TTS (port 5000) |
| Orchestrator | `orchestrator.py` | Receives screenshots from the Screen Monitor, runs the full bot pipeline, and returns the next action (port 5100) |
| Object Detector | `poker-vision-object-detector/` | Runs inference on screenshots, outputs bounding-box JSON per capture |
| Detection Enricher | `poker-vision-detection-enricher/` | Crops detections in memory and enriches them with classification, OCR, and spatial reasoning (port 5004) |
| Hand State Parser | `poker-vision-hand-state-parser/` | Converts enriched detections into the minimal HandState payload required by the decision engine |
| Decision Engine | `poker-vision-decision-engine/` | Consumes HandState, outputs next action for the bot (port 5002) |
| Card Snipper | `poker-vision-card-snipper/` | Crops detected card regions from screenshots into individual snip images (training pipeline) |
| Card Labeller | `poker-vision-card-labeller/` | Interactively assigns rank+suit labels to each snipped card |
| Card Classifier | `poker-vision-card-classifier/` | Classifies cropped card images into rank+suit labels; used in training and at runtime via the Detection Enricher (port 5001) |
| Auto-Annotator | `poker-vision-auto-annotator/` | Generates YOLO-format annotation files for all screenshots using existing Roboflow predictions plus template boxes for classes the model misses (e.g. `player_me`); outputs a dataset ready for Roboflow upload |

---

## Stage-by-stage

### 0. Screen Monitor
- **Port:** 5000
- **Input:** live game window (configured screen region)
- **Output:** screenshot PNG POSTed to the orchestrator webhook URL (`http://127.0.0.1:5100/decide`)
- **Decision handling:** the orchestrator's JSON response (`action`, `amount`, `reason`) is logged at INFO level and spoken aloud via Windows TTS — e.g. *"call 400"*, *"fold"*, *"raise 900"*. Uses `System.Speech.Synthesis.SpeechSynthesizer` via Windows PowerShell 5.1; no extra packages required. Works on any standard Windows 10/11 machine.
- **Setup:** add `http://127.0.0.1:5100/decide` as a webhook URL in the web dashboard at `http://localhost:5000`, enable external sending, and set format to `multipart`
- See `poker-vision-screen-monitor/README.md` for full setup and config

### 1. Object Detector
- **Input:** raw screenshots (not committed — see below)
- **Output:** `poker-vision-object-detector/output/<capture_id>/detections.json`
- **Idempotent:** skips captures whose output folder already exists
- See `poker-vision-object-detector/README.md` for setup and config


### 2. Detection Enricher
- **Input:** screenshot bytes + detector predictions
- **Output:** enriched detections with crop-derived fields such as card `classification`, `ocr_text`, and `spatial_info`
- **Purpose:**
   - Crops detected regions directly from the screenshot in memory
   - Runs card detections through classification
   - Runs OCR on numeric/text regions such as pots, blinds, and chip stacks
   - Runs spatial reasoning for table-position objects such as the dealer button

### 3. Hand State Parser
- **Input:** list of enriched detections
- **Output:** `HandState` object (structured JSON/dataclass)
- **Purpose:**
   - Interprets the enriched table state into the exact schema required by the decision engine
   - Uses explicit MVP defaults for unresolved fields until richer table-state reasoning is built
- See `poker-vision-hand-state-parser/README.md` for full field-by-field extraction logic and confidence gating

### 4. Decision Engine
- **Input:** `HandState`
- **Output:** Decision (action for the hero: fold, call, raise, etc.)
- **Purpose:** Applies poker logic/strategy to decide the next move for the bot (“hero”)

### 5. Card Snipper (training pipeline)
- **Input:** screenshot + detector predictions
- **Output:** cropped card images saved to `poker-vision-card-snipper/output/`
- **Purpose:** produces the snip images used by the Card Labeller to build the training dataset

### 6. Card Labeller
- **Input:** snips directly from `poker-vision-card-snipper/output/` (no copying needed)
- **Output:** `poker-vision-card-labeller/labels.csv` — rows of `filename, label`
- **Idempotent:** resumes from where it left off; already-labelled files are skipped
- `filename` key is the relative path within the snipper output, e.g. `capture_20260219_174930_717002\card_00.png`
- Run: `cd poker-vision-card-labeller && python labeller.py`

### 7. Card Classifier
- **Port:** 5001
- **Input:** card image (PNG/JPG)
- **Output:** predicted rank+suit label with confidence score
- **Purpose:** classifies individual card crops; used at training time to evaluate the model, and at runtime by the Detection Enricher

### 8. Auto-Annotator (training pipeline)
- **Input:** screenshot archive + existing Roboflow raw prediction files
- **Output:** `poker-vision-auto-annotator/output/yolo_dataset/` — YOLO `.txt` label files, image copies, and `data.yaml` ready for Roboflow upload
- **Purpose:** bootstraps a well-labelled training dataset without manual labelling from scratch
  - Converts all existing Roboflow predictions above a confidence threshold to YOLO format
  - Injects template bounding boxes for classes the current model misses consistently (e.g. `player_me` detected in only ~26% of shots before retraining)
  - Template positions are expressed as fractions of the `poker-table` bounding box, so they adapt automatically to any window size or screen resolution
  - `--api` flag fetches fresh predictions from Roboflow for screenshots with no cached result
  - `--preview` / `--preview-only` flags save annotated images for visual verification before upload
- **Calibration:** adjust `cx_rel`/`cy_rel`/`w_rel`/`h_rel` in `auto_annotate_config.yaml` then rerun `--preview-only` to verify
- **Run:**
  ```bash
  uv run python poker-vision-auto-annotator/auto_annotate.py            # existing predictions only
  uv run python poker-vision-auto-annotator/auto_annotate.py --api       # fetch missing predictions
  uv run python poker-vision-auto-annotator/auto_annotate.py --preview   # save preview images
  ```

---

## MVP P0 Status (April 2026)

Focus is on delivering a reliable preflop end-to-end pipeline.

| Module | Status |
|---|---|
| Screen capture | Complete and tested |
| Object detector | Functional but under-trained; needs substantially more labelled screenshots and retraining for robust accuracy. Auto-annotator (`poker-vision-auto-annotator/`) generates a YOLO dataset from the screenshot archive with template-injected `player_me` boxes; 156/165 screenshots annotated, 72 with template. Dataset ready for Roboflow upload and retraining. |
| Detection enricher | OCR real (pytesseract, ~10–50 ms/crop); card classification calls port 5001 via HTTP with graceful fallback. Spatial reasoning fully implemented as a two-pass system: (1) `resolve_spatial_relationships` annotates `dealer_button` with nearest player, each `chip_stack` with the player above it, and each `bet`/`pot_bet` with nearest player; (2) `resolve_hero_position` uses clockwise seat ordering and the dealer annotation to determine the hero's position (BTN/SB/BB) and annotates `player_me`. Both 2-player and 3-player cases handled. `run_ocr` now returns real Tesseract per-word confidence (mean of word-level scores, normalised to 0–1); `ocr_conf` on every enriched object is a genuine quality signal, not a hardcoded constant. |
| Hand state parser | Fully implemented with real confidence-gated extraction for hero cards, blinds, stack, pot, amount-to-call, and `is_hero_turn`. Position now resolved end-to-end: enricher writes `player_me.spatial_info = {"position": "BTN"}` and the parser reads `spatial_info["position"]` — keys now align. **Gap:** position defaults to `"BTN"` when `player_me` is not detected by the object detector (detection quality dependent). Action history and `hero_folded` always use defaults because no enrichment path produces `action`/`player` fields. |
| Decision engine | Preflop rules complete for PREMIUM / STRONG / MEDIUM / WEAK hands across all situations. **Gap:** `SPECULATIVE` hands (suited connectors, suited aces A2s–A9s) have no dedicated rules and fall through to the `WEAK` branch — they fold where a real strategy would call or raise. `classify_situation()` misclassifies SB-completing scenarios (amount_to_call = 0.5 BB) as UNOPENED. |

### Remaining gaps — by impact

#### Critical — pipeline gives wrong or default answers today

1. **Action history is always empty** — no enrichment path produces `action`/`player` fields. `is_hero_turn` defaults to `True` and `hero_folded` defaults to `False` on every real run.

#### Notable — detection quality dependent

3. **Position defaults to BTN when `player_me` is not detected** — the full position pipeline (clockwise seat order → BTN/SB/BB) is implemented and tested, but requires the object detector to reliably fire a `player_me` detection. If it misses, position falls back to `"BTN"`. Improving detector training is the unlock.

#### Low-risk cleanup (resolved)

- ~~README parser description stated "mostly mocked/default-driven"~~ — corrected; the parser has real confidence-gated extraction logic.
- ~~Missing `facing_limp` tests~~ — added to `test_preflop.py`.
- ~~Missing `SPECULATIVE` hand tests~~ — added to `test_preflop.py`.
- ~~`spatial_reasoning.assign_dealer()` is a stub~~ — replaced with `resolve_spatial_relationships` + `resolve_hero_position` two-pass system.
- ~~Hand state parser does not consume spatial output~~ — enricher now writes `player_me.spatial_info = {"position": "BTN"|"SB"|"BB"}`; parser reads `spatial_info["position"]`; keys align.
- ~~`ocr_conf` hardcoded to 0.60~~ — `run_ocr` now returns real Tesseract per-word confidence; gating thresholds operate on genuine quality signals.

### Priority order

```
1. player_me detection quality  — position pipeline is complete; unlock via detector training
2. is_hero_turn via buttons     — dependent on object detector reliability improving first
3. End-to-end validation        — run against representative screenshots
```

---

## Architecture Considerations (Deferred)

### Hand-scoped context preservation (caching)

Some facts derived from a screenshot are **stable for the entire hand** and do not need to be recomputed on every frame. The dealer position is the clearest example — once identified, it does not change until the next hand begins. Re-running spatial reasoning on every screenshot is wasted work.

A lightweight per-hand cache (e.g. a dict keyed by hand ID or a "hand started" signal from the screen monitor) could store results that are stable for the hand's lifetime and skip recomputation on subsequent frames. Candidate fields for caching:

- **Dealer / button position** — set once when the dealer button is first detected; unchanged until a new hand.
- **Seat layout** — the spatial positions of `player_name` bounding boxes relative to the table; these don't move mid-hand.
- **Hero position (BTN/SB/BB)** — derived from dealer + seat layout via `resolve_hero_position`; stable for the entire hand once resolved.
- **Blinds** — posted at the start of the hand; stable until the next hand.

This is intentionally not implemented now. Introducing a cache requires a mechanism to detect hand boundaries (new hand = dealer button moved, new community cards dealt, pot reset) and a place in the architecture to hold that state — likely the Orchestrator. The correctness benefit is low until the pipeline is otherwise reliable. Revisit once end-to-end validation is complete.

---

## Parser-Enricher Contract (Preflop MVP)

To keep the preflop pipeline stable while vision quality improves, the detection enricher and hand state parser now use an explicit confidence-and-fallback contract.

### Enriched object confidence keys

Every enriched object includes core detection fields:

- `class_name`
- `bbox_xyxy`
- `confidence`

Processing-specific confidence fields are added only when that processing path is used:

- Classification path: `classification`, `classification_conf`
- OCR path: `ocr_text`, `ocr_conf`
- Spatial path: `spatial_info`, `spatial_conf`

Unsupported processing paths are marked with `processing: "none"` and do not include extraction confidence keys.

### Parser confidence bands

For each HandState field candidate, parser confidence is computed as:

- `field_conf = min(detection_conf, extraction_conf)`

Acceptance bands:

- Trusted: `field_conf >= 0.80`
- Usable (warning): `0.55 <= field_conf < 0.80`
- Rejected: `field_conf < 0.55` (fallback used)

### Fallback philosophy

MVP reliability is prioritized over completeness. If confidence is too low or source data is missing, the parser emits a valid HandState with explicit defaults (for example: hero cards, blinds, stack, pot, amount-to-call) so the decision engine can still operate deterministically.

---

## Setup

This is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/). From the repo root:

```bash
uv sync
```

This installs all dependencies for all modules into the shared `.venv`.

---

## Running the bot pipeline

### 1. Start all services

```bash
uv run python manage_services.py start
```

This checks each service's `/health` endpoint first. Services that are already running are skipped. Any service that is not running is launched as a background process, and the script waits (up to 30 s) for it to become healthy before moving on.

Service logs are written to `logs/<service-name>.log` in the repo root:

```
logs/detection-enricher.log
logs/hand-state-parser.log
logs/decision-engine.log
logs/orchestrator.log
```

The PID of each spawned process is recorded in `.services.pids` (repo root) so the stop command can find them later.

### 2. Check service health

```bash
uv run python manage_services.py status
```

Prints `up` or `DOWN` for each service. Exits with code 1 if any service is down — useful for scripting.

### 3. Run the pipeline against a screenshot

```bash
uv run python orchestrate_pipeline.py                                        # uses default screenshot
uv run python orchestrate_pipeline.py path/to/screenshot.png                # specific screenshot
uv run python orchestrate_pipeline.py path/to/screenshot.png --verbose      # diagnostic mode
```

Prints the decision JSON returned by the orchestrator and **speaks the action aloud** (e.g. *"call 400"*, *"fold"*) using Windows TTS. `watch` and `wait` states are silent.

**`--verbose` mode** bypasses the orchestrator and calls each service directly, printing intermediate outputs at every pipeline stage:

```
--- Stage 1: Object Detector ---
  8 detections:
    dealer_button  (conf 0.97)
    player_name    (conf 0.94)
    ...

--- Stage 2: Detection Enricher ---
  dealer_button   → dealer: "Rajiv"  (conf 0.97)
  player_me       → position: BTN    (conf 0.94)
  player_name     → "Rajiv"  stack: 1450  (conf 0.94)
  player_name     → "Alice"  stack: 2800  (conf 0.91)
  holecard        → Ah  (det 0.95, cls 0.93)
  holecard        → Kd  (det 0.95, cls 0.91)
  blinds          → 50/100  (conf 0.88)

--- Stage 3: Hand State ---
{
  "hero_cards": ["Ah", "Kd"],
  "position": "BTN",
  ...
}

--- Stage 4: Decision ---
{
  "action": "raise",
  "amount": 300,
  "reason": "..."
}
```

Use `--verbose` to validate the spatial reasoning, OCR extraction, and hand state fields when testing against new screenshots.

### 4. Stop all services

```bash
uv run python manage_services.py stop
```

Reads `.services.pids`, terminates each process tree (including any child processes spawned by `uv`), and deletes the PID file.

> **Note:** If services were started manually (not via `manage_services.py start`), the stop command will not know about them. Stop those manually or via your process manager.

### `.services.pids`

Runtime file created and deleted by `manage_services.py`. Contains a JSON map of service name → PID:

```json
{
  "detection-enricher": 18432,
  "hand-state-parser": 21104,
  "decision-engine": 9876,
  "orchestrator": 14200
}
```

If the file is stale (e.g. after a machine restart or manual kill), `manage_services.py stop` handles it gracefully — it reports each missing process and still cleans up the file. Do not commit this file.

---

## Phase 2 considerations

These are architectural evolutions planned for after the MVP pipeline is working end-to-end.

### 1. AI agent orchestration

The current bot pipeline is **deterministic and sequential** — every stage always runs in the same fixed order. A natural phase 2 upgrade is to replace the hard-coded orchestration with an **AI agent** that decides the flow dynamically based on what it observes.

**What an AI agent means here:** an LLM (e.g. GPT-4, Claude) that can call your Flask APIs as **tools**, and reasons about the results to decide what to call next. Rather than always running every step, it can adapt:

> "Confidence on two of the detected cards is below threshold — I should re-classify those specifically before making a decision."

The current Flask APIs are already structured exactly like agent tools. In frameworks like [LangGraph](https://www.langchain.com/langgraph) or [LangChain](https://www.langchain.com/), a tool is just a name, a description, and an HTTP call:

```python
# Note: port assignments below are illustrative.
# Actual services: Screen Monitor (5000), Card Classifier (5001),
# Decision Engine (5002), Detection Enricher (5004), Orchestrator (5100).
# The Object Detector is a CLI/batch tool with no HTTP API.
tools = [
    Tool(name="enrich",   description="Enrich detections from a screenshot", api="POST localhost:5004/enrich"),
    Tool(name="classify", description="Classify a card image",               api="POST localhost:5001/classify"),
    Tool(name="decide",   description="Get a poker decision",                api="POST localhost:5002/decide"),
]
```

The agent runs a **ReAct loop** (Reason → Act → Observe → repeat) — one of the foundational patterns in modern AI agent design:

```
observe: screenshot taken
reason:  "I should detect objects in it"
act:     POST /detect
observe: two cards have low confidence
reason:  "I should re-classify those two cards before deciding"
act:     POST /classify (targeted)
observe: cards now confirmed
reason:  "I have enough context to make a decision"
act:     POST /decide
observe: fold recommended
act:     execute fold
```

**Key concepts this covers** (useful for interviews and further study):

| Concept | Description |
|---|---|
| Tool calling / function calling | Giving an LLM the ability to invoke external APIs |
| ReAct pattern | Reason + Act loop — a core agent architecture pattern |
| LangGraph / LangChain | Frameworks for building agent pipelines |
| Structured outputs | Getting an LLM to return typed, parseable JSON |
| Prompt engineering | Instructing the agent with domain context (poker rules, stack sizes, etc.) |
| Agent memory | Persisting hand history so the agent reasons across streets |

### 2. Hybrid decision engine

The MVP decision engine is purely rule-based. For phase 2, the engine can evolve into a **hybrid**:

- **Rules handle the clear-cut math** — short stack shoves, obvious folds vs all-in with weak hands
- **LLM reasoning handles grey areas** — ambiguous spots where hand category alone isn't enough (e.g. a speculative hand facing a small raise from a positional disadvantage)

The LLM would receive a natural language description of the situation and reason about it the way a strong human player would, rather than matching against a fixed rule table.

### 3. Postflop logic

The MVP is preflop only. Phase 2 adds flop decision-making, which requires:
- Hand strength evaluation (made hand vs draw vs air)
- Board texture analysis (wet vs dry, paired, etc.)
- Pot odds calculation for draw decisions

### 4. Opponent modelling

With hand history accumulated over a session, it becomes possible to build simple frequency-based profiles per opponent (e.g. aggression frequency, fold-to-3bet %). This feeds into both rule adjustments and LLM context.

---

## Longer-term considerations (deferred)

Architectural points noted for the future — not blockers for MVP.

- **Formal JSON schema for `HandState`.** The input contract between the Game State Parser and the Decision Engine should eventually be pinned as a JSON Schema (or Pydantic model) so both sides can validate against it independently. Card notation (`"Ah"` = rank uppercase, suit lowercase) and `action_history` action strings need to be agreed across both modules.

- **API error handling strategy.** Decision on whether to use HTTP status codes for errors (400 for bad input) or always return 200 with an error field in the payload. Current pattern in the card classifier is 200 + payload signal — the decision engine should be consistent with that.

- **`GET /health` on all services.** Useful for an orchestrator to ping all services on startup to confirm the pipeline is up before attempting a full run. Trivial to add to a stateless service like the decision engine.

- **Hardening the decision engine API against malformed vision-layer input.** Currently only manual callers and test scripts use the API. Once the Game State Parser is built and feeding it real data, input validation should be tightened.

---

## What to commit to git

| Artifact | Committed? | Notes |
|---|---|---|
| Source code | Yes | All modules |
| `labels.csv` | Yes | Irreplaceable human annotation work |
| `detect_config.yaml` | Yes | Detection config |
| `.env.example` | Yes | Template only — never the real `.env` |
| Raw screenshots | **No** | Large raw data — store externally |
| Detection JSON | **No** | Derived, regenerate from screenshots |
| Snipped card images | **No** | Derived, regenerate from JSON |
| Model weights | **No** | Large, derived — store in releases or DVC |
| `.services.pids` | **No** | Runtime PID file — machine-specific, deleted on stop |
| `logs/` | **No** | Runtime service logs — regenerated each start |

`.gitignore` should cover:
```
.env
poker-vision-object-detector/output/
poker-vision-card-snipper/output/
.services.pids
logs/
```
