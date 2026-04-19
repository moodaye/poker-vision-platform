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
       ↓  screenshot PNG
Remote Orchestrator (NEW)
       ↓  (receives screenshot, runs full pipeline)
    Object Detector (Roboflow)
       ↓  detections
    Detection Enricher (5004)
       ↓  enriched detections: [{class_name, bbox_xyxy, classification|ocr_text|spatial_info, confidence}]
    Hand State Parser
       ↓  HandState JSON
    Decision Engine (5002)
       ↓  Decision (action + amount + reason)
Bot Action Executor (TBD)
```

#### Orchestrator Service Contract

- **Input:**
  - POST `/decide` (multipart/form-data)
  - `image`: screenshot (PNG/JPG)
- **Output:**
  - JSON: `{ "action": "call", "amount": 400, "reason": "Standard preflop call with suited connectors" }`
- The orchestrator handles all downstream calls (detection, enrichment, hand-state parsing, decision) and returns the next action.

### Why each component runs as a separate API

Each stage in the bot pipeline is a **standalone Flask service** with its own port. This architecture was chosen deliberately:

- **Independent development and testing.** Any component can be run and tested in isolation with a `curl` or test script, without the rest of the pipeline running.
- **Explicit contracts.** Each HTTP boundary forces a defined JSON schema, preventing tight coupling between components and making the interface between stages stable and documentable.
- **Failure isolation.** If one service is slow or crashes, others are unaffected. In a monolith, one slow component blocks the whole chain.
- **Replaceability.** Swapping a component (e.g. replacing Roboflow with a locally-trained detector, or upgrading to an ML-based decision engine) only requires matching the JSON contract at its boundary — no changes elsewhere.

The one cost is per-hop latency. For a live poker bot this is acceptable — local Flask services add milliseconds, well within the several-second window available between game events.

---

## Modules

| Module | Folder | Purpose |
|---|---|---|
| Object Detector | `poker-vision-object-detector/` | Runs inference on screenshots, outputs bounding-box JSON per capture |
| Detection Enricher | `poker-vision-detection-enricher/` | Crops detections in memory and enriches them with classification, OCR, and spatial reasoning |
| Hand State Parser | `hand_state_parser.py` | Converts enriched detections into the minimal HandState payload required by the decision engine |
| Card Labeller | `poker-vision-card-labeller/` | Interactively assigns rank+suit labels to each snipped card |
| Decision Engine | `poker-vision-decision-engine/` | Consumes HandState, outputs next action for the bot |

---

## Stage-by-stage

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

### 4. Decision Engine
- **Input:** `HandState`
- **Output:** Decision (action for the hero: fold, call, raise, etc.)
- **Purpose:** Applies poker logic/strategy to decide the next move for the bot (“hero”)

### 5. Card Labeller
- **Input:** snips directly from `poker-vision-card-snipper/output/` (no copying needed)
- **Output:** `poker-vision-card-labeller/labels.csv` — rows of `filename, label`
- **Idempotent:** resumes from where it left off; already-labelled files are skipped
- `filename` key is the relative path within the snipper output, e.g. `capture_20260219_174930_717002\card_00.png`
- Run: `cd poker-vision-card-labeller && python labeller.py`

---

## Setup

This is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/). From the repo root:

```bash
uv sync
```

This installs all dependencies for all modules into the shared `.venv`.

---

## Phase 2 considerations

These are architectural evolutions planned for after the MVP pipeline is working end-to-end.

### 1. AI agent orchestration

The current bot pipeline is **deterministic and sequential** — every stage always runs in the same fixed order. A natural phase 2 upgrade is to replace the hard-coded orchestration with an **AI agent** that decides the flow dynamically based on what it observes.

**What an AI agent means here:** an LLM (e.g. GPT-4, Claude) that can call your Flask APIs as **tools**, and reasons about the results to decide what to call next. Rather than always running every step, it can adapt:

> "Confidence on two of the detected cards is below threshold — I should re-classify those specifically before making a decision."

The current Flask APIs are already structured exactly like agent tools. In frameworks like [LangGraph](https://www.langchain.com/langgraph) or [LangChain](https://www.langchain.com/), a tool is just a name, a description, and an HTTP call:

```python
tools = [
    Tool(name="detect",   description="Detect objects in a screenshot", api="POST localhost:5000/detect"),
    Tool(name="classify", description="Classify a card image",          api="POST localhost:5001/classify"),
    Tool(name="decide",   description="Get a poker decision",           api="POST localhost:5002/decide"),
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

`.gitignore` should cover:
```
.env
poker-vision-object-detector/output/
poker-vision-card-snipper/output/
```
