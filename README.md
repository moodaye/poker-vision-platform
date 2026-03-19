# Poker Vision — Pipeline Overview

End-to-end pipeline for building a training dataset of labelled poker card images.

```
screenshots/  →  detector  →  detection JSON  →  snipper  →  snips  →  labeller  →  labels.csv  →  trainer  →  model
```

---

## Modules

| Module | Folder | Purpose |
|---|---|---|
| Object Detector | `poker-vision-object-detector/` | Runs inference on screenshots, outputs bounding-box JSON per capture |
| Card Snipper | `poker-vision-card-snipper/` | Crops detected card regions into individual image files |
| Card Labeller | `poker-vision-card-labeller/` | Interactively assigns rank+suit labels to each snipped card |

---

## Stage-by-stage

### 1. Object Detector
- **Input:** raw screenshots (not committed — see below)
- **Output:** `poker-vision-object-detector/output/<capture_id>/detections.json`
- **Idempotent:** skips captures whose output folder already exists
- See `poker-vision-object-detector/README.md` for setup and config

### 2. Card Snipper
- **Input:** detection JSON from stage 1
- **Output:** `poker-vision-card-snipper/output/<capture_id>/card_NN.png`
- **Idempotent:** skips captures whose output folder already exists
- Run: `cd poker-vision-card-snipper && python run.py`

### 3. Card Labeller
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
