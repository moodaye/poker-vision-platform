# poker-vision-card-snipper

Extracts individual card images ("snips") from poker-table screenshots using bounding boxes produced by the object detector. Snips are the training-data input to the card classifier.

---

## Purpose in the pipeline

```
Screenshot
    │
    ▼
poker-vision-object-detector      → detects card bounding boxes
    │
    ▼
poker-vision-card-snipper  ◄─── YOU ARE HERE
    │   crops each bbox from the screenshot
    │   saves card_00.png … card_NN.png
    ▼
poker-vision-card-labeller        → user assigns labels (TD, QH, …)
    │
    ▼
poker-vision-card-classifier      → trains EfficientNet-B0 on labelled crops
```

The snipper does not classify — it only crops. Classification happens downstream.

---

## Entry points

The module has three separate entry points for different use cases:

| Entry point | Use case | How to run |
|---|---|---|
| `card_snipper.py` | Library — used by the API and by the enricher | `from card_snipper import snip_flop_cards` |
| `api.py` / `run.py` | HTTP service — called by the detection enricher in the live pipeline | `uv run python run.py` |
| `batch.py` | CLI tool — offline processing of a full session's screenshots for training-data collection | `uv run python batch.py --detections … --images … --output …` |

---

## card_snipper.py — Core library

The extraction logic. Used by both `api.py` and `batch.py`.

### What it does

1. Receives a PIL `Image` and a list of detection dicts.
2. Filters detections to those with `class_name == "flop_card"` (configurable via `target_class`).
3. Sorts detections left-to-right by their `x1` coordinate so `card_00` is always the leftmost card on the table.
4. Crops each bounding box from the image, clamping coordinates to image bounds.
5. Returns a list of `CardSnip` dataclass objects.

### `CardSnip` dataclass

```python
@dataclass
class CardSnip:
    index:      int           # 0-based, left-to-right order
    confidence: float         # detector confidence for this bbox
    bbox_xyxy:  list[int]     # [x1, y1, x2, y2] in pixels (after clamping)
    image:      Image.Image   # PIL Image of the cropped card
```

### Function signature

```python
def snip_flop_cards(
    image:        Image.Image,
    detections:   list[dict],
    *,
    target_class: str = "flop_card",
) -> list[CardSnip]:
```

### Detection dict format

Each detection dict must contain:

```python
{
    "class_name": "flop_card",   # string — must match target_class
    "bbox_xyxy":  [x1, y1, x2, y2],  # integers, pixel coordinates
    "confidence": 0.995,         # float, optional (defaults to 0.0 if absent)
}
```

This matches the output format of `poker-vision-object-detector` and `poker-vision-detection-enricher`.

### Example

```python
from PIL import Image
from card_snipper import snip_flop_cards

image = Image.open("screenshot.png")
detections = [
    {"class_name": "flop_card", "bbox_xyxy": [762, 388, 850, 513], "confidence": 0.99},
    {"class_name": "flop_card", "bbox_xyxy": [860, 388, 948, 513], "confidence": 0.97},
]

cards = snip_flop_cards(image, detections)
for card in cards:
    card.image.save(f"card_{card.index:02d}.png")
    print(f"card_{card.index:02d}: bbox={card.bbox_xyxy}, conf={card.confidence:.3f}")
```

---

## api.py — HTTP service

FastAPI service called by `poker-vision-detection-enricher` during live inference to extract card crops from a screenshot on-the-fly.

### Start the service

```bash
uv run python run.py
# Starts on http://localhost:8000 with hot-reload
```

### Endpoints

#### `GET /health`

Liveness check.

```json
{"status": "ok"}
```

#### `POST /snip`

Accepts a multipart/form-data request.

**Form fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | ✓ | PNG or JPG screenshot |
| `detections_json` | string | ✓ | Detection payload JSON — either the full detector output object (with a `"detections"` key) or a bare JSON array of detections |
| `output_format` | string | — | `"json"` (default) or `"zip"` |

**JSON response (`output_format=json`):**

```json
{
  "source_image": "screenshot.png",
  "card_count": 3,
  "flop_cards": [
    {
      "index": 0,
      "confidence": 0.995522,
      "bbox_xyxy": [762, 388, 850, 513],
      "filename": "flop_card_00.png",
      "image_base64": "<base64-encoded PNG>"
    }
  ]
}
```

**ZIP response (`output_format=zip`):**

Returns `flop_cards.zip` containing `flop_card_00.png … flop_card_NN.png`.

**Error responses:**

| Status | Reason |
|---|---|
| 422 | Invalid `output_format`, malformed JSON, missing `detections` key, or unreadable image |

### `detections_json` format flexibility

The field accepts two shapes:

```json
// Shape 1: full detector payload (recommended — same JSON the orchestrator passes)
{"detections": [...], "image_width": 1920, "image_height": 1080}

// Shape 2: bare array (useful for ad-hoc calls)
[{"class_name": "flop_card", "bbox_xyxy": [762, 388, 850, 513], "confidence": 0.99}]
```

---

## batch.py — Offline batch processor

CLI tool for extracting all card crops from a recorded UAT session. Used for training-data collection: run it once after a session to populate `output/` with labellable crops.

### How it works

1. Scans `--detections` folder for `*.detections.json` files.
2. For each detection file, looks up the matching screenshot in `--images` by stem name (e.g., `capture_20260219_175731.detections.json` → `capture_20260219_175731.png`).
3. Runs `snip_flop_cards()` to extract crops.
4. Saves crops to `--output/<stem>/card_00.png … card_NN.png`.
5. Writes a `summary.json` to the output root.

### Usage

```bash
uv run python batch.py \
    --detections ../poker-vision-object-detector/output/detections_normalized \
    --images     ../poker-vision-object-detector/screenshots \
    --output     ./output
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--detections` | ✓ | — | Folder of `*.detections.json` files from the object detector |
| `--images` | ✓ | — | Folder of screenshot images (matched by filename stem) |
| `--output` | — | `./output` | Destination for snipped card folders |

### Output layout

```
output/
    capture_20260219_175731_401677/
        card_00.png     ← leftmost card
        card_01.png
        card_02.png
    capture_20260219_175742_066987/
        card_00.png
        card_01.png
        card_02.png
        card_03.png
        card_04.png     ← 5-card board
    summary.json        ← per-capture processing report
```

Cards within each folder are numbered `card_00` → `card_NN` in left-to-right table order.

### summary.json

```json
[
  {
    "detections_file": "capture_20260219_175731_401677.detections.json",
    "image_file": "capture_20260219_175731_401677.png",
    "status": "ok",
    "card_count": 3,
    "output_dir": "output/capture_20260219_175731_401677",
    "cards": [
      {"filename": "card_00.png", "confidence": 0.994, "bbox_xyxy": [762, 388, 850, 513]},
      ...
    ]
  },
  {
    "detections_file": "some_other.detections.json",
    "status": "skipped",
    "reason": "image not found"
  }
]
```

---

## run.py — Service runner

Thin wrapper that starts `api.py` via uvicorn with hot-reload.

```bash
uv run python run.py
# Equivalent to: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Not used in production deployments (managed via `manage_services.py` from the project root).

---

## Output directory (`output/`)

The `output/` directory is populated by `batch.py` and is the source directory for `poker-vision-card-labeller`. It is **not committed to git**.

### Current state (as of Phase 0 inventory — 2026-07-11)

| Metric | Value |
|---|---|
| Total capture folders | ~155 |
| Total crop files | 356 |
| Labelled crops | 349 (98%) |
| Unlabelled crops | 7 (all in `capture_holecard_runtime_training/`) |
| Distinct cards labelled | 52 / 52 |

The `capture_holecard_runtime_training/` folder contains hero hole-card crops from a live session. These 7 crops should be labelled to confirm they're correctly attributed.

### Snipping the test-screenshots

The `test-screenshots/` folder at the project root contains 18 preflop UAT screenshots (`screenshot_preflop_1.png` … `screenshot_preflop_18.png`). These have **not** been processed by `batch.py`. To extract crops from them:

1. Run the object detector on each screenshot to produce `*.detections.json` files.
2. Run `batch.py` pointing at the detection outputs and the `test-screenshots/` folder.
3. Label the resulting crops with `poker-vision-card-labeller`.

This is the primary source of new training data for under-represented classes (Td, Qd, 7S, and others with ≤3 labelled examples).

---

## Dependencies

```toml
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pillow>=10.0.0
python-multipart>=0.0.9
```

No PyTorch, no model weights — the snipper is a pure image-cropping module.

---

## Tests

```
tests/
    test_card_snipper.py    ← unit tests for snip_flop_cards()
```

```bash
cd poker-vision-card-snipper
uv run pytest tests/ -v
```
