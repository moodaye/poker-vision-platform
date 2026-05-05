# poker-vision-auto-annotator

Generates YOLO-format annotation files for the screenshot archive and uploads them to Roboflow for model retraining.

Part of the training pipeline:

```
screenshot archive + cached Roboflow predictions
        ↓
    auto_annotate.py  (generate YOLO labels + data.yaml)
        ↓
    upload_to_roboflow.py  (upload images + annotations via API)
        ↓
    Roboflow — generate new dataset version → retrain model
```

---

## Purpose

The current object detector (Roboflow YOLOv8) misses some classes consistently — most notably `player_me`, which it detects in only ~26% of screenshots before retraining. Manually labelling hundreds of screenshots from scratch would be very slow.

This module bootstraps a well-labelled training dataset automatically:

1. **Converts existing Roboflow predictions** (cached as JSON) to YOLO `.txt` label files, filtering to detections above a confidence threshold.
2. **Injects template bounding boxes** for classes the model misses. Template positions are expressed as fractions of the `poker-table` bounding box, so they adapt to any window size or screen resolution automatically.
3. **Writes `data.yaml`** — the class list file required by YOLO / Roboflow when uploading a zip dataset.
4. **Uploads to Roboflow** — `upload_to_roboflow.py` sends each image and its annotation via the REST API, explicitly passing the class names so they are correctly stored.

---

## How it works

### Step 1 — Build class vocabulary

`collect_class_vocabulary` scans all cached prediction JSON files (from `poker-vision-object-detector/output/predictions_raw/`) and collects every class name seen. Template class names (e.g. `player_me`) are added too. The result is a sorted, deduplicated list. Each class is assigned a numeric ID by its position in that list — this ID is what appears in the YOLO `.txt` files.

### Step 2 — Process each screenshot

For each `.png` / `.jpg` in the screenshot archive:

1. Load the cached Roboflow prediction JSON (if `--api` is set and no file exists, call the Roboflow API and cache the result).
2. Filter predictions to those with `confidence >= min_confidence` (default `0.70`) and not in `skip_classes`.
3. Convert each accepted detection from Roboflow's centre-based pixel coordinates to YOLO normalised format: `(class_id, cx, cy, w, h)` all in `[0, 1]`.
4. **Template pass:** if a `poker-table` detection is present and a template class was not already detected, add a template box. The template's centre and size are defined relative to the poker-table bounding box in `auto_annotate_config.yaml`.
5. Write a `<stem>.txt` YOLO label file to `output/yolo_dataset/labels/`.
6. Copy the image to `output/yolo_dataset/images/`.
7. Optionally draw annotated preview images to `output/yolo_previews/` (with `--preview`).

### Step 3 — Upload to Roboflow

`upload_to_roboflow.py` reads `output/yolo_dataset/data.yaml` to build the labelmap (`{"0": "bet_pot_button", "1": "check_button", ...}`), then for each image:

1. POST the base64-encoded image to `/dataset/{project}/upload`.
2. POST the `.txt` annotation to `/dataset/{project}/annotate/{image_id}`, passing the `labelmap` as a JSON query parameter so Roboflow stores class names rather than raw numeric IDs.

---

## Files

| File | Purpose |
|---|---|
| `auto_annotate.py` | Generates YOLO label files from cached predictions + templates |
| `upload_to_roboflow.py` | Uploads images and annotations to Roboflow via REST API |
| `auto_annotate_config.yaml` | Template box positions, confidence threshold, directory paths |
| `output/yolo_dataset/images/` | Copies of processed screenshots *(gitignored)* |
| `output/yolo_dataset/labels/` | YOLO `.txt` annotation files *(gitignored)* |
| `output/yolo_dataset/data.yaml` | Class list for YOLO / Roboflow *(gitignored)* |
| `output/yolo_previews/` | Annotated preview images for visual verification *(gitignored)* |

---

## Setup

Dependencies are managed from the repo root:

```powershell
cd c:\...\pokerProject
uv sync
```

The Roboflow API key is read from `poker-vision-object-detector/.env`:

```
ROBOFLOW_API_KEY=your_actual_key
```

---

## Running

### Generate labels (existing cached predictions only)

```powershell
cd poker-vision-auto-annotator
uv run python auto_annotate.py
```

### Generate labels (also fetch missing predictions from Roboflow API)

```powershell
uv run python auto_annotate.py --api
```

### Generate labels + save annotated preview images

```powershell
uv run python auto_annotate.py --preview
```

Preview images are written to `output/yolo_previews/` — open these to visually verify that boxes are correctly placed before uploading.

### Preview a single screenshot without writing any label file

```powershell
uv run python auto_annotate.py --preview-only path/to/screenshot.png
```

Useful for calibrating template positions. The preview is saved next to the input image as `<stem>_preview.jpg`.

### Upload to Roboflow (dry run first)

```powershell
uv run python upload_to_roboflow.py --dry-run
uv run python upload_to_roboflow.py
```

`--dry-run` prints what would be uploaded without making any API calls.

---

## Calibrating template positions

Template positions in `auto_annotate_config.yaml` are expressed as fractions of the `poker-table` bounding box:

```yaml
templates:
  player_me:
    cx_rel: 0.510   # centre-x, as fraction of table width from table left edge
    cy_rel: 0.840   # centre-y, as fraction of table height from table top edge
    w_rel:  0.155   # width, as fraction of table width
    h_rel:  0.237   # height, as fraction of table height
```

To adjust a template:

1. Run `--preview-only` on a representative screenshot.
2. Open the saved `<stem>_preview.jpg`.
3. Check that the template box (cyan for `player_me`) covers the correct region.
4. Adjust `cx_rel` / `cy_rel` / `w_rel` / `h_rel` and repeat until satisfied.
5. Re-run the full batch with `--preview` to verify across multiple screenshots.

---

## Class vocabulary

The YOLO numeric class IDs are assigned alphabetically from all classes seen in the cached prediction files. As of the last run:

| ID | Class |
|----|-------|
| 0 | bet_pot_button |
| 1 | check_button |
| 2 | check_fold_button |
| 3 | chip_stack |
| 4 | dealer_button |
| 5 | flop_card |
| 6 | fold_button |
| 7 | holecard |
| 8 | level |
| 9 | nextblinds |
| 10 | player_me |
| 11 | player_name |
| 12 | player_other |
| 13 | poker-table |
| 14 | prizepool |
| 15 | total_pot |
| 16 | win_card |
| 17 | win_chips |

These names match exactly what the Roboflow model returns and what the rest of the pipeline (detection enricher, hand state parser, orchestrator) expects.

---

## Known issues

### Duplicate image uploads silently skip annotation

When `upload_to_roboflow.py` uploads an image that already exists in the Roboflow dataset, the API returns `{"duplicate": true}` with no `id` field. The current code does `data.get("id")` which returns `None`, causing the annotation upload to be skipped for that image. This means re-running the uploader after a partial run will leave previously-uploaded images without updated annotations.

### `data.yaml` uses integer-keyed dict format

`auto_annotate.py` writes `names` as an integer-keyed YAML dict (`{0: class_name, ...}`) rather than a YAML list (`- class_name`). Both are equivalent for `upload_to_roboflow.py` (which reads it programmatically), but Roboflow's web UI ZIP import expects the list format. If uploading via ZIP, this can cause class names to be stored as raw numbers. Use `upload_to_roboflow.py` instead of ZIP upload to avoid this.

---

## After uploading

Once images and annotations are uploaded to Roboflow:

1. Open the Roboflow project (`pokertabledetection`).
2. Review a sample of annotations in the **Annotate** tab.
3. Generate a new dataset version (**Dataset** tab → **Generate**).
4. Retrain the model on the new version.
5. Update `detect_config.yaml` in `poker-vision-object-detector/` to point to the new model version.
