"""auto_annotate.py — auto-generate YOLO annotation files for the screenshot archive.

Each screenshot is processed as follows:
  1. Load raw predictions from predictions_dir/ (Roboflow JSON format).
     If no prediction file exists, optionally call the Roboflow API (--api flag).
  2. Identify the poker-table bounding box — used as a layout anchor so
     template positions adapt to any window size or screen position.
  3. Convert all detections above `min_confidence` to YOLO format.
  4. For `player_me` (detected in only ~26% of shots by the current model):
     if not already detected AND a poker-table anchor is available, add a
     template bounding box at the configured relative position.
  5. Write <stem>.txt YOLO label files and a data.yaml class list ready for
     upload to Roboflow as a training dataset.

Usage
-----
Process all screenshots using existing prediction files only:
    uv run python auto_annotate.py

Also call the Roboflow API for screenshots with no existing prediction file:
    uv run python auto_annotate.py --api

Save annotated preview images alongside labels (for visual verification):
    uv run python auto_annotate.py --preview

Preview a single image without writing any label file (quick calibration check):
    uv run python auto_annotate.py --preview-only path/to/screenshot.png

Output
------
    output/yolo_dataset/
        images/          ← copies of processed screenshots
        labels/          ← one YOLO .txt file per screenshot
        data.yaml        ← class list for Roboflow upload
    output/yolo_previews/
        *.png            ← annotated images (only with --preview or --preview-only)

Uploading to Roboflow
---------------------
    zip -r dataset.zip output/yolo_dataset/
Then upload dataset.zip to your Roboflow project using the web UI or:
    pip install roboflow
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_KEY")
    project = rf.workspace("YOUR_WORKSPACE").project("pokertabledetection")
    project.version(6).upload("dataset.zip")
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import requests
import yaml
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_CONFIG_PATH = _HERE / "auto_annotate_config.yaml"
load_dotenv(dotenv_path=_HERE / "../poker-vision-object-detector" / ".env")

ROBOFLOW_API_URL = os.environ.get(
    "ROBOFLOW_API_URL", "https://detect.roboflow.com/pokertabledetection/6"
)
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")

# Colour palette for preview images (class_name → RGB)
_PREVIEW_COLOURS: dict[str, tuple[int, int, int]] = {
    "player_me": (0, 255, 255),  # cyan  — template generated
    "player_other": (255, 165, 0),  # orange
    "player_name": (255, 255, 0),  # yellow
    "chip_stack": (0, 255, 0),  # green
    "holecard": (255, 0, 128),  # pink
    "flop_card": (255, 0, 128),
    "turn_card": (255, 0, 128),
    "river_card": (255, 0, 128),
    "dealer_button": (255, 255, 255),  # white
    "poker-table": (100, 100, 255),  # blue
    "total_pot": (200, 200, 0),
    "blinds": (200, 200, 0),
}
_DEFAULT_COLOUR = (180, 180, 180)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: Path = _CONFIG_PATH) -> dict[str, Any]:
    with open(path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Prediction loading / API call
# ---------------------------------------------------------------------------


def load_prediction(pred_path: Path) -> dict[str, Any] | None:
    """Load a Roboflow raw-prediction JSON file. Returns None if not found."""
    if not pred_path.exists():
        return None
    with open(pred_path) as f:
        return json.load(f)  # type: ignore[return-value]


def call_roboflow_api(image_path: Path) -> dict[str, Any]:
    """Call the Roboflow API and return the raw prediction payload."""
    if not ROBOFLOW_API_KEY:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set — cannot call API. "
            "Either add it to poker-vision-object-detector/.env or export it."
        )
    with open(image_path, "rb") as fh:
        response = requests.post(
            ROBOFLOW_API_URL,
            params={"api_key": ROBOFLOW_API_KEY},
            files={"file": fh},
            timeout=30,
        )
    response.raise_for_status()
    return response.json()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _xyxy(pred: dict[str, Any]) -> tuple[float, float, float, float]:
    """Convert Roboflow center-based prediction to (x0, y0, x1, y1)."""
    cx, cy, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _to_yolo(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Normalise absolute-pixel center+size to YOLO [0,1] format."""
    return cx / img_w, cy / img_h, w / img_w, h / img_h


# ---------------------------------------------------------------------------
# Core annotation logic
# ---------------------------------------------------------------------------


def build_annotations(
    raw: dict[str, Any],
    class_to_id: dict[str, int],
    templates: dict[str, dict[str, float]],
    min_confidence: float,
    skip_classes: set[str],
) -> list[tuple[int, float, float, float, float]]:
    """Return list of (class_id, cx_n, cy_n, w_n, h_n) for one screenshot.

    Detected objects above *min_confidence* are included first.
    Template boxes are added for classes not already present in the detections
    (requires a ``poker-table`` detection to serve as the layout anchor).
    """
    img_w: int = raw["image"]["width"]
    img_h: int = raw["image"]["height"]
    predictions: list[dict[str, Any]] = raw.get("predictions", [])

    # Filter to confident, in-scope detections
    accepted = [
        p
        for p in predictions
        if p["confidence"] >= min_confidence and p["class"] not in skip_classes
    ]

    # Convert to YOLO format
    rows: list[tuple[int, float, float, float, float]] = []
    detected_classes: set[str] = set()

    for p in accepted:
        cls = p["class"]
        if cls not in class_to_id:
            continue  # class not in our vocabulary — skip
        cx_n, cy_n, w_n, h_n = _to_yolo(
            p["x"], p["y"], p["width"], p["height"], img_w, img_h
        )
        rows.append((class_to_id[cls], cx_n, cy_n, w_n, h_n))
        detected_classes.add(cls)

    # Template pass — add boxes for classes the model misses
    poker_table = next((p for p in predictions if p["class"] == "poker-table"), None)
    if poker_table:
        pt_x0, pt_y0, pt_x1, pt_y1 = _xyxy(poker_table)
        pt_w = pt_x1 - pt_x0
        pt_h = pt_y1 - pt_y0

        for cls_name, tpl in templates.items():
            if cls_name in detected_classes:
                continue  # already detected — don't duplicate
            if cls_name not in class_to_id:
                continue
            cx_abs = pt_x0 + tpl["cx_rel"] * pt_w
            cy_abs = pt_y0 + tpl["cy_rel"] * pt_h
            w_abs = tpl["w_rel"] * pt_w
            h_abs = tpl["h_rel"] * pt_h
            cx_n, cy_n, w_n, h_n = _to_yolo(cx_abs, cy_abs, w_abs, h_abs, img_w, img_h)
            # Clamp to [0, 1]
            cx_n = max(0.0, min(1.0, cx_n))
            cy_n = max(0.0, min(1.0, cy_n))
            w_n = max(0.001, min(1.0, w_n))
            h_n = max(0.001, min(1.0, h_n))
            rows.append((class_to_id[cls_name], cx_n, cy_n, w_n, h_n))

    return rows


# ---------------------------------------------------------------------------
# Preview image
# ---------------------------------------------------------------------------


def draw_preview(
    img_path: Path,
    rows: list[tuple[int, float, float, float, float]],
    id_to_class: dict[int, str],
    out_path: Path,
) -> None:
    """Draw bounding boxes on the screenshot and save to *out_path*."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    iw, ih = img.size

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for class_id, cx_n, cy_n, w_n, h_n in rows:
        cls = id_to_class.get(class_id, str(class_id))
        colour = _PREVIEW_COLOURS.get(cls, _DEFAULT_COLOUR)
        cx, cy, w, h = cx_n * iw, cy_n * ih, w_n * iw, h_n * ih
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=2)
        draw.text((x0 + 2, y0 + 2), cls, fill=colour, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def collect_class_vocabulary(pred_dir: Path, templates: dict[str, Any]) -> list[str]:
    """Return a sorted list of all class names seen in predictions + templates."""
    classes: set[str] = set(templates.keys())
    for pred_file in pred_dir.glob("*.json"):
        try:
            with open(pred_file) as f:
                data = json.load(f)
            for p in data.get("predictions", []):
                classes.add(p["class"])
        except Exception:
            pass
    return sorted(classes)


def write_data_yaml(out_dir: Path, class_names: list[str]) -> None:
    data = {
        "path": str(out_dir.resolve()),
        "train": "images",
        "val": "images",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    with open(out_dir / "data.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def process_screenshot(
    img_path: Path,
    pred_dir: Path,
    class_to_id: dict[str, int],
    id_to_class: dict[int, str],
    templates: dict[str, dict[str, float]],
    min_confidence: float,
    skip_classes: set[str],
    out_images_dir: Path,
    out_labels_dir: Path,
    previews_dir: Path | None,
    use_api: bool,
) -> dict[str, Any]:
    """Process one screenshot. Returns a status dict."""
    stem = img_path.stem
    pred_path = pred_dir / f"{stem}.json"

    raw = load_prediction(pred_path)
    source = "existing"

    if raw is None and use_api:
        try:
            raw = call_roboflow_api(img_path)
            # Cache the result for future runs
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pred_path, "w") as f:
                json.dump(raw, f, indent=2)
            source = "api"
        except Exception as exc:
            return {"stem": stem, "status": "api_error", "detail": str(exc)}

    if raw is None:
        return {"stem": stem, "status": "no_prediction"}

    has_table = any(p["class"] == "poker-table" for p in raw.get("predictions", []))
    rows = build_annotations(raw, class_to_id, templates, min_confidence, skip_classes)

    if not rows:
        return {"stem": stem, "status": "no_annotations", "source": source}

    # Write label file
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    label_path = out_labels_dir / f"{stem}.txt"
    with open(label_path, "w") as f:
        for class_id, cx_n, cy_n, w_n, h_n in rows:
            f.write(f"{class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n")

    # Copy image
    out_images_dir.mkdir(parents=True, exist_ok=True)
    dest_img = out_images_dir / img_path.name
    if not dest_img.exists():
        shutil.copy2(img_path, dest_img)

    # Preview
    if previews_dir is not None:
        draw_preview(img_path, rows, id_to_class, previews_dir / f"{stem}.jpg")

    template_classes = [
        id_to_class[r[0]] for r in rows if id_to_class.get(r[0]) in templates
    ]
    return {
        "stem": stem,
        "status": "ok",
        "source": source,
        "n_boxes": len(rows),
        "has_table": has_table,
        "template_classes": template_classes,
    }


def main() -> None:
    args = sys.argv[1:]
    use_api = "--api" in args
    do_preview = "--preview" in args
    preview_only = "--preview-only" in args

    # --preview-only path/to/screenshot.png
    if preview_only:
        img_args = [a for a in args if not a.startswith("--")]
        if not img_args:
            print("Usage: auto_annotate.py --preview-only path/to/screenshot.png")
            sys.exit(1)
        img_path = Path(img_args[0])
        cfg = load_config()
        pred_dir = (_HERE / cfg["predictions_dir"]).resolve()
        templates: dict[str, dict[str, float]] = cfg.get("templates", {})
        class_names = collect_class_vocabulary(pred_dir, templates)
        class_to_id = {c: i for i, c in enumerate(class_names)}
        id_to_class = {i: c for c, i in class_to_id.items()}
        min_confidence = float(cfg.get("min_confidence", 0.70))
        skip_classes: set[str] = set(cfg.get("skip_classes", []))

        stem = img_path.stem
        pred_path = pred_dir / f"{stem}.json"
        raw = load_prediction(pred_path)
        if raw is None and use_api:
            raw = call_roboflow_api(img_path)
        if raw is None:
            print(f"No prediction found for {img_path.name}")
            sys.exit(1)

        rows = build_annotations(
            raw, class_to_id, templates, min_confidence, skip_classes
        )
        out_path = img_path.parent / f"{stem}_preview.jpg"
        draw_preview(img_path, rows, id_to_class, out_path)
        print(f"Preview saved: {out_path}  ({len(rows)} boxes)")
        for r in rows:
            print(
                f"  {id_to_class[r[0]]}  cx={r[1]:.3f} cy={r[2]:.3f} w={r[3]:.3f} h={r[4]:.3f}"
            )
        return

    # Full batch run
    cfg = load_config()
    screenshot_dir = (_HERE / cfg["screenshot_dir"]).resolve()
    pred_dir = (_HERE / cfg["predictions_dir"]).resolve()
    out_dir = (_HERE / cfg["output_dir"]).resolve()
    previews_dir = (_HERE / cfg["previews_dir"]).resolve() if do_preview else None
    templates = cfg.get("templates", {})
    min_confidence = float(cfg.get("min_confidence", 0.70))
    skip_classes = set(cfg.get("skip_classes", []))

    # Build class vocabulary from all available predictions + template classes
    print("Building class vocabulary...")
    class_names = collect_class_vocabulary(pred_dir, templates)
    class_to_id = {c: i for i, c in enumerate(class_names)}
    id_to_class = {i: c for c, i in class_to_id.items()}
    print(f"  {len(class_names)} classes: {', '.join(class_names)}")

    out_images_dir = out_dir / "images"
    out_labels_dir = out_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write data.yaml
    write_data_yaml(out_dir, class_names)

    # Process each screenshot
    screenshots = sorted(screenshot_dir.glob("*.png")) + sorted(
        screenshot_dir.glob("*.jpg")
    )
    print(f"\nProcessing {len(screenshots)} screenshots...")

    stats: dict[str, int] = {
        "ok": 0,
        "no_prediction": 0,
        "no_annotations": 0,
        "api_error": 0,
    }
    template_hits = 0

    for img_path in screenshots:
        result = process_screenshot(
            img_path=img_path,
            pred_dir=pred_dir,
            class_to_id=class_to_id,
            id_to_class=id_to_class,
            templates=templates,
            min_confidence=min_confidence,
            skip_classes=skip_classes,
            out_images_dir=out_images_dir,
            out_labels_dir=out_labels_dir,
            previews_dir=previews_dir,
            use_api=use_api,
        )
        status = result["status"]
        stats[status] = stats.get(status, 0) + 1
        if result.get("template_classes"):
            template_hits += 1
        if status != "ok":
            print(f"  SKIP {result['stem']}: {status}")

    print("\n=== Summary ===")
    print(f"  Annotated:          {stats.get('ok', 0)}")
    print(f"  Template boxes used: {template_hits}")
    print(
        f"  No prediction file:  {stats.get('no_prediction', 0)}"
        + (
            "  (run with --api to fetch)"
            if stats.get("no_prediction", 0) and not use_api
            else ""
        )
    )
    print(f"  No annotations:      {stats.get('no_annotations', 0)}")
    if stats.get("api_error", 0):
        print(f"  API errors:          {stats['api_error']}")
    print(f"\nDataset written to: {out_dir}")
    print(f"  data.yaml: {out_dir / 'data.yaml'}")
    print(f"  images/:   {out_images_dir}")
    print(f"  labels/:   {out_labels_dir}")
    if previews_dir:
        print(f"  previews/: {previews_dir}")
    print("\nTo upload to Roboflow, zip the dataset folder:")
    print(f"  Compress-Archive -Path '{out_dir}' -DestinationPath dataset.zip")


if __name__ == "__main__":
    main()
