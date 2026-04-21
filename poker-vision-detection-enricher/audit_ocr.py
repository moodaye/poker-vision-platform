"""
OCR audit script — crops OCR-routed objects from real screenshots and runs
run_ocr on each, saving the crop image alongside the OCR result.

Usage:
    uv run python poker-vision-detection-enricher/audit_ocr.py

Output is written to poker-vision-detection-enricher/snips/ocr_audit/
Each crop is saved as:
    <capture_id>__<class_name>_<index>.png
A summary table is printed to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Paths — adjust if your archive / detections live elsewhere
# ---------------------------------------------------------------------------
ARCHIVE_DIR = Path("poker-vision-screenshot-archive")
DETECTIONS_DIR = Path("poker-vision-object-detector/output/detections_normalized")
OUTPUT_DIR = Path("poker-vision-detection-enricher/snips/ocr_audit")

OCR_CLASSES = {"chip_stack", "pot", "total_pot", "blinds", "player_name"}

# Max captures to process (set to None to run all)
MAX_CAPTURES = 10


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Deferred import so the model only loads when the script actually runs
    from ocr_module import run_ocr  # noqa: PLC0415

    detection_files = sorted(DETECTIONS_DIR.glob("*.detections.json"))
    if not detection_files:
        print(f"No detection files found in {DETECTIONS_DIR}", file=sys.stderr)
        sys.exit(1)

    if MAX_CAPTURES:
        detection_files = detection_files[:MAX_CAPTURES]

    rows: list[dict] = []

    for det_path in detection_files:
        data = json.loads(det_path.read_text())
        detections = data.get("detections", [])

        ocr_dets = [d for d in detections if d.get("class_name") in OCR_CLASSES]
        if not ocr_dets:
            continue

        capture_id = det_path.stem.replace(".detections", "")
        img_path = ARCHIVE_DIR / f"{capture_id}.png"
        if not img_path.exists():
            print(f"  [SKIP] screenshot not found: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        print(f"\n{capture_id}  ({len(ocr_dets)} OCR objects)")

        class_counts: dict[str, int] = {}
        for det in ocr_dets:
            cls = det["class_name"]
            idx = class_counts.get(cls, 0)
            class_counts[cls] = idx + 1

            bbox = det.get("bbox_xyxy") or det.get("bbox")
            if not bbox or len(bbox) != 4:
                print(f"  [SKIP] {cls} — no valid bbox")
                continue

            crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            ocr_text = run_ocr(crop)

            crop_filename = f"{capture_id}__{cls}_{idx}.png"
            crop.save(OUTPUT_DIR / crop_filename)

            status = "OK" if ocr_text else "EMPTY"
            print(f"  [{status}] {cls:15s}  bbox={bbox}  → {ocr_text!r}")
            rows.append(
                {
                    "capture": capture_id,
                    "class": cls,
                    "bbox": bbox,
                    "ocr_text": ocr_text,
                    "crop_file": crop_filename,
                }
            )

    # Summary
    total = len(rows)
    empty = sum(1 for r in rows if not r["ocr_text"])
    print(f"\n{'=' * 60}")
    print(f"Total OCR objects: {total}")
    print(f"  Non-empty:       {total - empty}")
    print(f"  Empty:           {empty}")
    print(f"Crops saved to:    {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
