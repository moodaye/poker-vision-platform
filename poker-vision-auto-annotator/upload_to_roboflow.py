"""upload_to_roboflow.py — upload the auto-annotated YOLO dataset to Roboflow.

Uploads each image then immediately uploads its YOLO annotation.
Uses the Roboflow REST API directly (no SDK required).

Usage:
    uv run python upload_to_roboflow.py [--dry-run]

    --dry-run  Print what would be uploaded without making any API calls.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

_HERE = Path(__file__).parent
load_dotenv(dotenv_path=_HERE / "../poker-vision-object-detector" / ".env")

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
PROJECT = "pokertabledetection"

IMAGES_DIR = _HERE / "output/yolo_dataset/images"
LABELS_DIR = _HERE / "output/yolo_dataset/labels"
DATA_YAML = _HERE / "output/yolo_dataset/data.yaml"

# Delay between uploads to avoid rate-limiting (seconds)
_UPLOAD_DELAY = 0.2


def load_labelmap() -> dict[str, str]:
    """Return {str(class_id): class_name} from data.yaml.

    Handles both the YOLO standard list format (names: [cls0, cls1, ...])
    and the legacy integer-keyed dict format (names: {0: cls0, 1: cls1, ...}).
    """
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, list):
        return {str(i): name for i, name in enumerate(names)}
    return {str(k): v for k, v in names.items()}


def upload_image(img_path: Path) -> str | None:
    """Upload image to Roboflow. Returns image_id, or None on failure."""
    import base64

    img_bytes = img_path.read_bytes()
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    resp = requests.post(
        f"https://api.roboflow.com/dataset/{PROJECT}/upload",
        params={
            "api_key": ROBOFLOW_API_KEY,
            "name": img_path.name,
            "split": "train",
        },
        data=b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    if not data.get("success") and not data.get("duplicate"):
        print(f"\n    API response: {data}")
        return None
    return data.get("id")


def upload_annotation(
    image_id: str, label_path: Path, labelmap: dict[str, str]
) -> bool:
    """Upload YOLO annotation for an already-uploaded image."""
    annotation_content = label_path.read_text()

    resp = requests.post(
        f"https://api.roboflow.com/dataset/{PROJECT}/annotate/{image_id}",
        params={
            "api_key": ROBOFLOW_API_KEY,
            "name": label_path.name,
            "nocache": "true",
            "labelmap": json.dumps(labelmap),
        },
        data=annotation_content,
        headers={"Content-Type": "text/plain"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("success", False)


def main() -> None:
    args = sys.argv[1:]
    dry_run = "--dry-run" in args

    if not ROBOFLOW_API_KEY:
        print("ERROR: ROBOFLOW_API_KEY not set in poker-vision-object-detector/.env")
        sys.exit(1)

    labelmap = load_labelmap()
    images = sorted(IMAGES_DIR.glob("*.png")) + sorted(IMAGES_DIR.glob("*.jpg"))

    if not images:
        print(f"No images found in {IMAGES_DIR}")
        sys.exit(1)

    print(
        f"{'[DRY RUN] ' if dry_run else ''}Uploading {len(images)} images to {PROJECT}"
    )
    print(f"Classes: {', '.join(labelmap.values())}\n")

    ok = 0
    no_label = 0
    failed = 0

    for i, img_path in enumerate(images, 1):
        label_path = LABELS_DIR / f"{img_path.stem}.txt"

        if not label_path.exists():
            print(f"  [{i:3}/{len(images)}] SKIP (no label): {img_path.name}")
            no_label += 1
            continue

        if dry_run:
            n_boxes = len(label_path.read_text().strip().splitlines())
            print(
                f"  [{i:3}/{len(images)}] WOULD UPLOAD: {img_path.name}  ({n_boxes} boxes)"
            )
            ok += 1
            continue

        print(f"  [{i:3}/{len(images)}] {img_path.name} ...", end=" ", flush=True)
        try:
            image_id = upload_image(img_path)
            if image_id is None:
                print("image upload failed")
                failed += 1
                continue

            ann_ok = upload_annotation(image_id, label_path, labelmap)
            if ann_ok:
                print("ok")
                ok += 1
            else:
                print("annotation failed")
                failed += 1

        except requests.HTTPError as exc:
            print(f"HTTP {exc.response.status_code}: {exc.response.text[:120]}")
            failed += 1
        except Exception as exc:
            print(f"ERROR: {exc}")
            failed += 1

        time.sleep(_UPLOAD_DELAY)

    print("\n=== Summary ===")
    print(f"  Uploaded:   {ok}")
    print(f"  No label:   {no_label}")
    print(f"  Failed:     {failed}")
    if not dry_run:
        print("\nOpen Roboflow and generate a new dataset version to retrain.")


if __name__ == "__main__":
    main()
