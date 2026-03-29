"""
batch.py
--------
Batch-process a folder of screenshots through the snipper.

Matches each .detections.json file in the detections folder to an image in the
images folder by stem name, then saves snipped object images to the output folder.

Usage
-----
python batch.py \\
    --detections ../poker-vision-object-detector/output/detections_normalized \\
    --images     ../poker-vision-object-detector/screenshots \\
    --output     ./output \\
    --target-classes flop_card holecard chip_stack

Output layout
-------------
<output>/
    <image_stem>/
        flop_card_00.png
        holecard_00.png
        ...
    summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from card_snipper import snip_objects
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def process_batch(
    detections_dir: Path,
    images_dir: Path,
    output_dir: Path,
    target_classes: list[str] | None = None,
) -> list[dict[str, Any]]:
    detection_files = sorted(detections_dir.glob("*.detections.json"))
    if not detection_files:
        print(f"No .detections.json files found in: {detections_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []

    for det_path in detection_files:
        # Stem is everything before the first '.' (e.g. "capture_20260219_175731_401677")
        stem = det_path.name.split(".")[0]

        image_path = find_image(images_dir, stem)
        if image_path is None:
            print(f"  [SKIP] No image found for {det_path.name}", file=sys.stderr)
            summary.append(
                {
                    "detections_file": det_path.name,
                    "status": "skipped",
                    "reason": "image not found",
                }
            )
            continue

        try:
            image = Image.open(image_path)
        except UnidentifiedImageError as exc:
            print(
                f"  [SKIP] Cannot open image {image_path.name}: {exc}", file=sys.stderr
            )
            summary.append(
                {
                    "detections_file": det_path.name,
                    "status": "skipped",
                    "reason": str(exc),
                }
            )
            continue

        payload = json.loads(det_path.read_text(encoding="utf-8"))
        detections = (
            payload.get("detections", payload) if isinstance(payload, dict) else payload
        )

        cards = snip_objects(image, detections, target_classes=target_classes)

        card_dir = output_dir / stem
        card_dir.mkdir(parents=True, exist_ok=True)

        saved: list[dict[str, Any]] = []
        for card in cards:
            card_filename = f"{card.class_name}_{card.index:02d}.png"
            card.image.save(card_dir / card_filename)
            saved.append(
                {
                    "filename": card_filename,
                    "confidence": card.confidence,
                    "bbox_xyxy": card.bbox_xyxy,
                }
            )

        print(f"  [OK] {stem}: {len(cards)} card(s) saved → {card_dir}")
        summary.append(
            {
                "detections_file": det_path.name,
                "image_file": image_path.name,
                "status": "ok",
                "card_count": len(cards),
                "output_dir": str(card_dir),
                "cards": saved,
            }
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch snip flop cards from poker screenshots using object-detector output."
    )
    parser.add_argument(
        "--target-classes",
        nargs="*",
        metavar="CLASS",
        help="Class names to snip (e.g. flop_card holecard chip_stack). If omitted, all classes are snipped.",
    )
    parser.add_argument(
        "--detections",
        required=True,
        type=Path,
        help="Folder containing *.detections.json files produced by the object detector.",
    )
    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="Folder containing the original screenshot images.",
    )
    parser.add_argument(
        "--output",
        default=Path("output"),
        type=Path,
        help="Folder where snipped card images will be saved (default: ./output).",
    )
    args = parser.parse_args()

    if not args.detections.is_dir():
        parser.error(
            f"--detections path does not exist or is not a directory: {args.detections}"
        )
    if not args.images.is_dir():
        parser.error(
            f"--images path does not exist or is not a directory: {args.images}"
        )

    print(f"Detections : {args.detections}")
    print(f"Images     : {args.images}")
    print(f"Output     : {args.output}")
    if args.target_classes:
        print(f"Classes    : {', '.join(args.target_classes)}")
    else:
        print("Classes    : all")
    print()

    summary = process_batch(
        args.detections, args.images, args.output, args.target_classes or None
    )

    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {summary_path}")

    ok = sum(1 for s in summary if s["status"] == "ok")
    skipped = len(summary) - ok
    print(f"Done: {ok} processed, {skipped} skipped.")


if __name__ == "__main__":
    main()
