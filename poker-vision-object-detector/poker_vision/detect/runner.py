"""
Runner: directory scanning, per-image processing, output writing, run summary.
"""

from __future__ import annotations

import datetime
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .client import RoboflowAPIError, RoboflowClient
from .config import DetectConfig, save_resolved_config
from .normalize import normalize_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------


def collect_images(
    input_dir: Path, extensions: list[str], recursive: bool
) -> list[Path]:
    """Return a sorted list of image paths under *input_dir*."""
    exts = {e.lower() for e in extensions}
    if recursive:
        candidates = input_dir.rglob("*")
    else:
        candidates = input_dir.glob("*")
    paths = [p for p in candidates if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths)


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------


def _get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Return (width, height) of the image using Pillow."""
    with Image.open(image_path) as img:
        return img.width, img.height


def process_image(
    image_path: Path,
    client: RoboflowClient,
    cfg: DetectConfig,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """
    Run inference on a single image.

    Returns (raw_response, normalized, failure_record).
    On success: failure_record is None.
    On failure: raw_response and normalized are None.
    """
    try:
        raw = client.predict(image_path)
    except RoboflowAPIError as exc:
        return (
            None,
            None,
            {
                "source_image": str(image_path),
                "message": str(exc),
                "http_status": exc.http_status,
                "exception_type": type(exc).__name__,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return (
            None,
            None,
            {
                "source_image": str(image_path),
                "message": str(exc),
                "exception_type": type(exc).__name__,
            },
        )

    try:
        width, height = _get_image_dimensions(image_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read image dimensions for %s: %s", image_path, exc)
        width, height = 0, 0

    rf = cfg.roboflow
    normalized = normalize_response(
        raw_response=raw,
        source_image=str(image_path),
        image_width=width,
        image_height=height,
        api_base=rf.api_base,
        project=rf.project,
        version=rf.version,
        confidence_threshold=rf.confidence_threshold,
        overlap_threshold=rf.overlap_threshold,
    )

    return raw, normalized, None


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _stem(image_path: Path) -> str:
    return image_path.stem


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run(cfg: DetectConfig, api_key: str) -> int:
    """
    Execute the full detection pipeline.

    Returns 0 on success (partial failures ok), 1 if all images failed.
    """
    started_at = datetime.datetime.now(tz=datetime.timezone.utc)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    save_resolved_config(cfg, output_dir)

    # Collect images
    input_dir = Path(cfg.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

    image_paths = collect_images(
        input_dir, cfg.io.image_extensions, cfg.io.recursive_input
    )
    logger.info("Found %d image(s) in %s", len(image_paths), input_dir)

    if not image_paths:
        logger.warning("No images found in %s", input_dir)

    client = RoboflowClient(cfg.roboflow, api_key)

    raw_dir = output_dir / "predictions_raw"
    norm_dir = output_dir / "detections_normalized"

    failures: list[dict[str, Any]] = []
    detections_per_class: dict[str, int] = defaultdict(int)
    total_detections = 0
    succeeded = 0

    for image_path in image_paths:
        logger.info("Processing %s", image_path)
        raw, normalized, failure = process_image(image_path, client, cfg)

        if failure is not None:
            logger.error("Failed to process %s: %s", image_path, failure["message"])
            failures.append(failure)
            continue

        succeeded += 1
        stem = _stem(image_path)

        if cfg.run.save_raw_predictions:
            _write_json(raw_dir / f"{stem}.json", raw)

        if cfg.run.save_normalized_detections and normalized is not None:
            _write_json(norm_dir / f"{stem}.detections.json", normalized)
            for det in normalized.get("detections", []):
                cls = det["class_name"]
                detections_per_class[cls] += 1
                total_detections += 1

    finished_at = datetime.datetime.now(tz=datetime.timezone.utc)

    summary = {
        "run_id": cfg.run.run_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "images_total": len(image_paths),
        "images_succeeded": succeeded,
        "images_failed": len(failures),
        "detections_total": total_detections,
        "detections_per_class": dict(detections_per_class),
        "failures": failures,
    }

    if cfg.run.save_run_summary:
        _write_json(output_dir / "run_summary.json", summary)

    logger.info(
        "Run complete: %d/%d succeeded, %d detection(s), %d failure(s)",
        succeeded,
        len(image_paths),
        total_detections,
        len(failures),
    )

    # Exit code: non-zero only if all images failed (and there were images)
    if image_paths and succeeded == 0:
        return 1
    return 0
