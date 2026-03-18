"""
Bbox conversion and prediction normalization.

Roboflow returns predictions in center-xywh format. This module converts
them to xyxy and builds the normalized detection schema.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Bbox arithmetic
# ---------------------------------------------------------------------------


def center_xywh_to_xyxy(x: float, y: float, w: float, h: float) -> list[int]:
    """
    Convert Roboflow center-xywh bbox to corner xyxy (all values rounded to int).

    Parameters
    ----------
    x, y : float
        Center coordinates.
    w, h : float
        Width and height.

    Returns
    -------
    [x1, y1, x2, y2] as ints.
    """
    x1 = round(x - w / 2)
    y1 = round(y - h / 2)
    x2 = round(x + w / 2)
    y2 = round(y + h / 2)
    return [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_response(
    raw_response: dict[str, Any],
    source_image: str,
    image_width: int,
    image_height: int,
    api_base: str,
    project: str,
    version: Any,
    confidence_threshold: float,
    overlap_threshold: float,
) -> dict[str, Any]:
    """
    Build the normalized detection output dict from a raw Roboflow API response.

    Filters out predictions below *confidence_threshold* (client-side safety net).
    """
    predictions = raw_response.get("predictions", [])

    detections: list[dict[str, Any]] = []
    for pred in predictions:
        conf: float = float(pred.get("confidence", 0.0))
        if conf < confidence_threshold:
            continue  # client-side confidence filter

        x = float(pred.get("x", 0.0))
        y = float(pred.get("y", 0.0))
        w = float(pred.get("width", 0.0))
        h = float(pred.get("height", 0.0))

        bbox_xywh: list[float] = [round(x), round(y), round(w), round(h)]
        bbox_xyxy: list[int] = center_xywh_to_xyxy(x, y, w, h)

        class_id: int | None = pred.get("class_id")
        if class_id is not None:
            try:
                class_id = int(class_id)
            except (TypeError, ValueError):
                class_id = None

        detections.append(
            {
                "class_name": str(pred.get("class", "")),
                "class_id": class_id,
                "confidence": round(conf, 6),
                "bbox_xywh_center": bbox_xywh,
                "bbox_xyxy": bbox_xyxy,
            }
        )

    return {
        "source_image": source_image,
        "image_width": image_width,
        "image_height": image_height,
        "model": {
            "provider": "roboflow",
            "api_base": api_base,
            "project": project,
            "version": str(version),
            "requested_confidence_threshold": confidence_threshold,
            "requested_overlap_threshold": overlap_threshold,
        },
        "detections": detections,
    }
