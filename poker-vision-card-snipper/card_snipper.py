"""
card_snipper.py
---------------
Core module for snipping object regions from a screenshot using detector bounding boxes.

Usage
-----
from PIL import Image
from card_snipper import snip_objects

img = Image.open("screenshot.png")
snips = snip_objects(img, detections_list, target_classes=["flop_card", "holecard"])
for snip in snips:
    snip.image.save(f"{snip.class_name}_{snip.index:02d}.png")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass
class ObjectSnip:
    """A single snipped object image with its associated metadata."""

    class_name: str
    index: int
    confidence: float
    bbox_xyxy: list[int]  # [x1, y1, x2, y2]
    image: Image.Image


# Backward-compatible alias
CardSnip = ObjectSnip


def snip_objects(
    image: Image.Image,
    detections: list[dict[str, Any]],
    *,
    target_classes: list[str] | None = None,
) -> list[ObjectSnip]:
    """
    Extract detected object regions from a screenshot.

    Parameters
    ----------
    image:
        A PIL Image of the screenshot.
    detections:
        List of detection dicts. Each dict must contain at minimum:
          - ``class_name``  (str)
          - ``bbox_xyxy``   ([x1, y1, x2, y2] in pixel coordinates)
          - ``confidence``  (float, optional)
    target_classes:
        List of class names to snip. If ``None``, all detected classes are snipped.

    Returns
    -------
    list[ObjectSnip]
        Snips sorted by class_name, then left-to-right by x1 coordinate.
    """
    if target_classes is not None:
        filtered = [d for d in detections if d.get("class_name") in target_classes]
    else:
        filtered = list(detections)

    filtered.sort(key=lambda d: (d.get("class_name", ""), d["bbox_xyxy"][0]))

    results: list[ObjectSnip] = []
    for i, det in enumerate(filtered):
        x1, y1, x2, y2 = det["bbox_xyxy"]

        # Clamp to image bounds
        width, height = image.size
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        obj_img = image.crop((x1, y1, x2, y2))
        results.append(
            ObjectSnip(
                class_name=det.get("class_name", "unknown"),
                index=i,
                confidence=det.get("confidence", 0.0),
                bbox_xyxy=[x1, y1, x2, y2],
                image=obj_img,
            )
        )

    return results


def snip_flop_cards(
    image: Image.Image,
    detections: list[dict[str, Any]],
    *,
    target_class: str = "flop_card",
) -> list[ObjectSnip]:
    """
    Backward-compatible wrapper: snip a single target class, sorted left-to-right.

    Prefer ``snip_objects()`` for new code.
    """
    return snip_objects(image, detections, target_classes=[target_class])
