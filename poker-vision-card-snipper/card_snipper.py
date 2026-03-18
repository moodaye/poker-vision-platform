"""
card_snipper.py
---------------
Core module for snipping card images from a poker-table screenshot.

Usage
-----
from PIL import Image
from card_snipper import snip_flop_cards

img = Image.open("screenshot.png")
cards = snip_flop_cards(img, detections_list)
for card in cards:
    card.image.save(f"card_{card.index:02d}.png")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass
class CardSnip:
    """A single snipped card image with its associated metadata."""

    index: int
    confidence: float
    bbox_xyxy: list[int]  # [x1, y1, x2, y2]
    image: Image.Image


def snip_flop_cards(
    image: Image.Image,
    detections: list[dict[str, Any]],
    *,
    target_class: str = "flop_card",
) -> list[CardSnip]:
    """
    Extract detected card regions from a poker-table screenshot.

    Parameters
    ----------
    image:
        A PIL Image of the poker-table screenshot.
    detections:
        List of detection dicts. Each dict must contain at minimum:
          - ``class_name``  (str)
          - ``bbox_xyxy``   ([x1, y1, x2, y2] in pixel coordinates)
          - ``confidence``  (float, optional)
    target_class:
        The class name to snip. Defaults to ``"flop_card"``.

    Returns
    -------
    list[CardSnip]
        Card snips sorted left-to-right by their x1 coordinate.
    """
    flop_detections = [d for d in detections if d.get("class_name") == target_class]
    flop_detections.sort(key=lambda d: d["bbox_xyxy"][0])

    results: list[CardSnip] = []
    for i, det in enumerate(flop_detections):
        x1, y1, x2, y2 = det["bbox_xyxy"]

        # Clamp to image bounds
        width, height = image.size
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        card_img = image.crop((x1, y1, x2, y2))
        results.append(
            CardSnip(
                index=i,
                confidence=det.get("confidence", 0.0),
                bbox_xyxy=[x1, y1, x2, y2],
                image=card_img,
            )
        )

    return results
