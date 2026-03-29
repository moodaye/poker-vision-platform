"""
Tests for snip_objects() and backward-compat snip_flop_cards().
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PIL import Image

# Ensure the package root is on the path when running from the workspace root
sys.path.insert(0, str(Path(__file__).parent.parent))

from card_snipper import CardSnip, ObjectSnip, snip_flop_cards, snip_objects

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(width: int = 200, height: int = 150) -> Image.Image:
    return Image.new("RGB", (width, height), color=(30, 30, 30))


def _det(
    class_name: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    confidence: float = 0.9,
) -> dict[str, Any]:
    return {
        "class_name": class_name,
        "bbox_xyxy": [x1, y1, x2, y2],
        "confidence": confidence,
    }


DETECTIONS: list[dict[str, Any]] = [
    _det("flop_card", 10, 10, 40, 60, confidence=0.95),
    _det("flop_card", 50, 10, 80, 60, confidence=0.90),
    _det("holecard", 100, 10, 130, 60, confidence=0.88),
    _det("chip_stack", 5, 80, 35, 110, confidence=0.75),
    _det("pot", 70, 80, 100, 110, confidence=0.70),
]


# ---------------------------------------------------------------------------
# snip_objects — target_classes filtering
# ---------------------------------------------------------------------------


def test_snip_objects_all_classes_when_none() -> None:
    img = _make_image()
    result = snip_objects(img, DETECTIONS, target_classes=None)
    assert len(result) == 5


def test_snip_objects_single_class() -> None:
    img = _make_image()
    result = snip_objects(img, DETECTIONS, target_classes=["flop_card"])
    assert len(result) == 2
    assert all(s.class_name == "flop_card" for s in result)


def test_snip_objects_multiple_classes() -> None:
    img = _make_image()
    result = snip_objects(img, DETECTIONS, target_classes=["flop_card", "holecard"])
    assert len(result) == 3
    class_names = [s.class_name for s in result]
    assert "flop_card" in class_names
    assert "holecard" in class_names
    assert "chip_stack" not in class_names


def test_snip_objects_unknown_class_returns_empty() -> None:
    img = _make_image()
    result = snip_objects(img, DETECTIONS, target_classes=["dealer_button"])
    assert result == []


def test_snip_objects_empty_detections() -> None:
    img = _make_image()
    result = snip_objects(img, [], target_classes=None)
    assert result == []


# ---------------------------------------------------------------------------
# snip_objects — sort order
# ---------------------------------------------------------------------------


def test_snip_objects_sorted_by_class_then_x1() -> None:
    img = _make_image()
    result = snip_objects(img, DETECTIONS, target_classes=None)
    keys = [(s.class_name, s.bbox_xyxy[0]) for s in result]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# ObjectSnip schema
# ---------------------------------------------------------------------------


def test_object_snip_has_class_name() -> None:
    img = _make_image()
    result = snip_objects(img, DETECTIONS, target_classes=["holecard"])
    assert len(result) == 1
    s = result[0]
    assert s.class_name == "holecard"
    assert isinstance(s.index, int)
    assert isinstance(s.confidence, float)
    assert len(s.bbox_xyxy) == 4
    assert isinstance(s.image, Image.Image)


def test_snip_crops_correct_dimensions() -> None:
    img = _make_image()
    result = snip_objects(
        img, [_det("flop_card", 10, 20, 50, 70)], target_classes=["flop_card"]
    )
    assert len(result) == 1
    crop = result[0].image
    assert crop.size == (40, 50)  # width=50-10, height=70-20


def test_snip_clamps_bbox_to_image_bounds() -> None:
    img = _make_image(width=100, height=100)
    # bbox deliberately extends beyond image bounds
    result = snip_objects(img, [_det("pot", 80, 80, 150, 150)], target_classes=["pot"])
    assert len(result) == 1
    x1, y1, x2, y2 = result[0].bbox_xyxy
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 100 and y2 <= 100


# ---------------------------------------------------------------------------
# Backward-compat aliases
# ---------------------------------------------------------------------------


def test_card_snip_is_alias_for_object_snip() -> None:
    assert CardSnip is ObjectSnip


def test_snip_flop_cards_returns_only_flop_cards() -> None:
    img = _make_image()
    result = snip_flop_cards(img, DETECTIONS)
    assert len(result) == 2
    assert all(s.class_name == "flop_card" for s in result)


def test_snip_flop_cards_target_class_override() -> None:
    img = _make_image()
    result = snip_flop_cards(img, DETECTIONS, target_class="holecard")
    assert len(result) == 1
    assert result[0].class_name == "holecard"


def test_snip_flop_cards_sorted_left_to_right() -> None:
    img = _make_image()
    result = snip_flop_cards(img, DETECTIONS)
    x1_values = [s.bbox_xyxy[0] for s in result]
    assert x1_values == sorted(x1_values)
