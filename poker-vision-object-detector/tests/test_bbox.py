"""
Unit tests for bbox conversion: center xywh -> xyxy.
"""

import pytest
from poker_vision.detect.normalize import center_xywh_to_xyxy

# (x_center, y_center, width, height) -> expected [x1, y1, x2, y2]
BBOX_CASES = [
    # Simple integer-aligned box
    (100.0, 200.0, 40.0, 60.0, [80, 170, 120, 230]),
    # Zero-size box
    (50.0, 50.0, 0.0, 0.0, [50, 50, 50, 50]),
    # Fractional center — Python round() uses banker's rounding (round half to even)
    (10.5, 20.5, 10.0, 20.0, [6, 10, 16, 30]),  # round(10.5)=10, round(30.5)=30
    # Large coordinates
    (1920.0, 1080.0, 200.0, 100.0, [1820, 1030, 2020, 1130]),
    # Odd dimension — banker's rounding: round(44.5)=44, round(55.5)=56
    (50.0, 50.0, 11.0, 11.0, [44, 44, 56, 56]),
    # Sub-pixel center
    (0.4, 0.4, 0.8, 0.8, [0, 0, 1, 1]),
]


@pytest.mark.parametrize("x,y,w,h,expected", BBOX_CASES)
def test_center_xywh_to_xyxy(x, y, w, h, expected):
    result = center_xywh_to_xyxy(x, y, w, h)
    assert result == expected, (
        f"center_xywh_to_xyxy({x}, {y}, {w}, {h}) = {result}, expected {expected}"
    )


def test_returns_list_of_four_ints():
    result = center_xywh_to_xyxy(50.0, 50.0, 20.0, 30.0)
    assert isinstance(result, list)
    assert len(result) == 4
    for val in result:
        assert isinstance(val, int)


def test_xyxy_ordering():
    """x1 <= x2 and y1 <= y2 for positive dimensions."""
    result = center_xywh_to_xyxy(100.0, 100.0, 50.0, 80.0)
    x1, y1, x2, y2 = result
    assert x1 <= x2
    assert y1 <= y2
