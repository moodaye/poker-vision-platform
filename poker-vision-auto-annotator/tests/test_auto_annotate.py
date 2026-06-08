"""Tests for auto_annotate.py — annotation generation logic."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# Make the parent directory importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from auto_annotate import _to_yolo, build_annotations, write_data_yaml

# ---------------------------------------------------------------------------
# Fixtures — minimal Roboflow prediction payloads
# ---------------------------------------------------------------------------


def _raw(
    predictions: list[dict[str, float | str]],
    img_w: int = 1920,
    img_h: int = 1080,
) -> dict[str, object]:
    """Build a minimal Roboflow raw-prediction dict."""
    return {"image": {"width": img_w, "height": img_h}, "predictions": predictions}


def _pred(
    cls: str,
    x: float,
    y: float,
    w: float,
    h: float,
    conf: float = 0.95,
) -> dict[str, float | str]:
    """Build a single Roboflow prediction entry."""
    return {"class": cls, "x": x, "y": y, "width": w, "height": h, "confidence": conf}


CLASS_TO_ID = {
    "poker-table": 0,
    "player_me": 1,
    "player_other": 2,
    "chip_stack": 3,
    "holecard": 4,
}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

# A poker-table prediction used as the layout anchor for templates
_TABLE = _pred("poker-table", x=960, y=540, w=1600, h=900)  # fills most of 1920x1080


# ---------------------------------------------------------------------------
# _to_yolo — coordinate normalisation
# ---------------------------------------------------------------------------


class TestToYolo:
    def test_centre_image(self) -> None:
        cx_n, cy_n, w_n, h_n = _to_yolo(960, 540, 1920, 1080, 1920, 1080)
        assert cx_n == pytest.approx(0.5)
        assert cy_n == pytest.approx(0.5)
        assert w_n == pytest.approx(1.0)
        assert h_n == pytest.approx(1.0)

    def test_normalised_values_in_range(self) -> None:
        cx_n, cy_n, w_n, h_n = _to_yolo(100, 200, 50, 80, 1920, 1080)
        assert 0.0 <= cx_n <= 1.0
        assert 0.0 <= cy_n <= 1.0
        assert 0.0 < w_n <= 1.0
        assert 0.0 < h_n <= 1.0


# ---------------------------------------------------------------------------
# build_annotations — confidence filtering
# ---------------------------------------------------------------------------


class TestConfidenceFiltering:
    def test_detection_above_threshold_included(self) -> None:
        raw = _raw([_pred("holecard", 200, 200, 50, 80, conf=0.95)])
        rows = build_annotations(
            raw, CLASS_TO_ID, {}, min_confidence=0.70, skip_classes=set()
        )
        assert any(r[0] == CLASS_TO_ID["holecard"] for r in rows)

    def test_detection_below_threshold_excluded(self) -> None:
        raw = _raw([_pred("holecard", 200, 200, 50, 80, conf=0.50)])
        rows = build_annotations(
            raw, CLASS_TO_ID, {}, min_confidence=0.70, skip_classes=set()
        )
        assert not any(r[0] == CLASS_TO_ID["holecard"] for r in rows)

    def test_detection_at_exact_threshold_included(self) -> None:
        raw = _raw([_pred("holecard", 200, 200, 50, 80, conf=0.70)])
        rows = build_annotations(
            raw, CLASS_TO_ID, {}, min_confidence=0.70, skip_classes=set()
        )
        assert any(r[0] == CLASS_TO_ID["holecard"] for r in rows)

    def test_skip_classes_excluded(self) -> None:
        raw = _raw([_pred("chip_stack", 200, 200, 50, 50, conf=0.95)])
        rows = build_annotations(
            raw, CLASS_TO_ID, {}, min_confidence=0.70, skip_classes={"chip_stack"}
        )
        assert not any(r[0] == CLASS_TO_ID["chip_stack"] for r in rows)

    def test_unknown_class_excluded(self) -> None:
        raw = _raw([_pred("unknown_object", 200, 200, 50, 50, conf=0.99)])
        rows = build_annotations(
            raw, CLASS_TO_ID, {}, min_confidence=0.70, skip_classes=set()
        )
        assert rows == []


# ---------------------------------------------------------------------------
# build_annotations — YOLO coordinate format
# ---------------------------------------------------------------------------


class TestYoloCoordinates:
    def test_output_values_normalised(self) -> None:
        raw = _raw([_pred("holecard", 200, 200, 100, 160, conf=0.95)])
        rows = build_annotations(
            raw, CLASS_TO_ID, {}, min_confidence=0.70, skip_classes=set()
        )
        assert len(rows) == 1
        _class_id, cx_n, cy_n, w_n, h_n = rows[0]
        assert 0.0 <= cx_n <= 1.0
        assert 0.0 <= cy_n <= 1.0
        assert 0.0 < w_n <= 1.0
        assert 0.0 < h_n <= 1.0

    def test_output_is_five_tuple(self) -> None:
        raw = _raw([_pred("holecard", 200, 200, 100, 160, conf=0.95)])
        rows = build_annotations(
            raw, CLASS_TO_ID, {}, min_confidence=0.70, skip_classes=set()
        )
        assert len(rows[0]) == 5

    def test_correct_class_id_assigned(self) -> None:
        raw = _raw([_pred("player_other", 500, 300, 100, 80, conf=0.95)])
        rows = build_annotations(
            raw, CLASS_TO_ID, {}, min_confidence=0.70, skip_classes=set()
        )
        assert rows[0][0] == CLASS_TO_ID["player_other"]


# ---------------------------------------------------------------------------
# build_annotations — template injection
# ---------------------------------------------------------------------------


class TestTemplateInjection:
    TEMPLATES = {
        "player_me": {"cx_rel": 0.51, "cy_rel": 0.84, "w_rel": 0.155, "h_rel": 0.237}
    }

    def test_template_injected_when_class_not_detected(self) -> None:
        """player_me template should be added when only poker-table is present."""
        raw = _raw([_TABLE])
        rows = build_annotations(
            raw, CLASS_TO_ID, self.TEMPLATES, min_confidence=0.70, skip_classes=set()
        )
        class_ids = [r[0] for r in rows]
        assert CLASS_TO_ID["player_me"] in class_ids

    def test_template_not_injected_when_already_detected(self) -> None:
        """Template must not duplicate a class that the model already detected."""
        raw = _raw([_TABLE, _pred("player_me", 960, 900, 250, 210, conf=0.95)])
        rows = build_annotations(
            raw, CLASS_TO_ID, self.TEMPLATES, min_confidence=0.70, skip_classes=set()
        )
        player_me_rows = [r for r in rows if r[0] == CLASS_TO_ID["player_me"]]
        assert len(player_me_rows) == 1  # exactly one, not two

    def test_template_not_injected_without_poker_table(self) -> None:
        """Template requires a poker-table anchor — must be skipped if table not detected."""
        raw = _raw([_pred("holecard", 200, 200, 50, 80, conf=0.95)])
        rows = build_annotations(
            raw, CLASS_TO_ID, self.TEMPLATES, min_confidence=0.70, skip_classes=set()
        )
        class_ids = [r[0] for r in rows]
        assert CLASS_TO_ID["player_me"] not in class_ids

    def test_template_coordinates_clamped(self) -> None:
        """All template box coordinates must stay within [0, 1]."""
        raw = _raw([_TABLE])
        rows = build_annotations(
            raw, CLASS_TO_ID, self.TEMPLATES, min_confidence=0.70, skip_classes=set()
        )
        for _class_id, cx_n, cy_n, w_n, h_n in rows:
            assert 0.0 <= cx_n <= 1.0, f"cx_n={cx_n} out of range"
            assert 0.0 <= cy_n <= 1.0, f"cy_n={cy_n} out of range"
            assert 0.0 < w_n <= 1.0, f"w_n={w_n} out of range"
            assert 0.0 < h_n <= 1.0, f"h_n={h_n} out of range"


# ---------------------------------------------------------------------------
# write_data_yaml — format must be a YAML list, not an integer-keyed dict
# ---------------------------------------------------------------------------


class TestWriteDataYaml:
    """
    Roboflow (and the YOLO standard) expect `names` as a YAML sequence:
        names:
          - bet_pot_button
          - check_button
    Not an integer-keyed mapping:
        names:
          0: bet_pot_button
          1: check_button
    The latter causes Roboflow to store raw numeric IDs as class names.
    """

    CLASS_NAMES = ["bet_pot_button", "check_button", "chip_stack", "player_me"]

    def test_names_is_list_in_yaml(self, tmp_path: Path) -> None:
        write_data_yaml(tmp_path, self.CLASS_NAMES)
        with open(tmp_path / "data.yaml") as f:
            content = yaml.safe_load(f)
        assert isinstance(content["names"], list), (
            "'names' must be a YAML list — got dict with integer keys. "
            "This causes Roboflow to store numeric IDs instead of class names."
        )

    def test_names_order_preserved(self, tmp_path: Path) -> None:
        write_data_yaml(tmp_path, self.CLASS_NAMES)
        with open(tmp_path / "data.yaml") as f:
            content = yaml.safe_load(f)
        assert content["names"] == self.CLASS_NAMES

    def test_names_count_matches(self, tmp_path: Path) -> None:
        write_data_yaml(tmp_path, self.CLASS_NAMES)
        with open(tmp_path / "data.yaml") as f:
            content = yaml.safe_load(f)
        assert len(content["names"]) == len(self.CLASS_NAMES)

    def test_required_keys_present(self, tmp_path: Path) -> None:
        write_data_yaml(tmp_path, self.CLASS_NAMES)
        with open(tmp_path / "data.yaml") as f:
            content = yaml.safe_load(f)
        assert "names" in content
        assert "train" in content
        assert "val" in content
