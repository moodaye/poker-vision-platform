"""
Tests for runner output file structure and run_summary.json schema.
Uses mocked HTTP (no live API) with tiny Pillow-generated images.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from poker_vision.detect.config import (
    DetectConfig,
    IOConfig,
    NormalizationConfig,
    RoboflowConfig,
    RunConfig,
)
from poker_vision.detect.runner import collect_images, run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_image(
    path: Path,
    color: tuple[int, int, int] = (255, 0, 0),
    size: tuple[int, int] = (10, 10),
) -> None:
    img = Image.new("RGB", size, color)
    img.save(path)


FAKE_RAW_RESPONSE = {
    "predictions": [
        {
            "x": 5.0,
            "y": 5.0,
            "width": 4.0,
            "height": 4.0,
            "confidence": 0.95,
            "class": "holecard",
            "class_id": 10,
        },
        {
            "x": 7.0,
            "y": 7.0,
            "width": 2.0,
            "height": 2.0,
            "confidence": 0.30,
            "class": "bet_box",
            "class_id": 1,
        },
    ],
    "image": {"width": 10, "height": 10},
}


# ---------------------------------------------------------------------------
# collect_images tests
# ---------------------------------------------------------------------------


def test_collect_images_sorted(tmp_path: Path) -> None:
    for name in ("c.png", "a.jpg", "b.jpeg"):
        _make_tiny_image(tmp_path / name)
    paths = collect_images(tmp_path, [".png", ".jpg", ".jpeg"], recursive=False)
    assert [p.name for p in paths] == ["a.jpg", "b.jpeg", "c.png"]


def test_collect_images_recursive(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    _make_tiny_image(tmp_path / "root.png")
    _make_tiny_image(sub / "nested.png")
    paths = collect_images(tmp_path, [".png"], recursive=True)
    assert len(paths) == 2


def test_collect_images_non_recursive(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    _make_tiny_image(tmp_path / "root.png")
    _make_tiny_image(sub / "nested.png")
    paths = collect_images(tmp_path, [".png"], recursive=False)
    assert len(paths) == 1
    assert paths[0].name == "root.png"


def test_collect_images_extension_filter(tmp_path: Path) -> None:
    _make_tiny_image(tmp_path / "a.png")
    (tmp_path / "b.txt").write_text("not an image")
    paths = collect_images(tmp_path, [".png"], recursive=False)
    assert len(paths) == 1


# ---------------------------------------------------------------------------
# run() output structure tests (mocked via monkeypatching)
# ---------------------------------------------------------------------------


def _build_cfg(input_dir: Path, output_dir: Path) -> DetectConfig:
    cfg = DetectConfig(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        roboflow=RoboflowConfig(
            api_base="https://detect.roboflow.com",
            project="test-project",
            version=1,
            confidence_threshold=0.50,
            overlap_threshold=0.50,
        ),
        io=IOConfig(recursive_input=False, image_extensions=[".png"]),
        normalization=NormalizationConfig(bbox_rounding="round"),
        run=RunConfig(
            run_id="test-run-001",
            save_raw_predictions=True,
            save_normalized_detections=True,
            save_run_summary=True,
        ),
    )
    return cfg


def test_run_creates_output_files(tmp_path: Path, monkeypatch: Any) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    _make_tiny_image(input_dir / "card1.png")

    cfg = _build_cfg(input_dir, output_dir)

    # Patch RoboflowClient.predict to avoid real HTTP
    from poker_vision.detect import client as client_mod

    monkeypatch.setattr(
        client_mod.RoboflowClient,
        "predict",
        lambda self, path: FAKE_RAW_RESPONSE,
    )

    exit_code = run(cfg, api_key="fake-key")

    assert exit_code == 0
    assert (output_dir / "run_summary.json").exists()
    assert (output_dir / "run_config.resolved.json").exists()
    assert (output_dir / "predictions_raw" / "card1.json").exists()
    assert (output_dir / "detections_normalized" / "card1.detections.json").exists()


def test_run_summary_schema(tmp_path: Path, monkeypatch: Any) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    _make_tiny_image(input_dir / "card1.png")

    cfg = _build_cfg(input_dir, output_dir)

    from poker_vision.detect import client as client_mod

    monkeypatch.setattr(
        client_mod.RoboflowClient,
        "predict",
        lambda self, path: FAKE_RAW_RESPONSE,
    )

    run(cfg, api_key="fake-key")

    summary = json.loads((output_dir / "run_summary.json").read_text())
    for key in (
        "run_id",
        "started_at",
        "finished_at",
        "images_total",
        "images_succeeded",
        "images_failed",
        "detections_total",
        "detections_per_class",
        "failures",
    ):
        assert key in summary, f"Missing key in run_summary: {key}"

    assert summary["run_id"] == "test-run-001"
    assert summary["images_total"] == 1
    assert summary["images_succeeded"] == 1
    assert summary["images_failed"] == 0
    # Only holecard passes the 0.50 threshold; bet_box at 0.30 is filtered
    assert summary["detections_total"] == 1
    assert summary["detections_per_class"] == {"holecard": 1}
    assert summary["failures"] == []


def test_run_normalized_schema(tmp_path: Path, monkeypatch: Any) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    _make_tiny_image(input_dir / "card1.png")

    cfg = _build_cfg(input_dir, output_dir)

    from poker_vision.detect import client as client_mod

    monkeypatch.setattr(
        client_mod.RoboflowClient,
        "predict",
        lambda self, path: FAKE_RAW_RESPONSE,
    )

    run(cfg, api_key="fake-key")

    norm = json.loads(
        (output_dir / "detections_normalized" / "card1.detections.json").read_text()
    )
    for key in ("source_image", "image_width", "image_height", "model", "detections"):
        assert key in norm
    assert norm["model"]["provider"] == "roboflow"
    assert len(norm["detections"]) == 1
    det = norm["detections"][0]
    assert det["class_name"] == "holecard"
    assert len(det["bbox_xyxy"]) == 4
    assert len(det["bbox_xywh_center"]) == 4


def test_run_resolved_config_no_api_key(tmp_path: Path, monkeypatch: Any) -> None:
    """resolved config must not contain the api_key."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    _make_tiny_image(input_dir / "card1.png")

    cfg = _build_cfg(input_dir, output_dir)

    from poker_vision.detect import client as client_mod

    monkeypatch.setattr(
        client_mod.RoboflowClient,
        "predict",
        lambda self, path: FAKE_RAW_RESPONSE,
    )

    run(cfg, api_key="fake-key")

    resolved_text = (output_dir / "run_config.resolved.json").read_text()
    assert "fake-key" not in resolved_text
    assert "api_key" not in resolved_text


def test_run_continues_on_failure(tmp_path: Path, monkeypatch: Any) -> None:
    """A failed image should be logged in failures; succeeded images still processed."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    _make_tiny_image(input_dir / "card1.png")
    _make_tiny_image(input_dir / "card2.png")

    cfg = _build_cfg(input_dir, output_dir)

    call_count = {"n": 0}

    def fake_predict(self: Any, path: Path) -> dict[str, Any]:
        call_count["n"] += 1
        if path.name == "card1.png":
            from poker_vision.detect.client import RoboflowAPIError

            raise RoboflowAPIError("Simulated failure", http_status=500)
        return FAKE_RAW_RESPONSE

    from poker_vision.detect import client as client_mod

    monkeypatch.setattr(client_mod.RoboflowClient, "predict", fake_predict)

    exit_code = run(cfg, api_key="fake-key")

    # 1 succeeded, 1 failed -> exit 0 (not all failed)
    assert exit_code == 0
    summary = json.loads((output_dir / "run_summary.json").read_text())
    assert summary["images_succeeded"] == 1
    assert summary["images_failed"] == 1
    assert len(summary["failures"]) == 1
    assert summary["failures"][0]["http_status"] == 500


def test_run_all_failed_returns_nonzero(tmp_path: Path, monkeypatch: Any) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    _make_tiny_image(input_dir / "card1.png")

    cfg = _build_cfg(input_dir, output_dir)

    def fake_predict(self: Any, path: Path) -> dict[str, Any]:
        from poker_vision.detect.client import RoboflowAPIError

        raise RoboflowAPIError("All fail", http_status=500)

    from poker_vision.detect import client as client_mod

    monkeypatch.setattr(client_mod.RoboflowClient, "predict", fake_predict)

    exit_code = run(cfg, api_key="fake-key")
    assert exit_code == 1


def test_run_no_images_returns_zero(tmp_path: Path, monkeypatch: Any) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    cfg = _build_cfg(input_dir, output_dir)

    exit_code = run(cfg, api_key="fake-key")
    assert exit_code == 0
