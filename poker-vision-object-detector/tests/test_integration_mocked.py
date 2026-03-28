"""
Integration tests using `responses` library to mock HTTP calls to Roboflow.
Creates real temp images via Pillow; runs the full runner pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import responses as responses_lib
from PIL import Image
from poker_vision.detect.config import (
    DetectConfig,
    IOConfig,
    NormalizationConfig,
    RoboflowConfig,
    RunConfig,
)
from poker_vision.detect.runner import run

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_tiny_png(
    path: Path,
    color: tuple[int, int, int] = (0, 128, 255),
    size: tuple[int, int] = (8, 8),
) -> None:
    img = Image.new("RGB", size, color)
    img.save(path, format="PNG")


MOCK_RESPONSE_CARD_A = {
    "predictions": [
        {
            "x": 4.0,
            "y": 4.0,
            "width": 6.0,
            "height": 6.0,
            "confidence": 0.88,
            "class": "holecard",
            "class_id": 10,
        }
    ],
    "image": {"width": 8, "height": 8},
}

MOCK_RESPONSE_CARD_B = {
    "predictions": [
        {
            "x": 4.0,
            "y": 4.0,
            "width": 4.0,
            "height": 4.0,
            "confidence": 0.72,
            "class": "dealer_button",
            "class_id": 7,
        },
        {
            "x": 2.0,
            "y": 2.0,
            "width": 2.0,
            "height": 2.0,
            "confidence": 0.20,
            "class": "check_fold_button",
            "class_id": 5,
        },
    ],
    "image": {"width": 8, "height": 8},
}


def _build_cfg(input_dir: Path, output_dir: Path) -> DetectConfig:
    return DetectConfig(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        roboflow=RoboflowConfig(
            api_base="https://detect.roboflow.com",
            project="poker-vision",
            version=2,
            timeout_seconds=5,
            max_retries=0,  # no retries in tests
            backoff_seconds=0.0,
            confidence_threshold=0.50,
            overlap_threshold=0.50,
        ),
        io=IOConfig(recursive_input=False, image_extensions=[".png"]),
        normalization=NormalizationConfig(bbox_rounding="round"),
        run=RunConfig(
            run_id="integration-test-run",
            save_raw_predictions=True,
            save_normalized_detections=True,
            save_run_summary=True,
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@responses_lib.activate
def test_integration_two_images(tmp_path: Path) -> None:
    """Full pipeline with two images: output files exist and schema is correct."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    _make_tiny_png(input_dir / "card_a.png")
    _make_tiny_png(input_dir / "card_b.png", color=(255, 0, 0))

    endpoint = "https://detect.roboflow.com/poker-vision/2"

    # Register mock responses (responses lib matches by URL prefix)
    responses_lib.add(
        responses_lib.POST,
        endpoint,
        json=MOCK_RESPONSE_CARD_A,
        status=200,
    )
    responses_lib.add(
        responses_lib.POST,
        endpoint,
        json=MOCK_RESPONSE_CARD_B,
        status=200,
    )

    cfg = _build_cfg(input_dir, output_dir)
    exit_code = run(cfg, api_key="test-api-key")

    assert exit_code == 0

    # Check all expected output files exist
    assert (output_dir / "run_summary.json").exists()
    assert (output_dir / "run_config.resolved.json").exists()
    assert (output_dir / "predictions_raw" / "card_a.json").exists()
    assert (output_dir / "predictions_raw" / "card_b.json").exists()
    assert (output_dir / "detections_normalized" / "card_a.detections.json").exists()
    assert (output_dir / "detections_normalized" / "card_b.detections.json").exists()


@responses_lib.activate
def test_integration_run_summary_counts(tmp_path: Path) -> None:
    """run_summary.json has correct class counts and total detection count."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    _make_tiny_png(input_dir / "card_a.png")
    _make_tiny_png(input_dir / "card_b.png", color=(200, 100, 50))

    endpoint = "https://detect.roboflow.com/poker-vision/2"
    responses_lib.add(
        responses_lib.POST, endpoint, json=MOCK_RESPONSE_CARD_A, status=200
    )
    responses_lib.add(
        responses_lib.POST, endpoint, json=MOCK_RESPONSE_CARD_B, status=200
    )

    cfg = _build_cfg(input_dir, output_dir)
    run(cfg, api_key="test-api-key")

    summary = json.loads((output_dir / "run_summary.json").read_text())

    assert summary["images_total"] == 2
    assert summary["images_succeeded"] == 2
    assert summary["images_failed"] == 0
    # holecard (0.88) from card_a, dealer_button (0.72) from card_b
    # check_fold_button (0.20) is below threshold and filtered
    assert summary["detections_total"] == 2
    assert summary["detections_per_class"]["holecard"] == 1
    assert summary["detections_per_class"]["dealer_button"] == 1
    assert "check_fold_button" not in summary["detections_per_class"]
    assert summary["failures"] == []


@responses_lib.activate
def test_integration_normalized_bbox(tmp_path: Path) -> None:
    """Verify bbox conversion is correct in normalized output."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    _make_tiny_png(input_dir / "card_a.png")

    endpoint = "https://detect.roboflow.com/poker-vision/2"
    responses_lib.add(
        responses_lib.POST, endpoint, json=MOCK_RESPONSE_CARD_A, status=200
    )

    cfg = _build_cfg(input_dir, output_dir)
    run(cfg, api_key="test-api-key")

    norm = json.loads(
        (output_dir / "detections_normalized" / "card_a.detections.json").read_text()
    )
    assert len(norm["detections"]) == 1
    det = norm["detections"][0]
    # x=4, y=4, w=6, h=6 -> xyxy: x1=round(4-3)=1, y1=1, x2=round(4+3)=7, y2=7
    assert det["bbox_xyxy"] == [1, 1, 7, 7]
    assert det["bbox_xywh_center"] == [4, 4, 6, 6]
    assert det["class_name"] == "holecard"
    assert det["confidence"] == pytest.approx(0.88, abs=1e-5)


@responses_lib.activate
def test_integration_api_key_not_in_outputs(tmp_path: Path) -> None:
    """Ensure API key does not appear anywhere in output files."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    _make_tiny_png(input_dir / "card_a.png")

    endpoint = "https://detect.roboflow.com/poker-vision/2"
    responses_lib.add(
        responses_lib.POST, endpoint, json=MOCK_RESPONSE_CARD_A, status=200
    )

    cfg = _build_cfg(input_dir, output_dir)
    secret_key = "super-secret-api-key-12345"
    run(cfg, api_key=secret_key)

    for json_file in output_dir.rglob("*.json"):
        content = json_file.read_text()
        assert secret_key not in content, f"API key found in output file: {json_file}"


@responses_lib.activate
def test_integration_http_failure_recorded(tmp_path: Path) -> None:
    """HTTP 500 failure is recorded in run_summary and exit code reflects partial success."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    _make_tiny_png(input_dir / "card_a.png")
    _make_tiny_png(input_dir / "card_b.png", color=(0, 255, 0))

    endpoint = "https://detect.roboflow.com/poker-vision/2"
    # First call succeeds, second fails
    responses_lib.add(
        responses_lib.POST, endpoint, json=MOCK_RESPONSE_CARD_A, status=200
    )
    responses_lib.add(
        responses_lib.POST, endpoint, json={"error": "server error"}, status=500
    )

    cfg = _build_cfg(input_dir, output_dir)
    exit_code = run(cfg, api_key="test-api-key")

    # 1 success, 1 failure -> exit 0 (not all failed)
    assert exit_code == 0

    summary = json.loads((output_dir / "run_summary.json").read_text())
    assert summary["images_failed"] == 1
    assert len(summary["failures"]) == 1
    assert summary["failures"][0]["http_status"] == 500


@responses_lib.activate
def test_integration_query_params_sent(tmp_path: Path) -> None:
    """Verify that confidence, overlap, and api_key query params are sent."""
    import urllib.parse

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    _make_tiny_png(input_dir / "card_a.png")

    endpoint = "https://detect.roboflow.com/poker-vision/2"
    responses_lib.add(
        responses_lib.POST, endpoint, json=MOCK_RESPONSE_CARD_A, status=200
    )

    cfg = _build_cfg(input_dir, output_dir)
    run(cfg, api_key="my-test-key")

    assert len(responses_lib.calls) == 1
    call = responses_lib.calls[0]
    parsed = urllib.parse.urlparse(call.request.url)
    params = urllib.parse.parse_qs(parsed.query)  # type: ignore[type-var]

    assert "api_key" in params
    assert params["api_key"][0] == "my-test-key"
    assert "confidence" in params
    assert "overlap" in params
