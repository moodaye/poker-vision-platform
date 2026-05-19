from __future__ import annotations

import base64
import io
import json
from typing import Any

import responses as responses_lib

import orchestrator


def _upload_payload() -> dict[str, tuple[io.BytesIO, str]]:
    return {"image": (io.BytesIO(b"fake-image-bytes"), "table.png")}


@responses_lib.activate
def test_decide_runs_detector_enricher_parser_and_decision_engine() -> None:
    client = orchestrator.app.test_client()

    detector_response = {
        "predictions": [
            {"class": "holecard", "bbox": [10, 10, 40, 80], "confidence": 0.99},
            {"class": "holecard", "bbox": [45, 10, 75, 80], "confidence": 0.98},
        ]
    }
    enriched_response = {
        "objects": [
            {"class_name": "holecard", "classification": "Ah"},
            {"class_name": "holecard", "classification": "Kd"},
            {"class_name": "blinds", "ocr_text": "50/100"},
            {"class_name": "chip_stack", "ocr_text": "3000"},
            {"class_name": "pot", "ocr_text": "150"},
        ]
    }
    parser_response = {
        "hero_cards": ["Ah", "Kd"],
        "position": "BTN",
        "big_blind": 100,
        "small_blind": 50,
        "hero_stack": 3000,
        "pot": 150,
        "amount_to_call": 0,
        "action_history": [],
        "is_hero_turn": True,
        "hero_folded": False,
    }
    decision_response = {
        "action": "call",
        "amount": 100,
        "reason": "Integration test decision",
    }

    orchestrator.ROBOFLOW_API_KEY = "test-api-key"

    responses_lib.add(
        responses_lib.POST,
        orchestrator.ROBOFLOW_API_URL,
        json=detector_response,
        status=200,
    )

    def enricher_callback(request: Any) -> tuple[int, dict[str, str], str]:
        payload = json.loads(request.body.decode("utf-8"))
        assert payload["detections"] == detector_response["predictions"]
        assert base64.b64decode(payload["image_base64"]) == b"fake-image-bytes"
        return (200, {}, json.dumps(enriched_response))

    responses_lib.add_callback(
        responses_lib.POST,
        orchestrator.ENRICHER_URL,
        callback=enricher_callback,
        content_type="application/json",
    )

    responses_lib.add(
        responses_lib.POST,
        orchestrator.HAND_STATE_PARSER_URL,
        json=parser_response,
        status=200,
    )

    def decision_callback(request: Any) -> tuple[int, dict[str, str], str]:
        payload = json.loads(request.body.decode("utf-8"))
        assert payload == {
            "hero_cards": ["Ah", "Kd"],
            "position": "BTN",
            "big_blind": 100,
            "small_blind": 50,
            "hero_stack": 3000,
            "pot": 150,
            "amount_to_call": 0,
            "action_history": [],
            "is_hero_turn": True,
            "hero_folded": False,
        }
        return (200, {}, json.dumps(decision_response))

    responses_lib.add_callback(
        responses_lib.POST,
        orchestrator.DECISION_ENGINE_URL,
        callback=decision_callback,
        content_type="application/json",
    )

    response = client.post(
        "/decide",
        data=_upload_payload(),
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json() == decision_response


@responses_lib.activate
def test_decide_returns_502_when_enricher_fails() -> None:
    client = orchestrator.app.test_client()
    orchestrator.ROBOFLOW_API_KEY = "test-api-key"

    _det = [
        {
            "class": "player_me",
            "confidence": 0.90,
            "x": 100,
            "y": 200,
            "width": 50,
            "height": 30,
        }
    ]
    responses_lib.add(
        responses_lib.POST,
        orchestrator.ROBOFLOW_API_URL,
        json={"predictions": _det},
        status=200,
    )
    responses_lib.add(
        responses_lib.POST,
        orchestrator.ENRICHER_URL,
        json={"detail": "boom"},
        status=500,
    )

    response = client.post(
        "/decide",
        data=_upload_payload(),
        content_type="multipart/form-data",
    )

    assert response.status_code == 502
    assert "Detection enricher request failed" in response.get_json()["error"]


@responses_lib.activate
def test_decide_returns_502_when_detector_fails() -> None:
    client = orchestrator.app.test_client()
    orchestrator.ROBOFLOW_API_KEY = "test-api-key"

    responses_lib.add(
        responses_lib.POST,
        orchestrator.ROBOFLOW_API_URL,
        json={"error": "model error"},
        status=500,
    )

    response = client.post(
        "/decide",
        data=_upload_payload(),
        content_type="multipart/form-data",
    )

    assert response.status_code == 502
    assert "Object detector request failed" in response.get_json()["error"]


@responses_lib.activate
def test_decide_returns_502_when_parser_fails() -> None:
    client = orchestrator.app.test_client()
    orchestrator.ROBOFLOW_API_KEY = "test-api-key"

    _det = [
        {
            "class": "player_me",
            "confidence": 0.90,
            "x": 100,
            "y": 200,
            "width": 50,
            "height": 30,
        }
    ]
    responses_lib.add(
        responses_lib.POST,
        orchestrator.ROBOFLOW_API_URL,
        json={"predictions": _det},
        status=200,
    )
    responses_lib.add(
        responses_lib.POST,
        orchestrator.ENRICHER_URL,
        json={"objects": []},
        status=200,
    )
    responses_lib.add(
        responses_lib.POST,
        orchestrator.HAND_STATE_PARSER_URL,
        json={"detail": "parser exploded"},
        status=503,
    )

    response = client.post(
        "/decide",
        data=_upload_payload(),
        content_type="multipart/form-data",
    )

    assert response.status_code == 502
    assert "Hand state parser request failed" in response.get_json()["error"]


@responses_lib.activate
def test_decide_returns_502_when_decision_engine_fails() -> None:
    client = orchestrator.app.test_client()
    orchestrator.ROBOFLOW_API_KEY = "test-api-key"

    _det = [
        {
            "class": "player_me",
            "confidence": 0.90,
            "x": 100,
            "y": 200,
            "width": 50,
            "height": 30,
        }
    ]
    responses_lib.add(
        responses_lib.POST,
        orchestrator.ROBOFLOW_API_URL,
        json={"predictions": _det},
        status=200,
    )
    responses_lib.add(
        responses_lib.POST,
        orchestrator.ENRICHER_URL,
        json={"objects": []},
        status=200,
    )
    responses_lib.add(
        responses_lib.POST,
        orchestrator.HAND_STATE_PARSER_URL,
        json={"position": "BTN", "hero_cards": []},
        status=200,
    )
    responses_lib.add(
        responses_lib.POST,
        orchestrator.DECISION_ENGINE_URL,
        json={"detail": "strategy error"},
        status=500,
    )

    response = client.post(
        "/decide",
        data=_upload_payload(),
        content_type="multipart/form-data",
    )

    assert response.status_code == 502
    assert "Decision engine request failed" in response.get_json()["error"]


def test_decide_returns_500_when_api_key_missing() -> None:
    client = orchestrator.app.test_client()
    original_key = orchestrator.ROBOFLOW_API_KEY
    orchestrator.ROBOFLOW_API_KEY = None
    try:
        response = client.post(
            "/decide",
            data=_upload_payload(),
            content_type="multipart/form-data",
        )
        assert response.status_code == 500
        assert "ROBOFLOW_API_KEY" in response.get_json()["error"]
    finally:
        orchestrator.ROBOFLOW_API_KEY = original_key


@responses_lib.activate
def test_decide_returns_502_when_detector_returns_no_predictions_key() -> None:
    """Detector returns 200 but the JSON has no 'predictions' list."""
    client = orchestrator.app.test_client()
    orchestrator.ROBOFLOW_API_KEY = "test-api-key"

    responses_lib.add(
        responses_lib.POST,
        orchestrator.ROBOFLOW_API_URL,
        json={"unexpected_key": "data"},
        status=200,
    )

    response = client.post(
        "/decide",
        data=_upload_payload(),
        content_type="multipart/form-data",
    )

    assert response.status_code == 502
    assert "Object detector" in response.get_json()["error"]


@responses_lib.activate
def test_decide_returns_watching_when_no_detections() -> None:
    """Detector returns zero predictions — orchestrator short-circuits before calling
    any downstream service and returns a watching response."""
    client = orchestrator.app.test_client()
    orchestrator.ROBOFLOW_API_KEY = "test-api-key"

    responses_lib.add(
        responses_lib.POST,
        orchestrator.ROBOFLOW_API_URL,
        json={"predictions": []},
        status=200,
    )
    # No enricher/parser/engine mocks registered — they must NOT be called.

    response = client.post(
        "/decide",
        data=_upload_payload(),
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["action"] == "watching"
    assert "No poker table" in body["reason"]


@responses_lib.activate
def test_decide_returns_502_when_enricher_returns_no_objects_key() -> None:
    """Enricher returns 200 but the JSON has no 'objects' list."""
    client = orchestrator.app.test_client()
    orchestrator.ROBOFLOW_API_KEY = "test-api-key"

    _one_detection = [
        {
            "class": "player_me",
            "confidence": 0.90,
            "x": 100,
            "y": 200,
            "width": 50,
            "height": 30,
        }
    ]
    responses_lib.add(
        responses_lib.POST,
        orchestrator.ROBOFLOW_API_URL,
        json={"predictions": _one_detection},
        status=200,
    )
    responses_lib.add(
        responses_lib.POST,
        orchestrator.ENRICHER_URL,
        json={"wrong_key": "data"},
        status=200,
    )

    response = client.post(
        "/decide",
        data=_upload_payload(),
        content_type="multipart/form-data",
    )

    assert response.status_code == 502
    assert "Detection enricher" in response.get_json()["error"]
