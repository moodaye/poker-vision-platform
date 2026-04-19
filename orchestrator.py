from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, cast

import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request

from hand_state_parser import build_hand_state

load_dotenv(dotenv_path=Path(__file__).parent / "poker-vision-object-detector" / ".env")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROBOFLOW_API_URL = os.environ.get(
    "ROBOFLOW_API_URL", "https://detect.roboflow.com/pokertabledetection/6"
)
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ENRICHER_URL = os.environ.get("ENRICHER_URL", "http://localhost:5004/enrich")
DECISION_ENGINE_URL = os.environ.get(
    "DECISION_ENGINE_URL", "http://localhost:5002/decide"
)
REQUEST_TIMEOUT_SECONDS = 20

app = Flask(__name__)


def _json_error(message: str, status_code: int) -> tuple[Response, int]:
    return jsonify({"error": message}), status_code


def call_object_detector(image_bytes: bytes) -> list[dict[str, Any]]:
    if not ROBOFLOW_API_KEY:
        raise RuntimeError("ROBOFLOW_API_KEY is not configured")

    logger.info("Calling object detector")
    response = requests.post(
        ROBOFLOW_API_URL,
        params={"api_key": ROBOFLOW_API_KEY},
        files={"file": image_bytes},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    payload = response.json()
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("Object detector response did not contain predictions list")
    return predictions


def call_detection_enricher(
    image_bytes: bytes, detections: list[dict[str, Any]]
) -> dict[str, Any]:
    logger.info("Calling detection enricher")
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    response = requests.post(
        ENRICHER_URL,
        json={"image_base64": image_base64, "detections": detections},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload.get("objects"), list):
        raise ValueError("Detection enricher response did not contain objects list")
    return cast(dict[str, Any], payload)


def call_decision_engine(hand_state: dict[str, Any]) -> dict[str, Any]:
    logger.info("Calling decision engine")
    response = requests.post(
        DECISION_ENGINE_URL,
        json=hand_state,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Decision engine response must be a JSON object")
    return cast(dict[str, Any], payload)


@app.route("/health")
def health() -> Response:
    return jsonify({"status": "ok"})


@app.route("/decide", methods=["POST"])
def decide() -> tuple[Response, int] | Response:
    if "image" not in request.files:
        return _json_error("Missing image file", 400)

    image_file = request.files["image"]
    image_bytes = image_file.read()
    if not image_bytes:
        return _json_error("Uploaded image file is empty", 400)

    try:
        detections = call_object_detector(image_bytes)
    except RuntimeError as exc:
        logger.exception("Object detector configuration error")
        return _json_error(str(exc), 500)
    except requests.HTTPError as exc:
        logger.exception("Object detector request failed")
        return _json_error(f"Object detector request failed: {exc}", 502)
    except (requests.RequestException, ValueError) as exc:
        logger.exception("Object detector error")
        return _json_error(f"Object detector error: {exc}", 502)

    try:
        enriched_payload = call_detection_enricher(image_bytes, detections)
    except requests.HTTPError as exc:
        logger.exception("Detection enricher request failed")
        return _json_error(f"Detection enricher request failed: {exc}", 502)
    except (requests.RequestException, ValueError) as exc:
        logger.exception("Detection enricher error")
        return _json_error(f"Detection enricher error: {exc}", 502)

    try:
        hand_state = build_hand_state(enriched_payload)
    except ValueError as exc:
        logger.exception("Hand state parser error")
        return _json_error(f"Hand state parser error: {exc}", 422)

    try:
        decision = call_decision_engine(hand_state)
    except requests.HTTPError as exc:
        logger.exception("Decision engine request failed")
        return _json_error(f"Decision engine request failed: {exc}", 502)
    except (requests.RequestException, ValueError) as exc:
        logger.exception("Decision engine error")
        return _json_error(f"Decision engine error: {exc}", 502)

    return jsonify(decision)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5100, debug=False)
