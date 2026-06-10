from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any, cast

import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request

load_dotenv(dotenv_path=Path(__file__).parent / "poker-vision-object-detector" / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ROBOFLOW_API_URL = os.environ.get(
    "ROBOFLOW_API_URL", "https://serverless.roboflow.com/pokertabledetection/7"
)
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ENRICHER_URL = os.environ.get("ENRICHER_URL", "http://127.0.0.1:5004/enrich")
HAND_STATE_PARSER_URL = os.environ.get(
    "HAND_STATE_PARSER_URL", "http://127.0.0.1:5003/parse"
)
DECISION_ENGINE_URL = os.environ.get(
    "DECISION_ENGINE_URL", "http://127.0.0.1:5002/decide"
)
ACTION_EXECUTOR_URL = os.environ.get(
    "ACTION_EXECUTOR_URL", "http://127.0.0.1:5005/execute"
)
REQUEST_TIMEOUT_SECONDS = 30

app = Flask(__name__)


def _json_error(message: str, status_code: int) -> tuple[Response, int]:
    return jsonify({"error": message}), status_code


def call_object_detector(image_bytes: bytes) -> list[dict[str, Any]]:
    if not ROBOFLOW_API_KEY:
        raise RuntimeError("ROBOFLOW_API_KEY is not configured")

    logger.info("Calling object detector")
    t0 = time.perf_counter()
    response = requests.post(
        ROBOFLOW_API_URL,
        params={"api_key": ROBOFLOW_API_KEY},
        files={"file": image_bytes},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    logger.info("Object detector: %.2fs", time.perf_counter() - t0)

    payload = response.json()
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("Object detector response did not contain predictions list")
    return predictions


def call_detection_enricher(
    image_bytes: bytes, detections: list[dict[str, Any]]
) -> dict[str, Any]:
    logger.info("Calling detection enricher")
    t0 = time.perf_counter()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    response = requests.post(
        ENRICHER_URL,
        json={"image_base64": image_base64, "detections": detections},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    logger.info("Detection enricher: %.2fs", time.perf_counter() - t0)
    payload = response.json()
    if not isinstance(payload.get("objects"), list):
        raise ValueError("Detection enricher response did not contain objects list")
    return cast(dict[str, Any], payload)


def call_hand_state_parser(enriched_payload: dict[str, Any]) -> dict[str, Any]:
    logger.info("Calling hand state parser")
    t0 = time.perf_counter()
    response = requests.post(
        HAND_STATE_PARSER_URL,
        json=enriched_payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    logger.info("Hand state parser: %.2fs", time.perf_counter() - t0)
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Hand state parser response must be a JSON object")
    return cast(dict[str, Any], payload)


def call_decision_engine(hand_state: dict[str, Any]) -> dict[str, Any]:
    logger.info("Calling decision engine")
    t0 = time.perf_counter()
    response = requests.post(
        DECISION_ENGINE_URL,
        json=hand_state,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    logger.info("Decision engine: %.2fs", time.perf_counter() - t0)
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Decision engine response must be a JSON object")
    return cast(dict[str, Any], payload)


def call_action_executor(decision: dict[str, Any]) -> dict[str, Any] | None:
    """Enact the decision by clicking the appropriate button in the poker client.

    Skips silently when the decision action is 'watching' (nothing to do) or
    when the action executor service is not reachable (so the pipeline degrades
    gracefully without breaking the /decide response).

    Args:
        decision: The JSON payload returned by the decision engine.

    Returns:
        The action executor response payload, or None if skipped / unreachable.
    """
    action = decision.get("action", "watching")
    if action == "watching":
        return None
    try:
        logger.info("Calling action executor (action=%r)", action)
        t0 = time.perf_counter()
        response = requests.post(
            ACTION_EXECUTOR_URL,
            json={"action": action, "amount": decision.get("amount")},
            timeout=5,
        )
        response.raise_for_status()
        logger.info("Action executor: %.2fs", time.perf_counter() - t0)
        return cast(dict[str, Any], response.json())
    except requests.RequestException as exc:
        logger.warning("Action executor not available: %s", exc)
        return None


@app.route("/health")
def health() -> Response:
    return jsonify({"status": "ok"})


@app.route("/decide", methods=["POST"])
def decide() -> tuple[Response, int] | Response:
    logger.info(
        "Inbound /decide request from %s content_type=%s content_length=%s",
        request.remote_addr,
        request.content_type,
        request.content_length,
    )

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

    if not detections:
        # No poker objects found — not a valid game frame.
        # TODO(gh-issue): replace with a dedicated "game_active" object class in the
        # detection model so the system can distinguish an idle table from an active
        # hand rather than relying on any detection being present.
        logger.info("No detections — not a poker game frame, returning watching")
        return jsonify(
            {
                "action": "watching",
                "amount": None,
                "reason": "No poker table detected in frame",
            }
        )

    try:
        enriched_payload = call_detection_enricher(image_bytes, detections)
    except requests.HTTPError as exc:
        logger.exception("Detection enricher request failed")
        return _json_error(f"Detection enricher request failed: {exc}", 502)
    except (requests.RequestException, ValueError) as exc:
        logger.exception("Detection enricher error")
        return _json_error(f"Detection enricher error: {exc}", 502)

    try:
        hand_state = call_hand_state_parser(enriched_payload)
    except requests.HTTPError as exc:
        logger.exception("Hand state parser request failed")
        return _json_error(f"Hand state parser request failed: {exc}", 502)
    except (requests.RequestException, ValueError) as exc:
        logger.exception("Hand state parser error")
        return _json_error(f"Hand state parser error: {exc}", 502)

    try:
        decision = call_decision_engine(hand_state)
    except requests.HTTPError as exc:
        logger.exception("Decision engine request failed")
        return _json_error(f"Decision engine request failed: {exc}", 502)
    except (requests.RequestException, ValueError) as exc:
        logger.exception("Decision engine error")
        return _json_error(f"Decision engine error: {exc}", 502)

    execution_result = call_action_executor(decision)
    if execution_result:
        logger.info("Action executor: %s", execution_result.get("message"))

    response_data = dict(decision)
    if execution_result:
        response_data["execution"] = execution_result
    return jsonify(response_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5100, debug=False)
