"""
api.py — Hand state parser service.

Stateless: no model to load. Accepts an enriched detection payload as JSON
and returns a structured HandState dict for the decision engine.

Run:
    uv run python api.py

Endpoints:
    POST /parse   body: {"objects": [...]}  (enriched detection payload)
                  returns: HandState dict

    GET  /health  returns: {"status": "ok"}

Config (`config.yaml`):
    log_diagnostics (default: false)
        When true (1/true/yes/on), /parse logs full HandState and field diagnostics.
        Response payload remains HandState-only.

Example request body:
    {
        "objects": [
            {"class_name": "holecard", "classification": "Ah", "confidence": 0.95, "classification_conf": 0.93},
            {"class_name": "holecard", "classification": "Kd", "confidence": 0.95, "classification_conf": 0.91},
            {"class_name": "chip_stack", "ocr_text": "3000", "confidence": 0.90, "ocr_conf": 0.88},
            {"class_name": "blinds", "ocr_text": "50/100", "confidence": 0.88, "ocr_conf": 0.85}
        ]
    }
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import yaml
from flask import Flask, Response, jsonify, request
from hand_state_parser import build_hand_state, build_hand_state_with_diagnostics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
with _CONFIG_PATH.open() as _f:
    _CFG: dict = yaml.safe_load(_f)


def _should_log_diagnostics() -> bool:
    env_raw = os.environ.get("HAND_STATE_PARSER_LOG_DIAGNOSTICS")
    if env_raw is not None:
        return str(env_raw).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    raw = _CFG.get("log_diagnostics", False)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int | float):
        return bool(raw)
    return str(raw).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _should_pretty_log() -> bool:
    env_raw = os.environ.get("HAND_STATE_PARSER_PRETTY_JSON_LOGS")
    if env_raw is not None:
        return str(env_raw).strip().lower() in {"1", "true", "yes", "on"}

    raw = _CFG.get("pretty_json_logs", True)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _log_json(label: str, payload: dict[str, object]) -> None:
    indent = 2 if _should_pretty_log() else None
    logger.info("%s:\n%s", label, json.dumps(payload, sort_keys=True, default=str, indent=indent))


# ── Routes ────────────────────────────────────────────────────────────────────


@app.route("/health")
def health() -> Response:
    return jsonify({"status": "ok"})


@app.route("/parse", methods=["POST"])
def parse() -> tuple[Response, int] | Response:
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    if not isinstance(data.get("objects"), list):
        return jsonify({"error": "Request body must contain an 'objects' list"}), 400

    try:
        if _should_log_diagnostics():
            hand_state, diagnostics = build_hand_state_with_diagnostics(data)
            _log_json("Parsed hand state", hand_state)
            _log_json("Hand state diagnostics", diagnostics)
        else:
            hand_state = build_hand_state(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:
        logger.exception("Error in build_hand_state")
        return jsonify({"error": f"Internal error: {exc}"}), 500
    return jsonify(hand_state)


if __name__ == "__main__":
    port = int(_CFG.get("port", 5003))
    app.run(host="0.0.0.0", port=port, debug=False)
