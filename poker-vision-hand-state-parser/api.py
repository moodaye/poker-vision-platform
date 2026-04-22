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

import logging

from flask import Flask, Response, jsonify, request
from hand_state_parser import build_hand_state

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)


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
        hand_state = build_hand_state(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:
        logger.exception("Error in build_hand_state")
        return jsonify({"error": f"Internal error: {exc}"}), 500

    logger.info(
        f"Parsed hand state: position={hand_state.get('position')} hero_cards={hand_state.get('hero_cards')}"
    )
    return jsonify(hand_state)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=False)
