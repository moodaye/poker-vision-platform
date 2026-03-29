from typing import Any

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/decide", methods=["POST"])
def decide() -> Any:
    # 1. Get the screenshot image from the request
    if "image" not in request.files:
        return jsonify({"error": "Missing image file"}), 400

    _image_file = request.files["image"]  # will be used in pipeline steps below

    # TODO: Call object detector with the image
    # TODO: Call snipper with image + detections
    # TODO: Call game state parser with enriched detections
    # TODO: Call decision engine with HandState

    # For now, return a mock decision
    mock_decision = {
        "action": "call",
        "amount": 400,
        "reason": "MVP mock: always call preflop",
    }
    return jsonify(mock_decision)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5100, debug=True)
