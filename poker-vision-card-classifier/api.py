"""
api.py — Card classifier inference service.

Loads the trained model at startup and serves predictions over HTTP.

Run:
    uv run python api.py

Endpoints:
    POST /classify   body: {"image": "<base64 encoded image>"}
                     returns: {"label": "KH", "confidence": 0.97}

    GET  /health     returns: {"status": "ok", "classes": 30}
"""

from __future__ import annotations

import base64
import json
import logging
import time
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, Response, jsonify, request
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
MODEL_PATH = _HERE / "model" / "model.pt"
CLASSES_PATH = _HERE / "model" / "classes.json"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LOW_CONFIDENCE_THRESHOLD = 0.70


class LetterboxToSquare:
    """Pad image to square with black borders, preserving aspect ratio."""

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        size = max(w, h)
        new_img = Image.new("RGB", (size, size), (0, 0, 0))
        new_img.paste(img, ((size - w) // 2, (size - h) // 2))
        return new_img


# ── Model loading ─────────────────────────────────────────────────────────────


def load_model() -> tuple[nn.Module, dict[str, str]]:
    """Load saved weights and class mapping from model/."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"classes.json not found at {CLASSES_PATH}. Run train.py first."
        )

    with open(CLASSES_PATH) as f:
        idx_to_class: dict[str, str] = json.load(f)

    num_classes = len(idx_to_class)
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    logger.info(f"Model loaded — {num_classes} classes")
    return model, idx_to_class


# ── Inference ─────────────────────────────────────────────────────────────────

_infer_transform = transforms.Compose(
    [
        LetterboxToSquare(),
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def predict(
    model: nn.Module,
    idx_to_class: dict[str, str],
    image: Image.Image,
) -> tuple[str, float, bool]:
    """Return (label, confidence, low_confidence) for a single PIL image."""
    tensor = _infer_transform(image.convert("RGB")).unsqueeze(0)  # add batch dim
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, idx = probs.max(dim=1)
    label = idx_to_class[str(idx.item())]
    conf_value = round(confidence.item(), 4)
    return label, conf_value, conf_value < LOW_CONFIDENCE_THRESHOLD


def predict_batch(
    model: nn.Module,
    idx_to_class: dict[str, str],
    images: list[Image.Image],
) -> list[tuple[str, float, bool]]:
    """Return [(label, confidence, low_confidence), ...] for a list of PIL images.

    All images are classified in a single forward pass by stacking their
    preprocessed tensors into one batch. This is significantly faster than
    calling predict() per image because the model inference is amortised
    across the batch dimension.
    """
    if not images:
        return []
    tensors = [_infer_transform(img.convert("RGB")) for img in images]
    batch = torch.stack(tensors)  # shape: (N, 3, 224, 224)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        confidences, indices = probs.max(dim=1)
    results: list[tuple[str, float, bool]] = []
    for conf, idx in zip(confidences, indices):
        label = idx_to_class[str(idx.item())]
        conf_value = round(conf.item(), 4)
        results.append((label, conf_value, conf_value < LOW_CONFIDENCE_THRESHOLD))
    return results


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

_model: nn.Module | None = None
_idx_to_class: dict[str, str] | None = None
_model_error: str | None = None


@app.before_request  # type: ignore[misc]
def _ensure_loaded() -> None:
    from flask import request as _req
    if _req.endpoint == "health":
        return
    global _model, _idx_to_class, _model_error
    if _model is None and _model_error is None:
        try:
            _model, _idx_to_class = load_model()
        except FileNotFoundError as exc:
            _model_error = str(exc)


@app.route("/health")
def health() -> Response:
    if _model is not None:
        return jsonify({"status": "ok", "model_loaded": True, "classes": len(_idx_to_class) if _idx_to_class else 0})
    return jsonify({"status": "degraded", "model_loaded": False, "reason": "model not found — run train.py first"})


@app.route("/classify", methods=["POST"])
def classify() -> tuple[Response, int] | Response:
    t_start = time.perf_counter()

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field (base64 encoded)"}), 400

    t_decode = time.perf_counter()
    try:
        image_bytes = base64.b64decode(data["image"])
        image = Image.open(BytesIO(image_bytes))
    except Exception as e:
        return jsonify({"error": f"Could not decode image: {e}"}), 400
    t_decode = time.perf_counter() - t_decode

    if _model is None or _idx_to_class is None:
        return jsonify({"error": "Model not loaded — run train.py first"}), 503

    t_infer = time.perf_counter()
    label, confidence, low_confidence = predict(_model, _idx_to_class, image)
    t_infer = time.perf_counter() - t_infer

    t_total = time.perf_counter() - t_start
    logger.info(
        "[timing] POST /classify  decode=%.3fs  infer=%.3fs  total=%.3fs",
        t_decode, t_infer, t_total,
    )
    return jsonify(
        {"label": label, "confidence": confidence, "low_confidence": low_confidence}
    )


@app.route("/classify_batch", methods=["POST"])
def classify_batch() -> tuple[Response, int] | Response:
    """Classify multiple card images in a single batched forward pass.

    Request body:  {"images": ["<base64>", "<base64>", ...]}
    Response body: {"results": [{"label": "KH", "confidence": 0.97, "low_confidence": false}, ...]}
    """
    t_start = time.perf_counter()

    data = request.get_json(silent=True)
    if not data or "images" not in data:
        return jsonify({"error": "Missing 'images' field (list of base64 encoded images)"}), 400

    images_raw = data["images"]
    if not isinstance(images_raw, list) or len(images_raw) == 0:
        return jsonify({"error": "'images' must be a non-empty list of base64 encoded images"}), 400

    t_decode = time.perf_counter()
    images: list[Image.Image] = []
    for i, encoded in enumerate(images_raw):
        try:
            image_bytes = base64.b64decode(encoded)
            images.append(Image.open(BytesIO(image_bytes)))
        except Exception as e:
            return jsonify({"error": f"Could not decode image at index {i}: {e}"}), 400
    t_decode = time.perf_counter() - t_decode

    if _model is None or _idx_to_class is None:
        return jsonify({"error": "Model not loaded — run train.py first"}), 503

    t_infer = time.perf_counter()
    predictions = predict_batch(_model, _idx_to_class, images)
    t_infer = time.perf_counter() - t_infer

    t_total = time.perf_counter() - t_start
    logger.info(
        "[timing] POST /classify_batch  n=%d  decode=%.3fs  infer=%.3fs  total=%.3fs",
        len(images), t_decode, t_infer, t_total,
    )

    results = [
        {"label": label, "confidence": conf, "low_confidence": low}
        for label, conf, low in predictions
    ]
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
