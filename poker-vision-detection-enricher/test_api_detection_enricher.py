"""Smoke test for detection enricher API."""

import base64
import io

from api import app
from fastapi.testclient import TestClient
from PIL import Image


def _dummy_image_base64() -> str:
    image = Image.new("RGB", (120, 120), color="green")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_enrich() -> None:
    client = TestClient(app)
    payload = {
        "image_base64": _dummy_image_base64(),
        "detections": [
            {"class": "card", "bbox": [10, 10, 50, 80], "confidence": 0.99},
            {"class": "chip_stack", "bbox": [55, 10, 95, 50], "confidence": 0.88},
            {"class": "dealer_button", "bbox": [10, 85, 50, 115], "confidence": 0.95},
        ],
    }
    response = client.post("/enrich", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "objects" in body
    assert len(body["objects"]) == 3
