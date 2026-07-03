"""
Unit tests for the card classifier API, including the batch endpoint.

These tests use a mock model to avoid requiring the trained model.pt file.
Run from the poker-vision-card-classifier directory:
    uv run python -m pytest test_api.py -v
"""

from __future__ import annotations

import base64
import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Import the Flask app and helpers — we patch the model loading so no
# model.pt file is required to run these tests.
import api


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_image(color: str = "red", size: tuple[int, int] = (50, 70)) -> bytes:
    """Return PNG bytes for a small test image."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _encode_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("utf-8")


@pytest.fixture
def mock_model_loaded():
    """Patch the module-level model/idx_to_class so routes work without model.pt."""
    mock_model = MagicMock()
    fake_classes = {str(i): f"C{i}" for i in range(52)}
    with patch.object(api, "_model", mock_model), \
         patch.object(api, "_idx_to_class", fake_classes), \
         patch.object(api, "_model_error", None):
        yield mock_model, fake_classes


@pytest.fixture
def client(mock_model_loaded):
    """Flask test client with a mocked model."""
    app = api.app
    app.config["TESTING"] = True
    return app.test_client()


# ── predict_batch unit tests ─────────────────────────────────────────────────


class TestPredictBatch:
    """Unit tests for the predict_batch function (no HTTP, no Flask)."""

    def test_empty_list_returns_empty(self):
        mock_model = MagicMock()
        result = api.predict_batch(mock_model, {"0": "TS"}, [])
        assert result == []
        mock_model.assert_not_called()

    def test_single_image_returns_one_result(self):
        """A single image should produce one (label, conf, low_conf) tuple."""
        import torch

        # Mock model returns logits for 2 classes; softmax → class 1 wins
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.1, 5.0]])

        img = Image.new("RGB", (60, 80), color="blue")
        results = api.predict_batch(mock_model, {"0": "Fold", "1": "Call"}, [img])

        assert len(results) == 1
        label, conf, low = results[0]
        assert label == "Call"
        assert 0.0 <= conf <= 1.0
        assert isinstance(low, bool)

    def test_multiple_images_returns_one_result_each(self):
        """N images should produce N results in a single forward pass."""
        import torch

        mock_model = MagicMock()
        # 3 images, 2 classes — all predict class 1
        mock_model.return_value = torch.tensor([
            [0.1, 5.0],
            [0.2, 4.0],
            [0.3, 3.0],
        ])

        images = [Image.new("RGB", (60, 80), color=c) for c in ["red", "green", "blue"]]
        results = api.predict_batch(mock_model, {"0": "Fold", "1": "Call"}, images)

        assert len(results) == 3
        for label, conf, low in results:
            assert label == "Call"
            assert 0.0 <= conf <= 1.0
            assert isinstance(low, bool)

    def test_batch_calls_model_once(self):
        """The model forward pass should be called exactly once for N images."""
        import torch

        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.1, 5.0], [0.2, 4.0]])

        images = [Image.new("RGB", (60, 80)), Image.new("RGB", (60, 80))]
        api.predict_batch(mock_model, {"0": "A", "1": "B"}, images)

        assert mock_model.call_count == 1

    def test_low_confidence_flag(self):
        """low_confidence should be True when confidence < 0.70 threshold."""
        import torch

        mock_model = MagicMock()
        # Logits that produce ~50/50 split → low confidence
        mock_model.return_value = torch.tensor([[1.0, 1.01]])

        img = Image.new("RGB", (60, 80))
        results = api.predict_batch(mock_model, {"0": "A", "1": "B"}, [img])

        label, conf, low = results[0]
        assert low is True
        assert conf < 0.70


# ── /classify_batch endpoint tests ────────────────────────────────────────────


class TestClassifyBatchEndpoint:
    """HTTP-level tests for the POST /classify_batch endpoint."""

    def test_batch_success(self, client, mock_model_loaded):
        """A valid batch request returns 200 with results for each image."""
        import torch

        mock_model, fake_classes = mock_model_loaded
        mock_model.return_value = torch.tensor([
            [0.1, 5.0],
            [0.2, 4.0],
        ])

        images_b64 = [_encode_b64(_make_image("red")), _encode_b64(_make_image("blue"))]
        resp = client.post("/classify_batch", json={"images": images_b64})

        assert resp.status_code == 200
        data = resp.get_json()
        assert "results" in data
        assert len(data["results"]) == 2
        for item in data["results"]:
            assert "label" in item
            assert "confidence" in item
            assert "low_confidence" in item

    def test_batch_single_image(self, client, mock_model_loaded):
        """A batch with one image should still work (edge case)."""
        import torch

        mock_model, _ = mock_model_loaded
        mock_model.return_value = torch.tensor([[0.1, 5.0]])

        resp = client.post(
            "/classify_batch",
            json={"images": [_encode_b64(_make_image())]},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["results"]) == 1

    def test_batch_missing_images_field(self, client, mock_model_loaded):
        """Missing 'images' field returns 400."""
        resp = client.post("/classify_batch", json={})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_batch_empty_images_list(self, client, mock_model_loaded):
        """Empty images list returns 400."""
        resp = client.post("/classify_batch", json={"images": []})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_batch_images_not_a_list(self, client, mock_model_loaded):
        """'images' that is not a list returns 400."""
        resp = client.post("/classify_batch", json={"images": "not a list"})
        assert resp.status_code == 400

    def test_batch_invalid_base64(self, client, mock_model_loaded):
        """An invalid base64 string returns 400 with a helpful error."""
        resp = client.post(
            "/classify_batch",
            json={"images": ["not_valid_base64!!!"]},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        assert "index 0" in data["error"]

    def test_batch_model_not_loaded(self, mock_model_loaded):
        """When the model is not loaded, returns 503."""
        # Patch _model_error to a non-None value so the before_request
        # handler does not attempt to reload the model.
        with patch.object(api, "_model", None), \
             patch.object(api, "_idx_to_class", None), \
             patch.object(api, "_model_error", "model not found"):
            app = api.app
            app.config["TESTING"] = True
            client = app.test_client()
            resp = client.post(
                "/classify_batch",
                json={"images": [_encode_b64(_make_image())]},
            )
            assert resp.status_code == 503
            assert "error" in resp.get_json()

    def test_batch_results_order_matches_input_order(self, client, mock_model_loaded):
        """Results must be in the same order as the input images."""
        import torch

        mock_model, _ = mock_model_loaded
        # 3 images: class 0, class 1, class 0
        mock_model.return_value = torch.tensor([
            [5.0, 0.1],
            [0.1, 5.0],
            [5.0, 0.1],
        ])

        images_b64 = [
            _encode_b64(_make_image("red")),
            _encode_b64(_make_image("green")),
            _encode_b64(_make_image("blue")),
        ]
        resp = client.post("/classify_batch", json={"images": images_b64})
        assert resp.status_code == 200
        results = resp.get_json()["results"]
        assert len(results) == 3
        assert results[0]["label"] == "C0"
        assert results[1]["label"] == "C1"
        assert results[2]["label"] == "C0"


# ── /classify (single) still works ────────────────────────────────────────────


class TestClassifyEndpointStillWorks:
    """Sanity check that the existing single-image endpoint is unaffected."""

    def test_single_classify_success(self, client, mock_model_loaded):
        import torch

        mock_model, _ = mock_model_loaded
        mock_model.return_value = torch.tensor([[0.1, 5.0]])

        resp = client.post(
            "/classify",
            json={"image": _encode_b64(_make_image())},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "label" in data
        assert "confidence" in data
        assert "low_confidence" in data
