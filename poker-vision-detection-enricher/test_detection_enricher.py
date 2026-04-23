"""
Test script for DetectionEnricher
"""

from unittest.mock import MagicMock, patch

from detection_enricher import DetectionEnricher
from PIL import Image


def test_enricher_emits_confidence_metadata_by_processing_type() -> None:
    config = {
        "processing": {
            "card": "classify",
            "chip_stack": "ocr",
            "dealer_button": "spatial",
        },
        "save_snips": False,
    }
    enricher = DetectionEnricher(config)

    image = Image.new("RGB", (200, 200), color="green")
    detections = [
        {"class": "card", "bbox": [10, 10, 60, 90], "confidence": 0.91},
        {"class": "chip_stack", "bbox": [70, 10, 120, 60], "confidence": 0.84},
        {
            "class": "dealer_button",
            "bbox": [130, 10, 180, 60],
            "confidence": 0.88,
        },
        {"class": "unknown_ui", "bbox": [10, 100, 40, 140], "confidence": 0.75},
    ]

    mock_response = MagicMock()
    mock_response.json.return_value = {"label": "Ah", "confidence": 0.93}
    mock_response.raise_for_status.return_value = None

    # Mock run_ocr so the unit test does not trigger an EasyOCR model download.
    # run_ocr now returns (text, confidence) tuple.
    # Mock httpx.post so the unit test does not require a running classifier service.
    with (
        patch("detection_enricher.run_ocr", return_value=("450", 0.88)),
        patch("detection_enricher.httpx.post", return_value=mock_response),
    ):
        result = enricher.enrich(image, detections)
    objects = result["objects"]
    assert len(objects) == 4

    card_obj = next(obj for obj in objects if obj["class_name"] == "card")
    assert "classification" in card_obj
    assert card_obj["classification"] == "Ah"
    assert "classification_conf" in card_obj
    assert card_obj["classification_conf"] == 0.93

    chip_obj = next(obj for obj in objects if obj["class_name"] == "chip_stack")
    assert "ocr_text" in chip_obj
    assert chip_obj["ocr_text"] == "450"
    assert "ocr_conf" in chip_obj
    assert chip_obj["ocr_conf"] == 0.88, (
        "ocr_conf should be the real Tesseract confidence, not the hardcoded default"
    )

    dealer_obj = next(obj for obj in objects if obj["class_name"] == "dealer_button")
    assert "spatial_info" in dealer_obj
    assert "spatial_conf" in dealer_obj

    unsupported_obj = next(obj for obj in objects if obj["class_name"] == "unknown_ui")
    assert unsupported_obj["processing"] == "none"
    assert "classification_conf" not in unsupported_obj
    assert "ocr_conf" not in unsupported_obj
    assert "spatial_conf" not in unsupported_obj

    for obj in objects:
        assert "class_name" in obj
        assert "bbox_xyxy" in obj
        assert "confidence" in obj


def test_classifier_failure_falls_back_to_empty_label_and_default_conf() -> None:
    config = {
        "processing": {"card": "classify"},
        "save_snips": False,
        "default_classification_conf": 0.65,
    }
    enricher = DetectionEnricher(config)
    image = Image.new("RGB", (200, 200), color="green")
    detections = [{"class": "card", "bbox": [10, 10, 60, 90], "confidence": 0.91}]

    with patch(
        "detection_enricher.httpx.post",
        side_effect=Exception("connection refused"),
    ):
        result = enricher.enrich(image, detections)

    card_obj = result["objects"][0]
    assert card_obj["classification"] == ""
    assert card_obj["classification_conf"] == 0.65


if __name__ == "__main__":
    test_enricher_emits_confidence_metadata_by_processing_type()
