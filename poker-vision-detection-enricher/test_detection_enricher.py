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
    mock_response.json.return_value = {
        "results": [{"label": "Ah", "confidence": 0.93}]
    }
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


def test_bet_object_includes_player_name_in_enriched_output() -> None:
    config = {
        "processing": {
            "player_name": "ocr",
            "bet": "spatial",
        },
        "save_snips": False,
    }
    enricher = DetectionEnricher(config)

    image = Image.new("RGB", (220, 160), color="green")
    detections = [
        {"class": "player_name", "bbox": [10, 10, 110, 50], "confidence": 0.93},
        {"class": "bet", "bbox": [40, 70, 80, 110], "confidence": 0.89},
    ]

    with patch("detection_enricher.run_ocr", return_value=("Hero", 0.91)):
        result = enricher.enrich(image, detections)

    bet_obj = next(obj for obj in result["objects"] if obj["class_name"] == "bet")
    assert bet_obj["spatial_info"]["owner_player"] == "Hero"
    assert bet_obj["player_name"] == "Hero"


def test_batch_classify_multiple_cards_in_single_call() -> None:
    """Multiple card detections should be classified via a single batch HTTP call."""
    config = {"processing": {"card": "classify"}, "save_snips": False}
    enricher = DetectionEnricher(config)

    image = Image.new("RGB", (400, 200), color="green")
    detections = [
        {"class": "card", "bbox": [10, 10, 60, 90], "confidence": 0.91},
        {"class": "card", "bbox": [70, 10, 120, 90], "confidence": 0.88},
        {"class": "card", "bbox": [130, 10, 180, 90], "confidence": 0.85},
    ]

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"label": "Ah", "confidence": 0.95},
            {"label": "Kd", "confidence": 0.91},
            {"label": "8c", "confidence": 0.87},
        ]
    }
    mock_response.raise_for_status.return_value = None

    with patch("detection_enricher.httpx.post", return_value=mock_response) as mock_post:
        result = enricher.enrich(image, detections)

    # Should have made exactly one HTTP call (the batch call)
    assert mock_post.call_count == 1

    # The URL should be the batch endpoint
    call_args = mock_post.call_args
    assert "/classify_batch" in call_args[0][0]

    # All three cards should have their classification results mapped back
    objects = result["objects"]
    assert len(objects) == 3
    labels = [obj["classification"] for obj in objects]
    confs = [obj["classification_conf"] for obj in objects]
    assert labels == ["Ah", "Kd", "8c"]
    assert confs == [0.95, 0.91, 0.87]


def test_batch_classify_preserves_order_with_non_card_detections() -> None:
    """Card results must map back to the correct enriched objects even when
    non-card detections (OCR, spatial) are interleaved in the detection list."""
    config = {
        "processing": {
            "card": "classify",
            "chip_stack": "ocr",
            "dealer_button": "spatial",
        },
        "save_snips": False,
    }
    enricher = DetectionEnricher(config)

    image = Image.new("RGB", (400, 200), color="green")
    detections = [
        {"class": "chip_stack", "bbox": [10, 10, 60, 50], "confidence": 0.90},
        {"class": "card", "bbox": [70, 10, 120, 90], "confidence": 0.91},   # card 1
        {"class": "dealer_button", "bbox": [130, 10, 160, 40], "confidence": 0.88},
        {"class": "card", "bbox": [170, 10, 220, 90], "confidence": 0.85},   # card 2
    ]

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"label": "Ah", "confidence": 0.95},  # should map to card 1
            {"label": "Kd", "confidence": 0.91},  # should map to card 2
        ]
    }
    mock_response.raise_for_status.return_value = None

    with (
        patch("detection_enricher.run_ocr", return_value=("500", 0.90)),
        patch("detection_enricher.httpx.post", return_value=mock_response),
    ):
        result = enricher.enrich(image, detections)

    objects = result["objects"]
    assert len(objects) == 4

    # Card 1 is at index 1, card 2 is at index 3
    assert objects[1]["classification"] == "Ah"
    assert objects[1]["classification_conf"] == 0.95
    assert objects[3]["classification"] == "Kd"
    assert objects[3]["classification_conf"] == 0.91

    # Non-card objects should not have classification fields
    assert "classification" not in objects[0]  # chip_stack
    assert "classification" not in objects[2]  # dealer_button


def test_batch_classify_no_cards_makes_no_http_call() -> None:
    """When there are no card detections, no batch classify call should be made."""
    config = {
        "processing": {"chip_stack": "ocr", "dealer_button": "spatial"},
        "save_snips": False,
    }
    enricher = DetectionEnricher(config)

    image = Image.new("RGB", (200, 200), color="green")
    detections = [
        {"class": "chip_stack", "bbox": [10, 10, 60, 50], "confidence": 0.90},
        {"class": "dealer_button", "bbox": [70, 10, 100, 40], "confidence": 0.88},
    ]

    with (
        patch("detection_enricher.run_ocr", return_value=("500", 0.90)),
        patch("detection_enricher.httpx.post") as mock_post,
    ):
        result = enricher.enrich(image, detections)

    mock_post.assert_not_called()
    assert len(result["objects"]) == 2


def test_batch_classify_failure_returns_defaults_for_all_cards() -> None:
    """When the batch call fails, all cards should get empty label + default conf."""
    config = {
        "processing": {"card": "classify"},
        "save_snips": False,
        "default_classification_conf": 0.65,
    }
    enricher = DetectionEnricher(config)

    image = Image.new("RGB", (300, 200), color="green")
    detections = [
        {"class": "card", "bbox": [10, 10, 60, 90], "confidence": 0.91},
        {"class": "card", "bbox": [70, 10, 120, 90], "confidence": 0.85},
    ]

    with patch("detection_enricher.httpx.post", side_effect=Exception("connection refused")):
        result = enricher.enrich(image, detections)

    for obj in result["objects"]:
        assert obj["classification"] == ""
        assert obj["classification_conf"] == 0.65


if __name__ == "__main__":
    test_enricher_emits_confidence_metadata_by_processing_type()
