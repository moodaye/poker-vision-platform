"""
Test script for DetectionEnricher
"""

from unittest.mock import patch

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

    # Mock run_ocr so the unit test does not trigger an EasyOCR model download
    with patch("detection_enricher.run_ocr", return_value="450"):
        result = enricher.enrich(image, detections)
    objects = result["objects"]
    assert len(objects) == 4

    card_obj = next(obj for obj in objects if obj["class_name"] == "card")
    assert "classification" in card_obj
    assert "classification_conf" in card_obj

    chip_obj = next(obj for obj in objects if obj["class_name"] == "chip_stack")
    assert "ocr_text" in chip_obj
    assert "ocr_conf" in chip_obj

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


if __name__ == "__main__":
    test_enricher_emits_confidence_metadata_by_processing_type()
