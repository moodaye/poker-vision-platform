"""
Test script for DetectionEnricher
"""

from detection_enricher import DetectionEnricher
from PIL import Image


def test_enricher() -> None:
    config = {
        "processing": {
            "card": "classify",
            "chip_stack": "ocr",
            "dealer_button": "spatial",
        },
        "save_snips": False,
    }
    enricher = DetectionEnricher(config)
    # Dummy image and detections
    image = Image.new("RGB", (200, 200), color="green")
    detections = [
        {"class": "card", "bbox": [10, 10, 60, 90]},
        {"class": "chip_stack", "bbox": [70, 10, 120, 60]},
        {"class": "dealer_button", "bbox": [130, 10, 180, 60]},
    ]
    result = enricher.enrich(image, detections)
    print(result)


if __name__ == "__main__":
    test_enricher()
