"""
Unit tests for normalize_response.
"""

from poker_vision.detect.normalize import normalize_response

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RAW_RESPONSE = {
    "predictions": [
        {
            "x": 100.0,
            "y": 200.0,
            "width": 40.0,
            "height": 60.0,
            "confidence": 0.92,
            "class": "Ace_Spades",
            "class_id": 0,
        },
        {
            "x": 300.0,
            "y": 150.0,
            "width": 50.0,
            "height": 70.0,
            "confidence": 0.45,  # below 0.50 threshold — should be filtered
            "class": "Two_Hearts",
            "class_id": 1,
        },
        {
            "x": 500.0,
            "y": 400.0,
            "width": 60.0,
            "height": 80.0,
            "confidence": 0.75,
            "class": "King_Clubs",
            "class_id": 12,
        },
    ]
}

COMMON_KWARGS = dict(
    source_image="/path/to/img.png",
    image_width=1280,
    image_height=720,
    api_base="https://detect.roboflow.com",
    project="poker-cards",
    version=3,
    confidence_threshold=0.50,
    overlap_threshold=0.50,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_normalize_top_level_keys():
    result = normalize_response(RAW_RESPONSE, **COMMON_KWARGS)
    for key in ("source_image", "image_width", "image_height", "model", "detections"):
        assert key in result, f"Missing key: {key}"


def test_normalize_model_metadata():
    result = normalize_response(RAW_RESPONSE, **COMMON_KWARGS)
    model = result["model"]
    assert model["provider"] == "roboflow"
    assert model["api_base"] == "https://detect.roboflow.com"
    assert model["project"] == "poker-cards"
    assert model["version"] == "3"
    assert model["requested_confidence_threshold"] == 0.50
    assert model["requested_overlap_threshold"] == 0.50


def test_normalize_filters_low_confidence():
    result = normalize_response(RAW_RESPONSE, **COMMON_KWARGS)
    class_names = [d["class_name"] for d in result["detections"]]
    # Two_Hearts has confidence 0.45 < 0.50 and should be filtered out
    assert "Two_Hearts" not in class_names


def test_normalize_keeps_high_confidence():
    result = normalize_response(RAW_RESPONSE, **COMMON_KWARGS)
    class_names = [d["class_name"] for d in result["detections"]]
    assert "Ace_Spades" in class_names
    assert "King_Clubs" in class_names
    assert len(result["detections"]) == 2


def test_normalize_detection_schema():
    result = normalize_response(RAW_RESPONSE, **COMMON_KWARGS)
    det = result["detections"][0]  # Ace_Spades
    assert det["class_name"] == "Ace_Spades"
    assert det["class_id"] == 0
    assert isinstance(det["confidence"], float)
    assert len(det["bbox_xywh_center"]) == 4
    assert len(det["bbox_xyxy"]) == 4


def test_normalize_bbox_values():
    result = normalize_response(RAW_RESPONSE, **COMMON_KWARGS)
    det_ace = next(d for d in result["detections"] if d["class_name"] == "Ace_Spades")
    # x=100, y=200, w=40, h=60 -> xyxy: [80, 170, 120, 230]
    assert det_ace["bbox_xyxy"] == [80, 170, 120, 230]
    assert det_ace["bbox_xywh_center"] == [100, 200, 40, 60]


def test_normalize_source_and_dimensions():
    result = normalize_response(RAW_RESPONSE, **COMMON_KWARGS)
    assert result["source_image"] == "/path/to/img.png"
    assert result["image_width"] == 1280
    assert result["image_height"] == 720


def test_normalize_empty_predictions():
    result = normalize_response({"predictions": []}, **COMMON_KWARGS)
    assert result["detections"] == []


def test_normalize_all_below_threshold():
    raw = {
        "predictions": [
            {
                "x": 10.0,
                "y": 10.0,
                "width": 10.0,
                "height": 10.0,
                "confidence": 0.1,
                "class": "X",
                "class_id": 0,
            },
        ]
    }
    result = normalize_response(raw, **COMMON_KWARGS)
    assert result["detections"] == []


def test_normalize_version_always_string():
    result = normalize_response(RAW_RESPONSE, **{**COMMON_KWARGS, "version": 5})
    assert result["model"]["version"] == "5"


def test_normalize_missing_class_id():
    raw = {
        "predictions": [
            {
                "x": 50.0,
                "y": 50.0,
                "width": 20.0,
                "height": 20.0,
                "confidence": 0.80,
                "class": "Unknown",
            },
        ]
    }
    result = normalize_response(raw, **COMMON_KWARGS)
    assert result["detections"][0]["class_id"] is None
