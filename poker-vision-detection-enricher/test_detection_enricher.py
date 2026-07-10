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
    mock_response.json.return_value = {"results": [{"label": "Ah", "confidence": 0.93}]}
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

    with patch(
        "detection_enricher.httpx.post", return_value=mock_response
    ) as mock_post:
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
        {"class": "card", "bbox": [70, 10, 120, 90], "confidence": 0.91},  # card 1
        {"class": "dealer_button", "bbox": [130, 10, 160, 40], "confidence": 0.88},
        {"class": "card", "bbox": [170, 10, 220, 90], "confidence": 0.85},  # card 2
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

    with patch(
        "detection_enricher.httpx.post", side_effect=Exception("connection refused")
    ):
        result = enricher.enrich(image, detections)

    for obj in result["objects"]:
        assert obj["classification"] == ""
        assert obj["classification_conf"] == 0.65


# ---------------------------------------------------------------------------
# _halo_score unit tests
# ---------------------------------------------------------------------------


def _make_player_crop(
    width: int,
    height: int,
    *,
    top_v: int = 50,
    card_v: int = 170,
) -> Image.Image:
    """Synthetic player bbox: dark top band, bright card-back mid band.

    top_v:  V value for rows 0..int(h*0.28)  (the halo zone)
    card_v: V value for rows int(h*0.30)..int(h*0.65)  (card backs)
    """
    # Build an HSV image then convert to RGB for the crop API.
    # PIL's HSV mode: H=0, S=0, V=value → grey.
    img = Image.new("RGB", (width, height), (0, 0, 0))
    pixels = img.load()
    for y in range(height):
        if y < int(height * 0.28):
            v = top_v
        elif y < int(height * 0.65):
            v = card_v
        else:
            v = 30  # name/stack area — dark
        rgb = Image.new("HSV", (1, 1), (0, 0, v)).convert("RGB").getpixel((0, 0))
        for x in range(width):
            pixels[x, y] = rgb
    return img


def _halo_enricher() -> DetectionEnricher:
    return DetectionEnricher({"processing": {}, "save_snips": False})


def test_halo_score_bright_top_band_returns_positive_score() -> None:
    """Crop with bright top band (V=230) and dim card band → score > 0."""
    enricher = _halo_enricher()
    crop = _make_player_crop(90, 105, top_v=230, card_v=140)
    score = enricher._halo_score(crop)
    assert score > 0.0, f"Expected positive score for halo crop, got {score}"


def test_halo_score_uniform_crop_returns_zero() -> None:
    """Crop with identical brightness everywhere → no differential → score ≈ 0."""
    enricher = _halo_enricher()
    # Uniform mid-brightness: top and card bands equally bright → no signal
    crop = _make_player_crop(90, 105, top_v=170, card_v=170)
    score = enricher._halo_score(crop)
    assert score == 0.0, f"Expected 0.0 for uniform crop, got {score}"


def test_halo_score_card_backs_only_no_false_positive() -> None:
    """Bright pixels in card band only, dark top band → score = 0."""
    enricher = _halo_enricher()
    # Dark top, bright cards — the typical non-active player appearance
    crop = _make_player_crop(90, 105, top_v=80, card_v=220)
    score = enricher._halo_score(crop)
    assert score == 0.0, (
        f"Expected 0.0 when card band is brighter than top band, got {score}"
    )


def test_halo_score_too_small_returns_zero() -> None:
    """Image smaller than 8×8 must return 0.0 without error."""
    enricher = _halo_enricher()
    tiny = Image.new("RGB", (6, 6), (200, 200, 200))
    assert enricher._halo_score(tiny) == 0.0


def test_turn_active_assigned_to_correct_player() -> None:
    """Player with bright top band should get turn_active=True; other gets False."""
    config = {
        "processing": {"player_other": "halo"},
        "save_snips": False,
        "turn_halo_threshold": 0.05,
        "turn_halo_ambiguity_delta": 0.02,
    }
    enricher = DetectionEnricher(config)

    # Image: left half dark (no halo), right half has a bright top strip (halo).
    img_w, img_h = 200, 114
    image = Image.new("RGB", (img_w, img_h), (20, 20, 20))
    pixels = image.load()
    # Add bright top rows only in the right half (x >= 100)
    bright_rgb = Image.new("HSV", (1, 1), (0, 0, 230)).convert("RGB").getpixel((0, 0))
    for y in range(int(img_h * 0.12), int(img_h * 0.28)):
        for x in range(100, 200):
            pixels[x, y] = bright_rgb

    detections = [
        {
            "class": "player_other",
            "bbox": [0, 0, 99, img_h],
            "confidence": 0.90,
        },  # no halo
        {
            "class": "player_other",
            "bbox": [100, 0, 199, img_h],
            "confidence": 0.88,
        },  # halo
    ]

    result = enricher.enrich(image, detections)
    objects = result["objects"]

    no_halo = next(o for o in objects if o["bbox_xyxy"][0] == 0)
    has_halo = next(o for o in objects if o["bbox_xyxy"][0] == 100)

    assert has_halo.get("turn_active") is True, (
        f"Player with bright top band should have turn_active=True, got {has_halo.get('turn_active')}"
    )
    assert no_halo.get("turn_active") is False, (
        f"Player without halo should have turn_active=False, got {no_halo.get('turn_active')}"
    )
    assert has_halo["turn_halo_score"] > no_halo["turn_halo_score"]


def test_turn_active_ambiguous_scores_no_active() -> None:
    """When best and second scores are within ambiguity_delta, no player is active."""
    config = {
        "processing": {"player_other": "halo"},
        "save_snips": False,
        "turn_halo_threshold": 0.05,
        "turn_halo_ambiguity_delta": 0.10,  # large delta — forces ambiguous
    }
    enricher = DetectionEnricher(config)

    img_w, img_h = 200, 114
    image = Image.new("RGB", (img_w, img_h), (20, 20, 20))
    pixels = image.load()
    # Both halves get identical bright top rows
    bright_rgb = Image.new("HSV", (1, 1), (0, 0, 220)).convert("RGB").getpixel((0, 0))
    for y in range(int(img_h * 0.12), int(img_h * 0.28)):
        for x in range(img_w):
            pixels[x, y] = bright_rgb

    detections = [
        {"class": "player_other", "bbox": [0, 0, 99, img_h], "confidence": 0.90},
        {"class": "player_other", "bbox": [100, 0, 199, img_h], "confidence": 0.88},
    ]

    result = enricher.enrich(image, detections)
    for obj in result["objects"]:
        assert obj.get("turn_active") is False, (
            f"Ambiguous scores should leave all turn_active=False, got {obj.get('turn_active')}"
        )


if __name__ == "__main__":
    test_enricher_emits_confidence_metadata_by_processing_type()
