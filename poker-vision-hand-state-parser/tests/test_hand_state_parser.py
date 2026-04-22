from hand_state_parser import build_hand_state, build_hand_state_with_diagnostics

# ---------------------------------------------------------------------------
# Basic enriched-value extraction
# ---------------------------------------------------------------------------


def test_build_hand_state_uses_enriched_values() -> None:
    enriched_payload = {
        "objects": [
            {
                "class_name": "holecard",
                "classification": "Ah",
                "confidence": 0.95,
                "classification_conf": 0.96,
            },
            {
                "class_name": "holecard",
                "classification": "Kd",
                "confidence": 0.93,
                "classification_conf": 0.95,
            },
            {"class_name": "blinds", "ocr_text": "50/100", "ocr_conf": 0.90},
            {"class_name": "chip_stack", "ocr_text": "3,250", "ocr_conf": 0.92},
            {"class_name": "pot", "ocr_text": "Pot: 275", "ocr_conf": 0.90},
            {"class_name": "max_bet", "ocr_text": "100", "ocr_conf": 0.90},
        ]
    }

    hand_state = build_hand_state(enriched_payload)

    assert hand_state == {
        "hero_cards": ["Ah", "Kd"],
        "position": "BTN",
        "big_blind": 100,
        "small_blind": 50,
        "hero_stack": 3250,
        "pot": 275,
        "amount_to_call": 100,
        "action_history": [],
        "is_hero_turn": True,
        "hero_folded": False,
    }


def test_build_hand_state_falls_back_to_safe_defaults() -> None:
    hand_state = build_hand_state({"objects": [{"class_name": "holecard"}]})

    assert hand_state["hero_cards"] == ["Ah", "Kd"]
    assert hand_state["position"] == "BTN"
    assert hand_state["big_blind"] == 100
    assert hand_state["small_blind"] == 50
    assert hand_state["hero_stack"] == 3000
    assert hand_state["pot"] == 150
    assert hand_state["amount_to_call"] == 0
    assert hand_state["is_hero_turn"] is True
    assert hand_state["hero_folded"] is False


# ---------------------------------------------------------------------------
# Confidence gating
# ---------------------------------------------------------------------------


def test_low_confidence_cards_fallback_to_default_hero_cards() -> None:
    enriched_payload = {
        "objects": [
            {
                "class_name": "holecard",
                "classification": "Ah",
                "confidence": 0.50,  # below _MIN_DETECTION_FOR_CARDS (0.60)
                "classification_conf": 0.95,
            },
            {
                "class_name": "holecard",
                "classification": "Kd",
                "confidence": 0.50,
                "classification_conf": 0.95,
            },
        ]
    }

    hand_state = build_hand_state(enriched_payload)
    assert hand_state["hero_cards"] == ["Ah", "Kd"]


def test_low_classification_conf_cards_fallback() -> None:
    enriched_payload = {
        "objects": [
            {
                "class_name": "holecard",
                "classification": "Ah",
                "confidence": 0.90,
                "classification_conf": 0.60,  # below _MIN_CLASSIFICATION_FOR_CARDS (0.70)
            },
            {
                "class_name": "holecard",
                "classification": "Kd",
                "confidence": 0.90,
                "classification_conf": 0.60,
            },
        ]
    }

    hand_state = build_hand_state(enriched_payload)
    assert hand_state["hero_cards"] == ["Ah", "Kd"]


def test_mixed_confidence_uses_fallback_only_for_rejected_fields() -> None:
    enriched_payload = {
        "objects": [
            {
                "class_name": "holecard",
                "classification": "Ah",
                "confidence": 0.90,
                "classification_conf": 0.90,
            },
            {
                "class_name": "holecard",
                "classification": "Kd",
                "confidence": 0.85,
                "classification_conf": 0.85,
            },
            {
                "class_name": "blinds",
                "ocr_text": "25/50",
                "confidence": 0.95,
                "ocr_conf": 0.90,
            },
            {
                "class_name": "chip_stack",
                "ocr_text": "4200",
                "confidence": 0.90,
                "ocr_conf": 0.90,
            },
            {
                "class_name": "pot",
                "ocr_text": "999",
                "confidence": 0.52,  # below usable threshold — field rejected
                "ocr_conf": 0.95,
            },
            {
                "class_name": "max_bet",
                "ocr_text": "150",
                "confidence": 0.90,
                "ocr_conf": 0.90,
            },
        ]
    }

    hand_state = build_hand_state(enriched_payload)

    assert hand_state["hero_cards"] == ["Ah", "Kd"]
    assert hand_state["small_blind"] == 25
    assert hand_state["big_blind"] == 50
    assert hand_state["hero_stack"] == 4200
    assert hand_state["pot"] == 150  # fallback — pot confidence rejected
    assert hand_state["amount_to_call"] == 150


def test_ocr_text_with_comma_parses_correctly() -> None:
    enriched_payload = {
        "objects": [
            {"class_name": "chip_stack", "ocr_text": "1,500", "ocr_conf": 0.90},
        ]
    }
    hand_state = build_hand_state(enriched_payload)
    assert hand_state["hero_stack"] == 1500


def test_ocr_text_with_label_prefix_parses_integer() -> None:
    enriched_payload = {
        "objects": [
            {"class_name": "total_pot", "ocr_text": "Total Pot: 300", "ocr_conf": 0.88},
        ]
    }
    hand_state = build_hand_state(enriched_payload)
    assert hand_state["pot"] == 300


# ---------------------------------------------------------------------------
# Pot source priority
# ---------------------------------------------------------------------------


def test_pot_prefers_total_pot_over_pot() -> None:
    enriched_payload = {
        "objects": [
            {"class_name": "total_pot", "ocr_text": "400", "ocr_conf": 0.90},
            {"class_name": "pot", "ocr_text": "200", "ocr_conf": 0.90},
        ]
    }
    hand_state = build_hand_state(enriched_payload)
    assert hand_state["pot"] == 400


def test_amount_to_call_prefers_bet_over_max_bet() -> None:
    enriched_payload = {
        "objects": [
            {"class_name": "bet", "ocr_text": "100", "ocr_conf": 0.90},
            {"class_name": "max_bet", "ocr_text": "200", "ocr_conf": 0.90},
        ]
    }
    hand_state = build_hand_state(enriched_payload)
    assert hand_state["amount_to_call"] == 100


# ---------------------------------------------------------------------------
# Hero turn detection
# ---------------------------------------------------------------------------


def test_fold_button_sets_is_hero_turn() -> None:
    enriched_payload = {
        "objects": [
            {"class_name": "fold_button", "confidence": 0.92},
        ]
    }
    hand_state = build_hand_state(enriched_payload)
    assert hand_state["is_hero_turn"] is True


def test_no_action_controls_still_defaults_hero_turn_true() -> None:
    hand_state = build_hand_state({"objects": []})
    assert hand_state["is_hero_turn"] is True


# ---------------------------------------------------------------------------
# Blinds parsing
# ---------------------------------------------------------------------------


def test_blinds_slash_format() -> None:
    enriched_payload = {
        "objects": [
            {"class_name": "blinds", "ocr_text": "100/200", "ocr_conf": 0.90},
        ]
    }
    hand_state = build_hand_state(enriched_payload)
    assert hand_state["small_blind"] == 100
    assert hand_state["big_blind"] == 200


def test_blinds_invalid_order_falls_back() -> None:
    """If sb >= bb, the blinds value is considered invalid and defaults are used."""
    enriched_payload = {
        "objects": [
            {"class_name": "blinds", "ocr_text": "200/100", "ocr_conf": 0.90},
        ]
    }
    hand_state = build_hand_state(enriched_payload)
    assert hand_state["small_blind"] == 50
    assert hand_state["big_blind"] == 100


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_build_hand_state_with_diagnostics_reports_field_sources() -> None:
    enriched_payload = {
        "objects": [
            {
                "class_name": "holecard",
                "classification": "Ah",
                "confidence": 0.91,
                "classification_conf": 0.90,
            },
            {
                "class_name": "holecard",
                "classification": "Kd",
                "confidence": 0.91,
                "classification_conf": 0.90,
            },
            {
                "class_name": "blinds",
                "ocr_text": "50/100",
                "confidence": 0.92,
                "ocr_conf": 0.91,
            },
        ]
    }

    hand_state, diagnostics = build_hand_state_with_diagnostics(enriched_payload)

    assert hand_state["hero_cards"] == ["Ah", "Kd"]
    assert diagnostics["hero_cards"]["fallback_used"] is False
    assert diagnostics["small_blind"]["source"] == "blinds"
    assert diagnostics["pot"]["fallback_used"] is True


def test_diagnostics_fallback_used_when_confidence_rejected() -> None:
    enriched_payload = {
        "objects": [
            {
                "class_name": "chip_stack",
                "ocr_text": "5000",
                "confidence": 0.90,
                "ocr_conf": 0.40,  # below usable threshold
            },
        ]
    }
    _, diagnostics = build_hand_state_with_diagnostics(enriched_payload)
    assert diagnostics["hero_stack"]["fallback_used"] is True


def test_diagnostics_band_trusted_for_high_confidence() -> None:
    enriched_payload = {
        "objects": [
            {"class_name": "chip_stack", "ocr_text": "2500", "ocr_conf": 0.95},
        ]
    }
    _, diagnostics = build_hand_state_with_diagnostics(enriched_payload)
    assert diagnostics["hero_stack"]["band"] == "trusted"
    assert diagnostics["hero_stack"]["fallback_used"] is False


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


def test_invalid_payload_raises_value_error() -> None:
    import pytest

    with pytest.raises(ValueError, match="objects"):
        build_hand_state({"wrong_key": []})


def test_non_dict_objects_are_skipped() -> None:
    enriched_payload = {
        "objects": [
            None,
            "bad",
            42,
            {"class_name": "chip_stack", "ocr_text": "1000", "ocr_conf": 0.90},
        ]
    }
    hand_state = build_hand_state(enriched_payload)
    assert hand_state["hero_stack"] == 1000
