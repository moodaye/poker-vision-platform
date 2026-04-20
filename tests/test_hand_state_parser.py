from hand_state_parser import build_hand_state, build_hand_state_with_diagnostics


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
            {
                "class_name": "chip_stack",
                "ocr_text": "3,250",
                "ocr_conf": 0.92,
            },
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


def test_low_confidence_cards_fallback_to_default_hero_cards() -> None:
    enriched_payload = {
        "objects": [
            {
                "class_name": "holecard",
                "classification": "Ah",
                "confidence": 0.50,
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
                "confidence": 0.52,
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
    assert hand_state["pot"] == 150  # fallback due to rejected low-confidence pot
    assert hand_state["amount_to_call"] == 150


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
