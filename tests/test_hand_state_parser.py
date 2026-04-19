from hand_state_parser import build_hand_state


def test_build_hand_state_uses_enriched_values() -> None:
    enriched_payload = {
        "objects": [
            {"class_name": "holecard", "classification": "Ah"},
            {"class_name": "holecard", "classification": "Kd"},
            {"class_name": "blinds", "ocr_text": "50/100"},
            {"class_name": "chip_stack", "ocr_text": "3,250"},
            {"class_name": "pot", "ocr_text": "Pot: 275"},
            {"class_name": "max_bet", "ocr_text": "100"},
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
