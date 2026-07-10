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
            {"class_name": "total_pot", "ocr_text": "Total Pot: 275", "ocr_conf": 0.90},
            {"class_name": "max_bet", "ocr_text": "100", "ocr_conf": 0.90},
        ]
    }

    hand_state = build_hand_state(enriched_payload)

    assert hand_state == {
        "schema_version": "2.2.0",
        "hand_phase": "preflop",
        "hero_cards": ["Ah", "Kd"],
        "hero_cards_visibility": "exposed",
        "position": "BTN",
        "hero_seat": "BTN",
        "action_on": "none",
        "big_blind": 100,
        "small_blind": 50,
        "hero_stack": 3250,
        "pot": 275,
        "amount_to_call": 100,
        "seats": [
            {
                "seat": "BTN",
                "is_hero": True,
                "player_name": None,
                "status": "waiting_turn",
                "stack": 3250,
                "is_folded": False,
                "is_all_in": None,
                "has_cards": True,
            },
            {
                "seat": "SB",
                "is_hero": False,
                "player_name": None,
                "status": "waiting_turn",
                "stack": None,
                "is_folded": None,
                "is_all_in": None,
                "has_cards": None,
            },
            {
                "seat": "BB",
                "is_hero": False,
                "player_name": None,
                "status": "waiting_turn",
                "stack": None,
                "is_folded": None,
                "is_all_in": None,
                "has_cards": None,
            },
        ],
        "tournament_status": {
            "current_blind_level": None,
            "small_blind_amount": 50,
            "big_blind_amount": 100,
            "ante_amount": 0,
            "seconds_until_next_level": None,
        },
        "action_history": [],
        "is_hero_turn": False,
        "hero_folded": False,
    }


def test_board_cards_set_hand_phase_to_postflop() -> None:
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
            {"class_name": "flop_card", "classification": "2c", "confidence": 0.90},
        ]
    }

    hand_state = build_hand_state(enriched_payload)
    assert hand_state["hand_phase"] == "postflop"


def test_build_hand_state_falls_back_to_safe_defaults() -> None:
    hand_state = build_hand_state({"objects": [{"class_name": "holecard"}]})

    assert hand_state["hero_cards"] == []
    assert hand_state["hero_cards_visibility"] == "not_exposed"
    assert hand_state["position"] == "BTN"
    assert hand_state["hero_seat"] == "BTN"
    assert hand_state["action_on"] == "none"
    assert hand_state["big_blind"] == 100
    assert hand_state["small_blind"] == 50
    assert hand_state["hero_stack"] == 3000
    assert hand_state["pot"] == 150
    assert hand_state["amount_to_call"] == 0
    assert hand_state["tournament_status"]["ante_amount"] == 0
    assert hand_state["is_hero_turn"] is False
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


def test_classifier_uppercase_labels_are_normalized() -> None:
    enriched_payload = {
        "objects": [
            {
                "class_name": "holecard",
                "classification": "AH",
                "confidence": 0.95,
                "classification_conf": 0.95,
            },
            {
                "class_name": "holecard",
                "classification": "KD",
                "confidence": 0.95,
                "classification_conf": 0.95,
            },
        ]
    }

    hand_state = build_hand_state(enriched_payload)
    assert hand_state["hero_cards"] == ["Ah", "Kd"]


def test_ten_notation_is_normalized_to_t_rank() -> None:
    enriched_payload = {
        "objects": [
            {
                "class_name": "holecard",
                "classification": "10H",
                "confidence": 0.95,
                "classification_conf": 0.95,
            },
            {
                "class_name": "holecard",
                "classification": "9D",
                "confidence": 0.95,
                "classification_conf": 0.95,
            },
        ]
    }

    hand_state = build_hand_state(enriched_payload)
    assert hand_state["hero_cards"] == ["Th", "9d"]


def test_hero_cards_are_ordered_left_to_right() -> None:
    enriched_payload = {
        "objects": [
            {
                "class_name": "holecard",
                "classification": "3H",
                "confidence": 0.95,
                "classification_conf": 0.95,
                "x": 800,
            },
            {
                "class_name": "holecard",
                "classification": "JS",
                "confidence": 0.95,
                "classification_conf": 0.95,
                "x": 700,
            },
        ]
    }

    hand_state = build_hand_state(enriched_payload)
    assert hand_state["hero_cards"] == ["Js", "3h"]
    assert hand_state["hero_cards_visibility"] == "exposed"


def test_hero_cards_are_ordered_left_to_right_with_bbox_list() -> None:
    enriched_payload = {
        "objects": [
            {
                "class_name": "holecard",
                "classification": "AD",
                "confidence": 0.95,
                "classification_conf": 0.95,
                "bbox": [901, 731, 987, 854],
            },
            {
                "class_name": "holecard",
                "classification": "KC",
                "confidence": 0.95,
                "classification_conf": 0.95,
                "bbox": [810, 731, 896, 854],
            },
        ]
    }

    hand_state = build_hand_state(enriched_payload)
    assert hand_state["hero_cards"] == ["Kc", "Ad"]
    assert hand_state["hero_cards_visibility"] == "exposed"


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
                "class_name": "total_pot",
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


def test_hero_stack_prefers_chip_stack_owned_by_hero_player() -> None:
    enriched_payload = {
        "objects": [
            {
                "class_name": "player_me",
                "spatial_info": {"position": "BTN", "hero_player": "Hero"},
                "spatial_conf": 0.90,
                "confidence": 0.90,
            },
            {
                "class_name": "chip_stack",
                "ocr_text": "4500",
                "ocr_conf": 0.95,
                "confidence": 0.95,
                "spatial_info": {"owner_player": "Villain"},
            },
            {
                "class_name": "chip_stack",
                "ocr_text": "2700",
                "ocr_conf": 0.90,
                "confidence": 0.90,
                "spatial_info": {"owner_player": "Hero"},
            },
        ]
    }

    hand_state = build_hand_state(enriched_payload)
    assert hand_state["hero_stack"] == 2700


def test_hero_stack_falls_back_to_best_stack_when_owner_unavailable() -> None:
    """No hero-owned candidate exists → use highest-confidence accepted stack."""
    enriched_payload = {
        "objects": [
            {
                "class_name": "player_me",
                "spatial_info": {"position": "BTN", "hero_player": "Hero"},
                "spatial_conf": 0.90,
                "confidence": 0.90,
            },
            {
                "class_name": "chip_stack",
                "ocr_text": "4100",
                "ocr_conf": 0.95,
                "confidence": 0.95,
                "spatial_info": {"owner_player": "Villain"},
            },
            {
                "class_name": "chip_stack",
                "ocr_text": "2800",
                "ocr_conf": 0.85,
                "confidence": 0.85,
            },
        ]
    }

    hand_state, diagnostics = build_hand_state_with_diagnostics(enriched_payload)
    assert hand_state["hero_stack"] == 4100
    assert diagnostics["hero_stack"]["source"] == "chip_stack"
    assert diagnostics["hero_stack"]["fallback_used"] is False


def test_hero_stack_owner_match_wins_over_higher_confidence_opponent() -> None:
    """Hero-owned stack is used even when opponent stack has higher OCR confidence."""
    enriched_payload = {
        "objects": [
            {
                "class_name": "player_me",
                "spatial_info": {"position": "BTN", "hero_player": "moodaye_"},
                "spatial_conf": 0.90,
                "confidence": 0.94,
            },
            # Opponent stacks — high OCR confidence
            {
                "class_name": "chip_stack",
                "ocr_text": "480",
                "ocr_conf": 0.96,
                "confidence": 0.91,
                "spatial_info": {"owner_player": "Weave"},
            },
            {
                "class_name": "chip_stack",
                "ocr_text": "480",
                "ocr_conf": 0.95,
                "confidence": 0.91,
                "spatial_info": {"owner_player": "Donna1212"},
            },
            # Hero stack — low OCR confidence but correct owner
            {
                "class_name": "chip_stack",
                "ocr_text": "500",
                "ocr_conf": 0.47,
                "confidence": 0.94,
                "spatial_info": {"owner_player": "moodaye_"},
            },
        ]
    }

    hand_state, diagnostics = build_hand_state_with_diagnostics(enriched_payload)
    assert hand_state["hero_stack"] == 500, (
        "Owner match must take priority; hero stack must not be replaced by an opponent's value"
    )
    assert diagnostics["hero_stack"]["source"] == "chip_stack.owner_player"
    assert diagnostics["hero_stack"]["fallback_used"] is False
    assert (
        diagnostics["hero_stack"]["warning"] is not None
    )  # low-confidence warning present


def test_hero_stack_low_confidence_owner_match_emits_warning() -> None:
    """Hero-owned stack below usable threshold: accepted with warning, not silently dropped."""
    enriched_payload = {
        "objects": [
            {
                "class_name": "player_me",
                "spatial_info": {"position": "BTN", "hero_player": "Hero"},
                "spatial_conf": 0.90,
                "confidence": 0.90,
            },
            {
                "class_name": "chip_stack",
                "ocr_text": "1200",
                "ocr_conf": 0.40,  # below _USABLE_THRESHOLD (0.55)
                "confidence": 0.92,
                "spatial_info": {"owner_player": "Hero"},
            },
        ]
    }

    hand_state, diagnostics = build_hand_state_with_diagnostics(enriched_payload)
    assert hand_state["hero_stack"] == 1200
    assert diagnostics["hero_stack"]["source"] == "chip_stack.owner_player"
    assert diagnostics["hero_stack"]["fallback_used"] is False
    assert "below usable threshold" in (diagnostics["hero_stack"]["warning"] or "")


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


def test_pot_ignores_pot_object_when_total_pot_missing() -> None:
    enriched_payload = {
        "objects": [
            {"class_name": "pot", "ocr_text": "200", "ocr_conf": 0.90},
        ]
    }

    hand_state = build_hand_state(enriched_payload)
    assert hand_state["pot"] == 150


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
    assert hand_state["is_hero_turn"] is False


def test_no_action_controls_defaults_to_no_active_turn() -> None:
    hand_state = build_hand_state({"objects": []})
    assert hand_state["is_hero_turn"] is False


def test_turn_halo_on_hero_sets_action_on_hero_seat() -> None:
    enriched_payload = {
        "objects": [
            {
                "class_name": "player_me",
                "confidence": 0.90,
                "turn_active": True,
                "turn_halo_score": 0.91,
                "spatial_conf": 0.70,
                "spatial_info": {"position": "BB", "hero_player": "moodaye"},
            },
            {
                "class_name": "dealer_button",
                "confidence": 0.90,
                "spatial_conf": 0.70,
                "spatial_info": {"dealer_player": "Weave"},
            },
        ]
    }

    hand_state = build_hand_state(enriched_payload)
    assert hand_state["hero_seat"] == "BB"
    assert hand_state["action_on"] == "BB"
    assert hand_state["is_hero_turn"] is True


def test_turn_halo_on_opponent_sets_action_on_opponent_seat() -> None:
    enriched_payload = {
        "objects": [
            {
                "class_name": "player_me",
                "bbox_xyxy": [850, 700, 950, 820],
                "confidence": 0.90,
                "spatial_conf": 0.70,
                "spatial_info": {"position": "BB", "hero_player": "moodaye"},
            },
            {
                "class_name": "player_other",
                "bbox_xyxy": [345, 220, 485, 380],
                "confidence": 0.90,
                "turn_active": True,
                "turn_halo_score": 0.92,
            },
            {
                "class_name": "player_name",
                "bbox_xyxy": [360, 250, 470, 360],
                "confidence": 0.90,
                "spatial_info": {"seat": "SB"},
            },
            {
                "class_name": "player_name",
                "bbox_xyxy": [1270, 250, 1380, 360],
                "confidence": 0.90,
                "spatial_info": {"seat": "BTN"},
            },
            {
                "class_name": "dealer_button",
                "confidence": 0.90,
                "spatial_conf": 0.70,
                "spatial_info": {"dealer_player": "Weave"},
            },
        ]
    }

    hand_state = build_hand_state(enriched_payload)
    assert hand_state["hero_seat"] == "BB"
    assert hand_state["action_on"] == "SB"
    assert hand_state["is_hero_turn"] is False


def test_position_accepts_spatial_conf_when_detection_conf_is_low() -> None:
    """Regression: low player_me detection confidence should not force BTN fallback."""
    enriched_payload = {
        "objects": [
            {
                "class_name": "player_me",
                "confidence": 0.44,
                "spatial_conf": 0.70,
                "spatial_info": {"position": "BB", "hero_player": "moodaye"},
            },
            {
                "class_name": "dealer_button",
                "confidence": 0.93,
                "spatial_conf": 0.70,
                "spatial_info": {"dealer_player": "Donna1212"},
            },
        ]
    }

    hand_state, diagnostics = build_hand_state_with_diagnostics(enriched_payload)
    assert hand_state["position"] == "BB"
    assert hand_state["hero_seat"] == "BB"
    assert diagnostics["position"]["fallback_used"] is False


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
# Hero folded inference
# ---------------------------------------------------------------------------


def test_hidden_hero_cards_with_post_blind_pot_infers_folded() -> None:
    enriched_payload = {
        "objects": [
            {"class_name": "blinds", "ocr_text": "10/20", "ocr_conf": 0.90},
            {"class_name": "total_pot", "ocr_text": "40", "ocr_conf": 0.90},
            {"class_name": "chip_stack", "ocr_text": "500", "ocr_conf": 0.90},
        ]
    }

    hand_state, diagnostics = build_hand_state_with_diagnostics(enriched_payload)

    assert hand_state["hero_cards_visibility"] == "not_exposed"
    assert hand_state["hero_folded"] is True
    assert hand_state["seats"][0]["status"] == "folded_this_hand"
    assert diagnostics["hero_folded"]["source"] == "hidden_cards_post_blind_pot"


def test_hidden_hero_cards_with_forced_blinds_only_does_not_infer_folded() -> None:
    enriched_payload = {
        "objects": [
            {"class_name": "blinds", "ocr_text": "10/20", "ocr_conf": 0.90},
            {"class_name": "total_pot", "ocr_text": "30", "ocr_conf": 0.90},
        ]
    }

    hand_state = build_hand_state(enriched_payload)
    assert hand_state["hero_cards_visibility"] == "not_exposed"
    assert hand_state["hero_folded"] is False


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


# ---------------------------------------------------------------------------
# Seats player_name enrichment
# ---------------------------------------------------------------------------


def test_seats_include_player_name_from_player_name_objects() -> None:
    """player_name objects with spatial_info.seat should populate seats[*].player_name."""
    enriched_payload = {
        "objects": [
            {
                "class_name": "player_me",
                "confidence": 0.90,
                "spatial_conf": 0.85,
                "spatial_info": {"position": "BTN", "hero_player": "moodaye"},
            },
            {
                "class_name": "player_name",
                "ocr_text": "Weave",
                "confidence": 0.90,
                "spatial_info": {"seat": "SB"},
            },
            {
                "class_name": "player_name",
                "ocr_text": "Donna1212",
                "confidence": 0.90,
                "spatial_info": {"seat": "BB"},
            },
        ]
    }

    hand_state = build_hand_state(enriched_payload)
    seats_by_label = {s["seat"]: s for s in hand_state["seats"]}

    assert seats_by_label["BTN"]["player_name"] == "moodaye"
    assert seats_by_label["SB"]["player_name"] == "Weave"
    assert seats_by_label["BB"]["player_name"] == "Donna1212"


def test_seats_player_name_is_none_when_no_player_name_objects() -> None:
    """When no player_name objects are present, player_name is None for every seat."""
    enriched_payload = {
        "objects": [
            {"class_name": "chip_stack", "ocr_text": "3000", "ocr_conf": 0.90},
        ]
    }

    hand_state = build_hand_state(enriched_payload)
    for seat_entry in hand_state["seats"]:
        assert seat_entry["player_name"] is None


def test_seats_hero_player_name_comes_from_player_me_spatial_info() -> None:
    """Hero seat name is sourced from player_me.spatial_info.hero_player even without
    a matching player_name object for that seat."""
    enriched_payload = {
        "objects": [
            {
                "class_name": "player_me",
                "confidence": 0.90,
                "spatial_conf": 0.85,
                "spatial_info": {"position": "SB", "hero_player": "rajiv"},
            },
        ]
    }

    hand_state = build_hand_state(enriched_payload)
    seats_by_label = {s["seat"]: s for s in hand_state["seats"]}

    assert seats_by_label["SB"]["player_name"] == "rajiv"
    assert seats_by_label["BTN"]["player_name"] is None
    assert seats_by_label["BB"]["player_name"] is None


def test_hero_turn_inference_with_bet_box() -> None:
    enriched_payload = {
        "objects": [
            {"class_name": "bet_box", "confidence": 0.98},
            {"class_name": "holecard", "classification": "Ah", "confidence": 0.95},
            {"class_name": "holecard", "classification": "Kd", "confidence": 0.93},
        ]
    }

    hand_state, diagnostics = build_hand_state_with_diagnostics(enriched_payload)

    assert hand_state["is_hero_turn"] is True
    assert diagnostics["is_hero_turn"] == {
        "source": "bet_box_detection",
        "value": True,
        "confidence": 1.0,
    }


# ---------------------------------------------------------------------------
# All-in chip stack detection
# ---------------------------------------------------------------------------


def test_hero_chip_stack_all_in_text_sets_stack_zero_and_is_all_in() -> None:
    """chip_stack with 'All In' OCR text → hero_stack=0, seat is_all_in=True."""
    enriched_payload = {
        "objects": [
            {"class_name": "chip_stack", "ocr_text": "All In", "ocr_conf": 0.88},
            {"class_name": "blinds", "ocr_text": "50/100", "ocr_conf": 0.90},
        ]
    }

    hand_state = build_hand_state(enriched_payload)

    assert hand_state["hero_stack"] == 0
    hero_seat = hand_state["hero_seat"]
    hero_entry = next(s for s in hand_state["seats"] if s["seat"] == hero_seat)
    assert hero_entry["is_all_in"] is True


def test_hero_chip_stack_all_in_case_variants() -> None:
    """All-In / ALL IN / all in variants are all recognised."""
    for text in ("All In", "All-In", "ALL IN", "all in"):
        enriched_payload = {
            "objects": [
                {"class_name": "chip_stack", "ocr_text": text, "ocr_conf": 0.85},
            ]
        }
        hand_state = build_hand_state(enriched_payload)
        assert hand_state["hero_stack"] == 0, f"failed for ocr_text={text!r}"


def test_opponent_chip_stack_all_in_sets_seat_is_all_in() -> None:
    """An opponent with 'All In' chip_stack text gets is_all_in=True on their seat."""
    enriched_payload = {
        "objects": [
            {
                "class_name": "player_me",
                "confidence": 0.92,
                "spatial_conf": 0.90,
                "spatial_info": {"position": "BTN", "hero_player": "Hero"},
            },
            {
                "class_name": "player_name",
                "ocr_text": "Villain",
                "ocr_conf": 0.88,
                "confidence": 0.88,
                "spatial_info": {"seat": "SB"},
            },
            {
                "class_name": "chip_stack",
                "ocr_text": "1500",
                "ocr_conf": 0.90,
                "confidence": 0.90,
                "spatial_info": {"owner_player": "Hero"},
            },
            {
                "class_name": "chip_stack",
                "ocr_text": "All In",
                "ocr_conf": 0.85,
                "confidence": 0.85,
                "spatial_info": {"owner_player": "Villain"},
            },
        ]
    }

    hand_state = build_hand_state(enriched_payload)

    assert hand_state["hero_stack"] == 1500
    hero_entry = next(s for s in hand_state["seats"] if s["is_hero"])
    assert hero_entry["is_all_in"] is None  # hero not all-in
    sb_entry = next(s for s in hand_state["seats"] if s["seat"] == "SB")
    assert sb_entry["stack"] == 0
    assert sb_entry["is_all_in"] is True


# ---------------------------------------------------------------------------
# Opponent stack propagation via named ownership
# ---------------------------------------------------------------------------


def test_opponent_stack_populated_via_named_owner_matching() -> None:
    """chip_stack.spatial_info.owner_player matched to player_name.spatial_info.seat
    should populate the corresponding opponent seat's stack value."""
    enriched_payload = {
        "objects": [
            {
                "class_name": "player_me",
                "confidence": 0.92,
                "spatial_conf": 0.90,
                "spatial_info": {"position": "BTN", "hero_player": "Hero"},
            },
            {
                "class_name": "player_name",
                "ocr_text": "Villain",
                "confidence": 0.90,
                "spatial_info": {"seat": "SB"},
            },
            {
                "class_name": "chip_stack",
                "ocr_text": "2200",
                "ocr_conf": 0.90,
                "confidence": 0.90,
                "spatial_info": {"owner_player": "Hero"},
            },
            {
                "class_name": "chip_stack",
                "ocr_text": "4800",
                "ocr_conf": 0.90,
                "confidence": 0.90,
                "spatial_info": {"owner_player": "Villain"},
            },
        ]
    }

    hand_state = build_hand_state(enriched_payload)

    assert hand_state["hero_stack"] == 2200
    sb_entry = next(s for s in hand_state["seats"] if s["seat"] == "SB")
    assert sb_entry["stack"] == 4800
    # Unmatched opponent seat has no stack
    bb_entry = next(s for s in hand_state["seats"] if s["seat"] == "BB")
    assert bb_entry["stack"] is None


def test_opponent_stack_none_when_owner_name_unrecognised() -> None:
    """chip_stack whose owner_player doesn't match any player_name object
    leaves opponent seat stacks as None."""
    enriched_payload = {
        "objects": [
            {
                "class_name": "player_name",
                "ocr_text": "Weave",
                "confidence": 0.90,
                "spatial_info": {"seat": "SB"},
            },
            {
                "class_name": "chip_stack",
                "ocr_text": "3100",
                "ocr_conf": 0.90,
                "confidence": 0.90,
                "spatial_info": {"owner_player": "UnknownPlayer"},
            },
        ]
    }

    hand_state = build_hand_state(enriched_payload)
    sb_entry = next(s for s in hand_state["seats"] if s["seat"] == "SB")
    assert sb_entry["stack"] is None


# ---------------------------------------------------------------------------
# _extract_int OCR character substitution
# ---------------------------------------------------------------------------


def test_extract_int_ocr_substitution_O_for_zero() -> None:
    """OCR commonly misreads '0' as 'O'. Parser should substitute it."""
    enriched_payload = {
        "objects": [
            {"class_name": "chip_stack", "ocr_text": "3O0O", "ocr_conf": 0.85},
        ]
    }
    hand_state = build_hand_state(enriched_payload)
    assert hand_state["hero_stack"] == 3000


def test_extract_int_ocr_substitution_S_for_five() -> None:
    """OCR may misread '5' as 'S' when surrounded by digits."""
    enriched_payload = {
        "objects": [
            {"class_name": "chip_stack", "ocr_text": "1S00", "ocr_conf": 0.85},
        ]
    }
    hand_state = build_hand_state(enriched_payload)
    assert hand_state["hero_stack"] == 1500


def test_extract_int_ocr_ignores_prefix_labels() -> None:
    """Parser extracts the first integer even when a text label precedes it."""
    enriched_payload = {
        "objects": [
            {"class_name": "total_pot", "ocr_text": "Pot 750", "ocr_conf": 0.88},
        ]
    }
    hand_state = build_hand_state(enriched_payload)
    assert hand_state["pot"] == 750


# ---------------------------------------------------------------------------
# Blinds parsing — space-separated format
# ---------------------------------------------------------------------------


def test_blinds_space_separated_format() -> None:
    """Blinds OCR may return '50 100' instead of '50/100'."""
    enriched_payload = {
        "objects": [
            {"class_name": "blinds", "ocr_text": "50 100", "ocr_conf": 0.90},
        ]
    }
    hand_state = build_hand_state(enriched_payload)
    assert hand_state["small_blind"] == 50
    assert hand_state["big_blind"] == 100


# ---------------------------------------------------------------------------
# Position from dealer_button alone (no player_me in payload)
# ---------------------------------------------------------------------------


def test_position_from_dealer_button_hero_position_when_player_me_absent() -> None:
    """When player_me is not detected, position falls back to dealer_button's
    hero_position field if the spatial confidence passes the usable threshold."""
    enriched_payload = {
        "objects": [
            {
                "class_name": "dealer_button",
                "confidence": 0.90,
                "spatial_conf": 0.80,
                "spatial_info": {"hero_position": "SB"},
            },
        ]
    }

    hand_state, diagnostics = build_hand_state_with_diagnostics(enriched_payload)

    assert hand_state["position"] == "SB"
    assert hand_state["hero_seat"] == "SB"
    assert diagnostics["position"]["fallback_used"] is False


def test_position_defaults_to_btn_when_no_spatial_signals() -> None:
    """Without player_me or dealer_button, position defaults to BTN."""
    enriched_payload = {
        "objects": [
            {"class_name": "chip_stack", "ocr_text": "3000", "ocr_conf": 0.90},
        ]
    }

    hand_state, diagnostics = build_hand_state_with_diagnostics(enriched_payload)

    assert hand_state["position"] == "BTN"
    assert diagnostics["position"]["fallback_used"] is True
