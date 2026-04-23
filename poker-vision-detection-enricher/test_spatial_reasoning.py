"""Tests for the spatial reasoning post-pass."""

from spatial_reasoning import resolve_hero_position, resolve_spatial_relationships


def _make_obj(
    cls: str, bbox: list[int], ocr_text: str = "", confidence: float = 0.90
) -> dict:
    return {
        "class_name": cls,
        "class": cls,
        "bbox_xyxy": bbox,
        "confidence": confidence,
        "ocr_text": ocr_text,
    }


# ---------------------------------------------------------------------------
# dealer_button — nearest player_name by Euclidean distance
# ---------------------------------------------------------------------------


def test_dealer_button_matched_to_nearest_player_name() -> None:
    # player_A centre: (50, 50), player_B centre: (300, 50)
    # dealer_button centre: (60, 80) — clearly closest to player_A
    objects = [
        _make_obj("player_name", [0, 0, 100, 100], ocr_text="player_A"),
        _make_obj("player_name", [250, 0, 350, 100], ocr_text="player_B"),
        _make_obj("dealer_button", [30, 50, 90, 110]),
    ]
    resolve_spatial_relationships(objects, default_conf=0.70)

    dealer = next(o for o in objects if o["class_name"] == "dealer_button")
    assert dealer["spatial_info"]["dealer_player"] == "player_A"
    assert dealer["spatial_conf"] == 0.70


def test_dealer_button_no_player_names_gives_empty_spatial_info() -> None:
    objects = [_make_obj("dealer_button", [30, 50, 90, 110])]
    resolve_spatial_relationships(objects)

    dealer = objects[0]
    assert dealer["spatial_info"] == {}
    assert dealer["spatial_conf"] == 0.0


# ---------------------------------------------------------------------------
# chip_stack — nearest player_name directly above
# ---------------------------------------------------------------------------


def test_chip_stack_matched_to_player_name_above() -> None:
    # player_name centre: (100, 50) — above the chip stack
    # chip_stack centre: (100, 150)
    objects = [
        _make_obj("player_name", [50, 0, 150, 100], ocr_text="Hero"),
        _make_obj("chip_stack", [50, 100, 150, 200]),
    ]
    resolve_spatial_relationships(objects, default_conf=0.70)

    stack = next(o for o in objects if o["class_name"] == "chip_stack")
    assert stack["spatial_info"]["owner_player"] == "Hero"
    assert stack["spatial_conf"] == 0.70


def test_chip_stack_ignores_player_name_below() -> None:
    # player_name is BELOW the chip stack — should not match
    objects = [
        _make_obj("player_name", [50, 200, 150, 300], ocr_text="Hero"),
        _make_obj("chip_stack", [50, 0, 150, 100]),
    ]
    resolve_spatial_relationships(objects)

    stack = next(o for o in objects if o["class_name"] == "chip_stack")
    assert stack["spatial_info"] == {}
    assert stack["spatial_conf"] == 0.0


def test_chip_stack_ignores_player_name_too_far_horizontally() -> None:
    # player_name is above but 300px to the right — beyond _HORIZONTAL_THRESHOLD
    objects = [
        _make_obj("player_name", [350, 0, 450, 100], ocr_text="Villain"),
        _make_obj("chip_stack", [50, 100, 150, 200]),
    ]
    resolve_spatial_relationships(objects)

    stack = next(o for o in objects if o["class_name"] == "chip_stack")
    assert stack["spatial_info"] == {}
    assert stack["spatial_conf"] == 0.0


def test_chip_stack_picks_closest_player_above_when_multiple() -> None:
    # Two players above, one much closer vertically
    objects = [
        _make_obj("player_name", [50, 0, 150, 20], ocr_text="FarAway"),  # centre y=10
        _make_obj("player_name", [50, 80, 150, 100], ocr_text="Close"),  # centre y=90
        _make_obj("chip_stack", [50, 110, 150, 150]),  # centre y=130
    ]
    resolve_spatial_relationships(objects, default_conf=0.70)

    stack = next(o for o in objects if o["class_name"] == "chip_stack")
    assert stack["spatial_info"]["owner_player"] == "Close"


# ---------------------------------------------------------------------------
# bet / pot_bet — nearest player_name by Euclidean distance
# ---------------------------------------------------------------------------


def test_bet_matched_to_nearest_player_name() -> None:
    objects = [
        _make_obj("player_name", [0, 0, 100, 50], ocr_text="Hero"),
        _make_obj("player_name", [300, 0, 400, 50], ocr_text="Villain"),
        _make_obj("bet", [40, 60, 80, 100]),  # closer to Hero
    ]
    resolve_spatial_relationships(objects, default_conf=0.70)

    bet = next(o for o in objects if o["class_name"] == "bet")
    assert bet["spatial_info"]["owner_player"] == "Hero"
    assert bet["spatial_conf"] == 0.70


def test_pot_bet_matched_to_nearest_player_name() -> None:
    objects = [
        _make_obj("player_name", [0, 0, 100, 50], ocr_text="Hero"),
        _make_obj("player_name", [300, 0, 400, 50], ocr_text="Villain"),
        _make_obj("pot_bet", [320, 60, 380, 100]),  # closer to Villain
    ]
    resolve_spatial_relationships(objects, default_conf=0.70)

    pot_bet = next(o for o in objects if o["class_name"] == "pot_bet")
    assert pot_bet["spatial_info"]["owner_player"] == "Villain"


# ---------------------------------------------------------------------------
# Non-spatial objects are not mutated
# ---------------------------------------------------------------------------


def test_non_spatial_objects_not_mutated() -> None:
    card = _make_obj("card", [0, 0, 50, 80])
    pot = _make_obj("total_pot", [100, 100, 200, 150])
    objects = [card, pot]
    resolve_spatial_relationships(objects)

    assert "spatial_info" not in card
    assert "spatial_conf" not in card
    assert "spatial_info" not in pot
    assert "spatial_conf" not in pot


# ---------------------------------------------------------------------------
# resolve_hero_position — clockwise seat ordering + dealer offset
# ---------------------------------------------------------------------------
#
# Standard 3-player layout used in all tests below:
#   "Hero"    player_name bbox (250,350,350,450) → centre (300, 400)  ← bottom
#   "Alice"   player_name bbox (350, 50,450,150) → centre (400, 100)  ← top-right
#   "Bob"     player_name bbox (150, 50,250,150) → centre (200, 100)  ← top-left
#
# Centroid of the three players: (300, 200)
# Clockwise-from-bottom sort key (formula: (90 - atan2(dy,dx)_deg) % 360):
#   Hero:  dy=+200, dx=  0 → atan2=90°  → key=  0°  (sorts first)
#   Alice: dy=-100, dx=+100 → atan2=-45°→315° → key=135° (sorts second)
#   Bob:   dy=-100, dx=-100 → atan2=-135°→225° → key=225° (sorts third)
# Ordered: [Hero(0), Alice(1), Bob(2)]
#
# Offset = (hero_idx - dealer_idx) % 3
#   dealer=Hero(0)  → offset 0 → BTN
#   dealer=Bob(2)   → offset (0-2)%3=1 → SB
#   dealer=Alice(1) → offset (0-1)%3=2 → BB


def _make_player_name_with_ocr(name: str, bbox: list[int]) -> dict:
    obj = _make_obj("player_name", bbox, ocr_text=name)
    return obj


def _make_player_me(bbox: list[int]) -> dict:
    return _make_obj("player_me", bbox)


def _make_dealer_button_resolved(dealer_name: str, bbox: list[int]) -> dict:
    """Simulates a dealer_button already annotated by resolve_spatial_relationships."""
    obj = _make_obj("dealer_button", bbox)
    obj["spatial_info"] = {"dealer_player": dealer_name}
    obj["spatial_conf"] = 0.70
    return obj


def _three_player_objects(dealer_name: str) -> list:
    return [
        _make_player_name_with_ocr("Hero", [250, 350, 350, 450]),
        _make_player_name_with_ocr("Alice", [350, 50, 450, 150]),
        _make_player_name_with_ocr("Bob", [150, 50, 250, 150]),
        _make_player_me([255, 355, 345, 445]),  # sits on top of "Hero" name
        _make_dealer_button_resolved(dealer_name, [0, 0, 30, 30]),
    ]


def test_hero_is_btn_when_dealer_is_hero() -> None:
    objects = _three_player_objects("Hero")
    resolve_hero_position(objects, default_conf=0.70)

    player_me = next(o for o in objects if o["class_name"] == "player_me")
    assert player_me["spatial_info"]["position"] == "BTN"
    assert player_me["spatial_conf"] == 0.70


def test_hero_is_sb_when_dealer_is_bob() -> None:
    # Bob is last in clockwise order; hero is 1 step clockwise from Bob → SB
    objects = _three_player_objects("Bob")
    resolve_hero_position(objects, default_conf=0.70)

    player_me = next(o for o in objects if o["class_name"] == "player_me")
    assert player_me["spatial_info"]["position"] == "SB"


def test_hero_is_bb_when_dealer_is_alice() -> None:
    # Alice is second in clockwise order; hero is 2 steps clockwise from Alice → BB
    objects = _three_player_objects("Alice")
    resolve_hero_position(objects, default_conf=0.70)

    player_me = next(o for o in objects if o["class_name"] == "player_me")
    assert player_me["spatial_info"]["position"] == "BB"


def test_hero_position_two_players_hero_is_dealer() -> None:
    # Heads-up: hero at bottom, villain at top; hero is the dealer → BTN
    objects = [
        _make_player_name_with_ocr("Hero", [250, 350, 350, 450]),
        _make_player_name_with_ocr("Villain", [250, 50, 350, 150]),
        _make_player_me([255, 355, 345, 445]),
        _make_dealer_button_resolved("Hero", [0, 0, 30, 30]),
    ]
    resolve_hero_position(objects, default_conf=0.70)

    player_me = next(o for o in objects if o["class_name"] == "player_me")
    assert player_me["spatial_info"]["position"] == "BTN"


def test_hero_position_two_players_villain_is_dealer() -> None:
    # Heads-up: villain is the dealer → hero is BB
    objects = [
        _make_player_name_with_ocr("Hero", [250, 350, 350, 450]),
        _make_player_name_with_ocr("Villain", [250, 50, 350, 150]),
        _make_player_me([255, 355, 345, 445]),
        _make_dealer_button_resolved("Villain", [0, 0, 30, 30]),
    ]
    resolve_hero_position(objects, default_conf=0.70)

    player_me = next(o for o in objects if o["class_name"] == "player_me")
    assert player_me["spatial_info"]["position"] == "BB"


def test_hero_position_no_player_me_is_noop() -> None:
    objects = [
        _make_player_name_with_ocr("Hero", [250, 350, 350, 450]),
        _make_player_name_with_ocr("Alice", [350, 50, 450, 150]),
        _make_dealer_button_resolved("Hero", [0, 0, 30, 30]),
    ]
    resolve_hero_position(objects)
    # No player_me in objects — should not raise; nothing annotated
    assert all(
        "spatial_info" not in o or "position" not in o.get("spatial_info", {})
        for o in objects
        if o["class_name"] != "dealer_button"
    )


def test_hero_position_no_dealer_button_is_noop() -> None:
    objects = [
        _make_player_name_with_ocr("Hero", [250, 350, 350, 450]),
        _make_player_name_with_ocr("Alice", [350, 50, 450, 150]),
        _make_player_me([255, 355, 345, 445]),
    ]
    resolve_hero_position(objects)
    player_me = next(o for o in objects if o["class_name"] == "player_me")
    assert "spatial_info" not in player_me


def test_hero_position_dealer_name_not_in_player_names_is_noop() -> None:
    # dealer_player name doesn't match any player_name OCR text
    objects = _three_player_objects("UnknownPlayer")
    resolve_hero_position(objects)
    player_me = next(o for o in objects if o["class_name"] == "player_me")
    assert "spatial_info" not in player_me


def test_hero_position_dealer_name_matched_case_insensitively() -> None:
    objects = _three_player_objects("HERO")  # upper-case dealer name
    resolve_hero_position(objects, default_conf=0.70)
    player_me = next(o for o in objects if o["class_name"] == "player_me")
    assert player_me["spatial_info"]["position"] == "BTN"


if __name__ == "__main__":
    test_dealer_button_matched_to_nearest_player_name()
    test_dealer_button_no_player_names_gives_empty_spatial_info()
    test_chip_stack_matched_to_player_name_above()
    test_chip_stack_ignores_player_name_below()
    test_chip_stack_ignores_player_name_too_far_horizontally()
    test_chip_stack_picks_closest_player_above_when_multiple()
    test_bet_matched_to_nearest_player_name()
    test_pot_bet_matched_to_nearest_player_name()
    test_non_spatial_objects_not_mutated()
    test_hero_is_btn_when_dealer_is_hero()
    test_hero_is_sb_when_dealer_is_bob()
    test_hero_is_bb_when_dealer_is_alice()
    test_hero_position_two_players_hero_is_dealer()
    test_hero_position_two_players_villain_is_dealer()
    test_hero_position_no_player_me_is_noop()
    test_hero_position_no_dealer_button_is_noop()
    test_hero_position_dealer_name_not_in_player_names_is_noop()
    test_hero_position_dealer_name_matched_case_insensitively()
    print("All spatial reasoning tests passed.")
