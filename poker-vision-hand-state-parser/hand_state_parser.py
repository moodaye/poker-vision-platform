from __future__ import annotations

import re
from typing import Any

_CARD_LABEL_RE = re.compile(r"^[2-9TJQKA][cdhs]$")
_ALL_IN_RE = re.compile(r"^all[\W_]*in$", re.IGNORECASE)
_DEFAULT_SMALL_BLIND = 50
_DEFAULT_BIG_BLIND = 100
_DEFAULT_HERO_STACK = 3000
_DEFAULT_POT = 150
_DEFAULT_ANTE = 0
_TRUSTED_THRESHOLD = 0.80
_USABLE_THRESHOLD = 0.55
_MIN_DETECTION_FOR_CARDS = 0.60
_MIN_CLASSIFICATION_FOR_CARDS = 0.70
_MIN_ACTION_HISTORY_ENTRY_CONF = 0.65
_MIN_HERO_FOLD_CONF = 0.70
_SCHEMA_VERSION = "2.2.0"
_BOARD_CARD_CLASSES = frozenset({"flop_card", "turn_card", "river_card"})


def _object_class(obj: dict[str, Any]) -> str:
    return str(obj.get("class_name") or obj.get("class") or "unknown")


def _confidence(value: Any, fallback: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(0.0, min(1.0, parsed))


def _field_confidence(obj: dict[str, Any], extraction_key: str | None) -> float:
    detection_conf = _confidence(obj.get("confidence"), fallback=1.0)
    extraction_conf = 1.0
    if extraction_key is not None:
        extraction_conf = _confidence(obj.get(extraction_key), fallback=1.0)
    return min(detection_conf, extraction_conf)


def _confidence_band(field_conf: float) -> str:
    if field_conf >= _TRUSTED_THRESHOLD:
        return "trusted"
    if field_conf >= _USABLE_THRESHOLD:
        return "usable"
    return "rejected"


def _is_accepted(field_conf: float) -> bool:
    return field_conf >= _USABLE_THRESHOLD


def _diag_entry(
    source: str,
    field_conf: float,
    fallback_used: bool,
    warning: str | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "field_conf": round(field_conf, 3),
        "band": _confidence_band(field_conf),
        "fallback_used": fallback_used,
        "warning": warning,
    }


def _valid_card_label(label: Any) -> str | None:
    if not isinstance(label, str):
        return None

    normalized = label.strip()
    if not normalized:
        return None

    # Classifier may emit uppercase two-char labels (e.g. "AH", "KD")
    # or ten as "10H". Normalize to parser format (e.g. "Ah", "Kd", "Th").
    normalized = normalized.replace("10", "T")
    if len(normalized) != 2:
        return None

    normalized = normalized[0].upper() + normalized[1].lower()
    if _CARD_LABEL_RE.fullmatch(normalized):
        return normalized
    return None


def _extract_int(value: Any) -> int | None:
    if isinstance(value, int | float):
        return int(value)
    if not isinstance(value, str):
        return None

    normalized = value.replace(",", "")
    replacement_map = {
        "O": "0",
        "o": "0",
        "S": "5",
        "s": "5",
        "I": "1",
        "l": "1",
        "|": "1",
    }
    chars = list(normalized)
    for idx, ch in enumerate(chars):
        if ch not in replacement_map:
            continue
        prev_is_digit = idx > 0 and chars[idx - 1].isdigit()
        next_is_digit = idx + 1 < len(chars) and chars[idx + 1].isdigit()
        if prev_is_digit or next_is_digit:
            chars[idx] = replacement_map[ch]
    normalized = "".join(chars)

    match = re.search(r"\d+", normalized)
    if match is None:
        return None
    return int(match.group(0))


def _extract_blinds(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None

    normalized = value.replace(",", "")
    replacement_map = {
        "O": "0",
        "o": "0",
        "S": "5",
        "s": "5",
        "I": "1",
        "l": "1",
        "|": "1",
    }
    chars = list(normalized)
    for idx, ch in enumerate(chars):
        if ch not in replacement_map:
            continue
        prev_is_digit = idx > 0 and chars[idx - 1].isdigit()
        next_is_digit = idx + 1 < len(chars) and chars[idx + 1].isdigit()
        if prev_is_digit or next_is_digit:
            chars[idx] = replacement_map[ch]
    normalized = "".join(chars)
    matches = re.findall(r"\d+", normalized)
    if len(matches) < 2:
        return None
    small_blind = int(matches[0])
    big_blind = int(matches[1])
    return small_blind, big_blind


def _extract_position_from_spatial(spatial_info: Any) -> str | None:
    if not isinstance(spatial_info, dict):
        return None

    for key in ("seat", "position", "hero_position"):
        raw = spatial_info.get(key)
        if isinstance(raw, str):
            candidate = raw.strip().upper()
            if candidate in {"BTN", "SB", "BB"}:
                return candidate
    return None


def _first_object_for_classes(
    objects: list[dict[str, Any]],
    class_names: tuple[str, ...],
) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    for class_name in class_names:
        source_obj = next(
            (obj for obj in objects if _object_class(obj) == class_name),
            None,
        )
        if source_obj is not None:
            return class_name, source_obj
    return None, None


def _seat_status(
    *,
    is_hero: bool,
    is_hero_turn: bool,
    hero_folded: bool,
    hero_has_cards: bool,
    stack: int | None,
    seat: str = "",
    action_on: str = "none",
) -> str:
    if stack is not None and stack <= 0:
        return "eliminated_tournament"
    if is_hero:
        if hero_folded:
            return "folded_this_hand"
        if not hero_has_cards:
            return "watching_hand"
        return "deciding" if is_hero_turn else "waiting_turn"
    # Opponent: use action_on to distinguish deciding from waiting.
    if action_on == seat:
        return "deciding"
    return "waiting_turn"


def _extract_hero_player_name(spatial_info: Any) -> str | None:
    if not isinstance(spatial_info, dict):
        return None

    for key in ("hero_player", "player_name", "owner_player"):
        raw = spatial_info.get(key)
        if isinstance(raw, str):
            candidate = raw.strip()
            if candidate:
                return candidate
    return None


def _extract_card_x(obj: dict[str, Any]) -> float | None:
    raw_x = obj.get("x")
    if isinstance(raw_x, int | float):
        return float(raw_x)

    bbox = obj.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, _, x2, _ = bbox
        if isinstance(x1, int | float) and isinstance(x2, int | float):
            return (float(x1) + float(x2)) / 2.0

    if isinstance(bbox, dict):
        for key in ("x", "center_x", "cx", "left", "x1"):
            value = bbox.get(key)
            if isinstance(value, int | float):
                return float(value)
        left = bbox.get("left")
        right = bbox.get("right")
        if isinstance(left, int | float) and isinstance(right, int | float):
            return (float(left) + float(right)) / 2.0

    bbox_xyxy = obj.get("bbox_xyxy")
    if isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4:
        x1, _, x2, _ = bbox_xyxy
        if isinstance(x1, int | float) and isinstance(x2, int | float):
            return (float(x1) + float(x2)) / 2.0

    return None


def _bbox_centre(obj: dict[str, Any]) -> tuple[float, float] | None:
    bbox = obj.get("bbox_xyxy") or obj.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = bbox[:4]
    if not all(isinstance(v, int | float) for v in (x1, y1, x2, y2)):
        return None
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def _nearest_seat_for_object(
    obj: dict[str, Any],
    seated_players: list[dict[str, Any]],
) -> str | None:
    centre = _bbox_centre(obj)
    if centre is None:
        return None
    best_seat: str | None = None
    best_dist = float("inf")
    for player in seated_players:
        seat = _extract_position_from_spatial(player.get("spatial_info"))
        if seat is None:
            continue
        pcentre = _bbox_centre(player)
        if pcentre is None:
            continue
        dist = ((centre[0] - pcentre[0]) ** 2 + (centre[1] - pcentre[1]) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_seat = seat
    return best_seat


def _order_cards_left_to_right(
    cards: list[tuple[str, float | None, int]],
) -> list[str]:
    sorted_cards = sorted(
        cards,
        key=lambda item: (
            item[1] is None,
            item[1] if item[1] is not None else float("inf"),
            item[2],
        ),
    )
    return [card for card, _, _ in sorted_cards]


def _derive_hand_phase(objects: list[dict[str, Any]]) -> str:
    """Infer hand phase from visible board-card detections.

    Any visible board card means the hand is postflop.
    """
    board_card_count = sum(
        1 for obj in objects if _object_class(obj) in _BOARD_CARD_CLASSES
    )
    return "postflop" if board_card_count > 0 else "preflop"


def build_hand_state_with_diagnostics(
    enriched_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    raw_objects = enriched_payload.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("enriched payload must contain an 'objects' list")

    objects = [obj for obj in raw_objects if isinstance(obj, dict)]
    diagnostics: dict[str, dict[str, Any]] = {}
    hand_phase = _derive_hand_phase(objects)

    # hero_cards: prefer holecard, then card
    hero_card_candidates: list[
        tuple[str, float, str, str | None, float | None, int]
    ] = []
    for target_class in ("holecard", "card"):
        for obj_index, obj in enumerate(objects):
            if _object_class(obj) != target_class:
                continue
            label = _valid_card_label(obj.get("classification"))
            if label is None:
                continue

            det_conf = _confidence(obj.get("confidence"), fallback=1.0)
            cls_conf = _confidence(obj.get("classification_conf"), fallback=1.0)
            field_conf = min(det_conf, cls_conf)

            if det_conf < _MIN_DETECTION_FOR_CARDS:
                continue
            if cls_conf < _MIN_CLASSIFICATION_FOR_CARDS:
                continue
            if not _is_accepted(field_conf):
                continue

            warning = None
            if _confidence_band(field_conf) == "usable":
                warning = "usable confidence; accepted with caution"
            hero_card_candidates.append(
                (
                    label,
                    field_conf,
                    target_class,
                    warning,
                    _extract_card_x(obj),
                    obj_index,
                )
            )

    if len(hero_card_candidates) >= 2:
        selected_candidates = hero_card_candidates[:2]
        hero_cards = _order_cards_left_to_right(
            [(item[0], item[4], item[5]) for item in selected_candidates]
        )
        hero_cards_conf = min(selected_candidates[0][1], selected_candidates[1][1])
        source = f"{selected_candidates[0][2]}+{selected_candidates[1][2]}"
        warning = selected_candidates[0][3] or selected_candidates[1][3]
        hero_cards_visibility = "exposed"
        diagnostics["hero_cards"] = _diag_entry(source, hero_cards_conf, False, warning)
    else:
        # Relaxed fallback: still prefer model output, but ignore strict thresholds.
        relaxed_candidates: list[tuple[str, float, str, float | None, int]] = []
        for target_class in ("holecard", "card"):
            for obj_index, obj in enumerate(objects):
                if _object_class(obj) != target_class:
                    continue
                label = _valid_card_label(obj.get("classification"))
                if label is None:
                    continue
                det_conf = _confidence(obj.get("confidence"), fallback=1.0)
                cls_conf = _confidence(obj.get("classification_conf"), fallback=1.0)
                relaxed_candidates.append(
                    (
                        label,
                        min(det_conf, cls_conf),
                        target_class,
                        _extract_card_x(obj),
                        obj_index,
                    )
                )

        if len(relaxed_candidates) >= 2:
            selected_relaxed = relaxed_candidates[:2]
            hero_cards = _order_cards_left_to_right(
                [(item[0], item[3], item[4]) for item in selected_relaxed]
            )
            hero_cards_conf = min(selected_relaxed[0][1], selected_relaxed[1][1])
            source = f"relaxed_{selected_relaxed[0][2]}+{selected_relaxed[1][2]}"
            hero_cards_visibility = "exposed"
            diagnostics["hero_cards"] = _diag_entry(
                source,
                hero_cards_conf,
                False,
                "strict confidence gates rejected cards; accepted relaxed candidates",
            )
        else:
            hero_cards = []
            hero_cards_visibility = "not_exposed"
            diagnostics["hero_cards"] = _diag_entry(
                "not_exposed",
                0.0,
                False,
                "insufficient visible hero card candidates",
            )

    # position: derive from player_me + dealer_button spatial info
    player_me_obj = next(
        (obj for obj in objects if _object_class(obj) == "player_me"), None
    )
    dealer_obj = next(
        (obj for obj in objects if _object_class(obj) == "dealer_button"),
        None,
    )
    position = "BTN"

    player_pos = (
        _extract_position_from_spatial(player_me_obj.get("spatial_info"))
        if player_me_obj is not None
        else None
    )
    hero_player_name = (
        _extract_hero_player_name(player_me_obj.get("spatial_info"))
        if player_me_obj is not None
        else None
    )

    # Pre-build name→seat index from player_name objects enriched by Stage 2.
    # Used below to assign chip_stack owners to seats for opponent stack values.
    name_lower_to_seat: dict[str, str] = {}
    for obj in objects:
        if _object_class(obj) != "player_name":
            continue
        ocr_name = obj.get("ocr_text")
        obj_seat = _extract_position_from_spatial(obj.get("spatial_info"))
        if isinstance(ocr_name, str) and ocr_name.strip() and obj_seat is not None:
            name_lower_to_seat[ocr_name.strip().lower()] = obj_seat

    dealer_pos = (
        _extract_position_from_spatial(dealer_obj.get("spatial_info"))
        if dealer_obj is not None
        else None
    )
    if player_pos is not None and player_me_obj is not None:
        if dealer_obj is not None:
            # Position is a spatial post-pass output; gate it by spatial confidence
            # instead of raw detector confidence for player_me/dealer_button boxes.
            position_conf = min(
                _confidence(player_me_obj.get("spatial_conf"), fallback=0.0),
                _confidence(dealer_obj.get("spatial_conf"), fallback=0.0),
            )
            if _is_accepted(position_conf):
                position = player_pos
                warning = None
                if _confidence_band(position_conf) == "usable":
                    warning = "usable confidence; accepted with caution"
                diagnostics["position"] = _diag_entry(
                    "player_me+dealer_button",
                    position_conf,
                    False,
                    warning,
                )
            else:
                diagnostics["position"] = _diag_entry(
                    "fallback",
                    position_conf,
                    True,
                    "position confidence below minimum usable threshold",
                )
        else:
            position_conf = _confidence(player_me_obj.get("spatial_conf"), fallback=0.0)
            if _is_accepted(position_conf):
                position = player_pos
                diagnostics["position"] = _diag_entry(
                    "player_me",
                    position_conf,
                    False,
                    "dealer_button missing; using hero seat only",
                )
            else:
                diagnostics["position"] = _diag_entry(
                    "fallback",
                    position_conf,
                    True,
                    "position confidence below minimum usable threshold",
                )
    elif dealer_pos is not None and dealer_obj is not None:
        dealer_conf = _confidence(dealer_obj.get("spatial_conf"), fallback=0.0)
        if _is_accepted(dealer_conf):
            position = dealer_pos
            diagnostics["position"] = _diag_entry(
                "dealer_button",
                dealer_conf,
                False,
                "using dealer spatial hero_position",
            )
        else:
            diagnostics["position"] = _diag_entry(
                "fallback",
                dealer_conf,
                True,
                "position confidence below minimum usable threshold",
            )
    else:
        diagnostics["position"] = _diag_entry(
            "fallback",
            0.0,
            True,
            "no positional spatial signals available",
        )

    small_blind = _DEFAULT_SMALL_BLIND
    big_blind = _DEFAULT_BIG_BLIND
    blinds_obj = next((obj for obj in objects if _object_class(obj) == "blinds"), None)
    if blinds_obj is not None:
        parsed_blinds = _extract_blinds(blinds_obj.get("ocr_text"))
        blinds_conf = _field_confidence(blinds_obj, "ocr_conf")
        if (
            parsed_blinds is not None
            and parsed_blinds[0] > 0
            and parsed_blinds[1] > parsed_blinds[0]
            and _is_accepted(blinds_conf)
        ):
            small_blind, big_blind = parsed_blinds
            warning = None
            if _confidence_band(blinds_conf) == "usable":
                warning = "usable confidence; accepted with caution"
            diagnostics["small_blind"] = _diag_entry(
                "blinds", blinds_conf, False, warning
            )
            diagnostics["big_blind"] = _diag_entry(
                "blinds", blinds_conf, False, warning
            )
        else:
            diagnostics["small_blind"] = _diag_entry(
                "fallback",
                blinds_conf,
                True,
                "blinds OCR rejected or invalid",
            )
            diagnostics["big_blind"] = _diag_entry(
                "fallback",
                blinds_conf,
                True,
                "blinds OCR rejected or invalid",
            )
    else:
        diagnostics["small_blind"] = _diag_entry(
            "fallback", 0.0, True, "blinds object missing"
        )
        diagnostics["big_blind"] = _diag_entry(
            "fallback", 0.0, True, "blinds object missing"
        )

    hero_stack = _DEFAULT_HERO_STACK
    hero_is_all_in = False
    # tuple: (amount, conf, owner, is_all_in)
    chip_candidates: list[tuple[int, float, str | None, bool]] = []
    hero_owned_chip_candidates: list[tuple[int, float, str | None, bool]] = []
    for obj in objects:
        if _object_class(obj) != "chip_stack":
            continue
        raw_ocr = obj.get("ocr_text")
        parsed_stack = _extract_int(raw_ocr)
        is_all_in_stack = False
        if parsed_stack is None:
            if isinstance(raw_ocr, str) and _ALL_IN_RE.match(raw_ocr.strip()):
                parsed_stack = 0
                is_all_in_stack = True
            else:
                continue
        elif parsed_stack <= 0:
            continue
        candidate_conf = _field_confidence(obj, "ocr_conf")
        if _is_accepted(candidate_conf):
            owner = None
            spatial_info = obj.get("spatial_info")
            if isinstance(spatial_info, dict):
                owner_raw = spatial_info.get("owner_player")
                if isinstance(owner_raw, str):
                    owner = owner_raw.strip() or None
            candidate = (parsed_stack, candidate_conf, owner, is_all_in_stack)
            chip_candidates.append(candidate)
            if (
                hero_player_name is not None
                and owner is not None
                and owner.lower() == hero_player_name.lower()
            ):
                hero_owned_chip_candidates.append(candidate)

    selected_stack_candidates = hero_owned_chip_candidates or chip_candidates
    if selected_stack_candidates:
        selected_stack_candidates.sort(key=lambda item: item[1], reverse=True)
        selected_stack, selected_conf, _, selected_is_all_in = (
            selected_stack_candidates[0]
        )
        hero_stack = selected_stack
        hero_is_all_in = selected_is_all_in
        warning = None
        if _confidence_band(selected_conf) == "usable":
            warning = "usable confidence; accepted with caution"
        source = "chip_stack"
        if hero_owned_chip_candidates:
            source = "chip_stack.owner_player"
        elif hero_player_name is not None:
            warning = "hero chip stack owner match unavailable; using highest-confidence stack"
        diagnostics["hero_stack"] = _diag_entry(source, selected_conf, False, warning)
    else:
        diagnostics["hero_stack"] = _diag_entry(
            "fallback", 0.0, True, "chip stack OCR unavailable or low confidence"
        )

    # Accumulate opponent stacks: pick best-confidence chip_stack per seat for non-hero players.
    opponent_seat_stacks: dict[str, tuple[int, float]] = {}
    opponent_seat_all_in: set[str] = set()
    for amount, conf, owner, is_all_in_cand in chip_candidates:
        if owner is None:
            continue
        if hero_player_name is not None and owner.lower() == hero_player_name.lower():
            continue  # hero's stack handled separately
        seat_for_owner = name_lower_to_seat.get(owner.lower())
        if seat_for_owner is None:
            continue
        existing = opponent_seat_stacks.get(seat_for_owner)
        if existing is None or conf > existing[1]:
            opponent_seat_stacks[seat_for_owner] = (amount, conf)
            if is_all_in_cand:
                opponent_seat_all_in.add(seat_for_owner)
            else:
                opponent_seat_all_in.discard(seat_for_owner)

    pot = _DEFAULT_POT
    total_pot_obj = next(
        (obj for obj in objects if _object_class(obj) == "total_pot"),
        None,
    )
    if total_pot_obj is not None:
        parsed_pot = _extract_int(total_pot_obj.get("ocr_text"))
        pot_conf = _field_confidence(total_pot_obj, "ocr_conf")
        if parsed_pot is not None and parsed_pot >= 0 and _is_accepted(pot_conf):
            pot = parsed_pot
            warning = None
            if _confidence_band(pot_conf) == "usable":
                warning = "usable confidence; accepted with caution"
            diagnostics["pot"] = _diag_entry("total_pot", pot_conf, False, warning)
        else:
            diagnostics["pot"] = _diag_entry(
                "fallback",
                pot_conf,
                True,
                "total_pot OCR unavailable or low confidence",
            )
    else:
        diagnostics["pot"] = _diag_entry(
            "fallback", 0.0, True, "total_pot object missing"
        )

    amount_to_call = 0
    call_sources = ("bet", "max_bet", "min_bet")
    call_selected = False
    for source_class in call_sources:
        source_obj = next(
            (obj for obj in objects if _object_class(obj) == source_class),
            None,
        )
        if source_obj is None:
            continue
        parsed_call = _extract_int(source_obj.get("ocr_text"))
        call_conf = _field_confidence(source_obj, "ocr_conf")
        if parsed_call is None or parsed_call < 0 or not _is_accepted(call_conf):
            continue
        amount_to_call = parsed_call
        warning = None
        if _confidence_band(call_conf) == "usable":
            warning = "usable confidence; accepted with caution"
        diagnostics["amount_to_call"] = _diag_entry(
            source_class, call_conf, False, warning
        )
        call_selected = True
        break
    if not call_selected:
        diagnostics["amount_to_call"] = _diag_entry(
            "fallback",
            0.0,
            True,
            "amount-to-call OCR unavailable or low confidence",
        )

    ante_amount = _DEFAULT_ANTE
    ante_source, ante_obj = _first_object_for_classes(objects, ("ante", "bb_ante"))
    if ante_obj is not None and ante_source is not None:
        parsed_ante = _extract_int(ante_obj.get("ocr_text"))
        ante_conf = _field_confidence(ante_obj, "ocr_conf")
        if parsed_ante is not None and parsed_ante >= 0 and _is_accepted(ante_conf):
            ante_amount = parsed_ante
            diagnostics["ante_amount"] = _diag_entry(
                ante_source,
                ante_conf,
                False,
                None
                if _confidence_band(ante_conf) == "trusted"
                else "usable confidence; accepted with caution",
            )
        else:
            diagnostics["ante_amount"] = _diag_entry(
                "fallback",
                ante_conf,
                True,
                "ante OCR unavailable or low confidence",
            )
    else:
        diagnostics["ante_amount"] = _diag_entry(
            "fallback",
            0.0,
            True,
            "ante object missing",
        )

    current_blind_level: int | None = None
    level_source, level_obj = _first_object_for_classes(
        objects,
        ("blind_level", "current_blind_level"),
    )
    if level_obj is not None and level_source is not None:
        parsed_level = _extract_int(level_obj.get("ocr_text"))
        level_conf = _field_confidence(level_obj, "ocr_conf")
        if parsed_level is not None and parsed_level >= 0 and _is_accepted(level_conf):
            current_blind_level = parsed_level
            diagnostics["current_blind_level"] = _diag_entry(
                level_source,
                level_conf,
                False,
                None
                if _confidence_band(level_conf) == "trusted"
                else "usable confidence; accepted with caution",
            )
        else:
            diagnostics["current_blind_level"] = _diag_entry(
                "fallback",
                level_conf,
                True,
                "blind level OCR unavailable or low confidence",
            )
    else:
        diagnostics["current_blind_level"] = _diag_entry(
            "fallback",
            0.0,
            True,
            "blind level object missing",
        )

    seconds_until_next_level: int | None = None
    timer_source, timer_obj = _first_object_for_classes(
        objects,
        ("level_timer_seconds", "seconds_until_next_level", "blind_timer"),
    )
    if timer_obj is not None and timer_source is not None:
        parsed_timer = _extract_int(timer_obj.get("ocr_text"))
        timer_conf = _field_confidence(timer_obj, "ocr_conf")
        if parsed_timer is not None and parsed_timer >= 0 and _is_accepted(timer_conf):
            seconds_until_next_level = parsed_timer
            diagnostics["seconds_until_next_level"] = _diag_entry(
                timer_source,
                timer_conf,
                False,
                None
                if _confidence_band(timer_conf) == "trusted"
                else "usable confidence; accepted with caution",
            )
        else:
            diagnostics["seconds_until_next_level"] = _diag_entry(
                "fallback",
                timer_conf,
                True,
                "level timer OCR unavailable or low confidence",
            )
    else:
        diagnostics["seconds_until_next_level"] = _diag_entry(
            "fallback",
            0.0,
            True,
            "level timer object missing",
        )

    action_history: list[dict[str, Any]] = []
    action_confidences: list[float] = []
    for obj in objects:
        candidate_action = obj.get("action")
        candidate_player = obj.get("player")
        if not isinstance(candidate_action, str) or not isinstance(
            candidate_player, str
        ):
            continue
        parsed_amount = _extract_int(obj.get("amount"))
        entry_conf = _field_confidence(obj, None)
        if entry_conf < _MIN_ACTION_HISTORY_ENTRY_CONF:
            continue

        action_history.append(
            {
                "player": candidate_player,
                "action": candidate_action,
                "amount": parsed_amount,
            }
        )
        action_confidences.append(entry_conf)

    if action_history:
        diagnostics["action_history"] = _diag_entry(
            "detections",
            min(action_confidences),
            False,
            "action_history reconstructed from detection payload",
        )
    else:
        diagnostics["action_history"] = _diag_entry(
            "fallback", 0.0, True, "no actionable history entries"
        )

    active_candidates: list[tuple[dict[str, Any], float]] = []
    for obj in objects:
        if not bool(obj.get("turn_active")):
            continue
        halo_score = _confidence(obj.get("turn_halo_score"), fallback=0.0)
        det_conf = _confidence(obj.get("confidence"), fallback=1.0)
        active_candidates.append((obj, min(halo_score, det_conf)))

    action_on: str = "none"
    is_hero_turn = False
    # Check for hero's turn using bet_box objects
    is_hero_turn = any(_object_class(obj) == "bet_box" for obj in objects)

    # Ensure bet_box logic takes precedence for hero's turn
    if any(_object_class(obj) == "bet_box" for obj in objects):
        is_hero_turn = True
        diagnostics["is_hero_turn"] = {
            "source": "bet_box_detection",
            "value": True,
            "confidence": 1.0,
        }
    else:
        # Existing halo logic remains intact
        if active_candidates:
            active_candidates.sort(key=lambda item: item[1], reverse=True)
            active_obj, active_conf = active_candidates[0]

            active_seat = _extract_position_from_spatial(active_obj.get("spatial_info"))
            if active_seat is None:
                player_names_with_seats = [
                    obj
                    for obj in objects
                    if _object_class(obj) == "player_name"
                    and _extract_position_from_spatial(obj.get("spatial_info"))
                    is not None
                ]
                active_seat = _nearest_seat_for_object(
                    active_obj, player_names_with_seats
                )

            # If enricher set turn_active=True, trust it (threshold already validated there).
            # Just verify we have a valid seat mapping.
            if active_seat in {"BTN", "SB", "BB"}:
                action_on = active_seat
                is_hero_turn = active_seat == position
                diagnostics["is_hero_turn"] = _diag_entry(
                    "turn_halo",
                    active_conf,
                    False,
                    "halo-based turn detection (enricher threshold: 0.10)",
                )
            else:
                action_on = "unknown"
                is_hero_turn = False
                diagnostics["is_hero_turn"] = _diag_entry(
                    "fallback",
                    active_conf,
                    True,
                    f"active halo detected but seat mapping failed or invalid: {active_seat}",
                )
        else:
            diagnostics["is_hero_turn"] = _diag_entry(
                "turn_halo_none",
                0.0,
                False,
                "no active halo detected; table may be between actions",
            )

    hero_folded = False
    hero_fold_source = "fallback"
    hero_fold_warning: str | None = "no confident hero fold evidence"
    hero_fold_conf = 0.0
    for index, action in enumerate(action_history):
        if action["action"] != "fold":
            continue
        if action["player"] != position:
            continue
        entry_conf = action_confidences[index]
        if entry_conf >= _MIN_HERO_FOLD_CONF:
            hero_folded = True
            hero_fold_conf = entry_conf
            hero_fold_source = "action_history"
            hero_fold_warning = None
            break

    # Heuristic: once cards are hidden, any post-blind pot growth strongly suggests
    # hero has already folded this hand (as in screenshot_preflop_2.png).
    forced_preflop_contrib = small_blind + big_blind + (3 * ante_amount)
    hand_progressed_beyond_forced = pot > forced_preflop_contrib
    if (
        not hero_folded
        and hero_cards_visibility == "not_exposed"
        and hand_progressed_beyond_forced
    ):
        hero_folded = True
        hero_fold_conf = max(hero_fold_conf, _USABLE_THRESHOLD)
        hero_fold_source = "hidden_cards_post_blind_pot"
        hero_fold_warning = (
            "inferred fold: hero cards hidden after pot exceeded forced blinds/antes"
        )

    if hero_folded:
        diagnostics["hero_folded"] = _diag_entry(
            hero_fold_source,
            hero_fold_conf,
            False,
            hero_fold_warning,
        )
    else:
        diagnostics["hero_folded"] = _diag_entry(
            hero_fold_source,
            hero_fold_conf,
            True,
            hero_fold_warning,
        )

    hero_seat = position

    # Build seat → player name lookup from enriched player_name objects.
    # resolve_hero_position (Stage 2) sets spatial_info.seat on each player_name object.
    seat_to_player_name: dict[str, str] = {}
    if hero_player_name is not None:
        seat_to_player_name[hero_seat] = hero_player_name
    for obj in objects:
        if _object_class(obj) != "player_name":
            continue
        ocr_name = obj.get("ocr_text")
        if not isinstance(ocr_name, str) or not ocr_name.strip():
            continue
        obj_seat = _extract_position_from_spatial(obj.get("spatial_info"))
        if obj_seat is not None:
            seat_to_player_name[obj_seat] = ocr_name.strip()

    seats: list[dict[str, Any]] = []
    hero_has_cards = len(hero_cards) == 2
    for seat in ("BTN", "SB", "BB"):
        is_hero = seat == hero_seat
        if is_hero:
            seat_stack: int | None = hero_stack
            seat_is_all_in: bool | None = hero_is_all_in or None
        else:
            opp_entry = opponent_seat_stacks.get(seat)
            seat_stack = opp_entry[0] if opp_entry is not None else None
            seat_is_all_in = True if seat in opponent_seat_all_in else None
        seats.append(
            {
                "seat": seat,
                "is_hero": is_hero,
                "player_name": seat_to_player_name.get(seat),
                "status": _seat_status(
                    is_hero=is_hero,
                    is_hero_turn=is_hero_turn,
                    hero_folded=hero_folded,
                    hero_has_cards=hero_has_cards,
                    stack=seat_stack,
                    seat=seat,
                    action_on=action_on,
                ),
                "stack": seat_stack,
                "is_folded": hero_folded if is_hero else None,
                "is_all_in": seat_is_all_in,
                "has_cards": hero_has_cards if is_hero else None,
            }
        )

    hand_state = {
        "schema_version": _SCHEMA_VERSION,
        "hand_phase": hand_phase,
        "hero_cards": hero_cards,
        "hero_cards_visibility": hero_cards_visibility,
        "position": position,
        "hero_seat": hero_seat,
        "action_on": action_on,
        "big_blind": big_blind,
        "small_blind": small_blind,
        "hero_stack": hero_stack,
        "pot": pot,
        "amount_to_call": amount_to_call,
        "seats": seats,
        "tournament_status": {
            "current_blind_level": current_blind_level,
            "small_blind_amount": small_blind,
            "big_blind_amount": big_blind,
            "ante_amount": ante_amount,
            "seconds_until_next_level": seconds_until_next_level,
        },
        "action_history": action_history,
        "is_hero_turn": is_hero_turn,
        "hero_folded": hero_folded,
    }
    return hand_state, diagnostics


def build_hand_state(enriched_payload: dict[str, Any]) -> dict[str, Any]:
    """Build a minimal HandState payload from enriched detections."""
    hand_state, _ = build_hand_state_with_diagnostics(enriched_payload)
    return hand_state
