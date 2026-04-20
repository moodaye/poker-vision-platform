from __future__ import annotations

import re
from typing import Any

_CARD_LABEL_RE = re.compile(r"^[2-9TJQKA][cdhs]$")
_FALLBACK_HERO_CARDS = ["Ah", "Kd"]
_DEFAULT_SMALL_BLIND = 50
_DEFAULT_BIG_BLIND = 100
_DEFAULT_HERO_STACK = 3000
_DEFAULT_POT = 150
_TRUSTED_THRESHOLD = 0.80
_USABLE_THRESHOLD = 0.55
_MIN_DETECTION_FOR_CARDS = 0.60
_MIN_CLASSIFICATION_FOR_CARDS = 0.70
_MIN_ACTION_HISTORY_ENTRY_CONF = 0.65
_MIN_HERO_FOLD_CONF = 0.70


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
    if _CARD_LABEL_RE.fullmatch(normalized):
        return normalized
    return None


def _extract_int(value: Any) -> int | None:
    if isinstance(value, int | float):
        return int(value)
    if not isinstance(value, str):
        return None

    match = re.search(r"\d+", value.replace(",", ""))
    if match is None:
        return None
    return int(match.group(0))


def _extract_blinds(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None

    matches = re.findall(r"\d+", value.replace(",", ""))
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


def _collect_hero_cards(objects: list[dict[str, Any]]) -> list[str]:
    hero_cards: list[str] = []

    preferred_classes = ("holecard", "card")
    for target_class in preferred_classes:
        for obj in objects:
            if _object_class(obj) != target_class:
                continue
            label = _valid_card_label(obj.get("classification"))
            if label is not None:
                hero_cards.append(label)
            if len(hero_cards) == 2:
                return hero_cards

    for obj in objects:
        label = _valid_card_label(obj.get("classification"))
        if label is not None:
            hero_cards.append(label)
        if len(hero_cards) == 2:
            return hero_cards

    return _FALLBACK_HERO_CARDS.copy()


def build_hand_state_with_diagnostics(
    enriched_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    raw_objects = enriched_payload.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("enriched payload must contain an 'objects' list")

    objects = [obj for obj in raw_objects if isinstance(obj, dict)]
    diagnostics: dict[str, dict[str, Any]] = {}

    # hero_cards: prefer holecard, then card
    hero_card_candidates: list[tuple[str, float, str, str | None]] = []
    for target_class in ("holecard", "card"):
        for obj in objects:
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
            hero_card_candidates.append((label, field_conf, target_class, warning))

    if len(hero_card_candidates) >= 2:
        hero_cards = [hero_card_candidates[0][0], hero_card_candidates[1][0]]
        hero_cards_conf = min(hero_card_candidates[0][1], hero_card_candidates[1][1])
        source = f"{hero_card_candidates[0][2]}+{hero_card_candidates[1][2]}"
        warning = hero_card_candidates[0][3] or hero_card_candidates[1][3]
        diagnostics["hero_cards"] = _diag_entry(source, hero_cards_conf, False, warning)
    else:
        hero_cards = _FALLBACK_HERO_CARDS.copy()
        diagnostics["hero_cards"] = _diag_entry(
            "fallback",
            0.0,
            True,
            "insufficient card candidates passed confidence gates",
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
    dealer_pos = (
        _extract_position_from_spatial(dealer_obj.get("spatial_info"))
        if dealer_obj is not None
        else None
    )
    if player_pos is not None and player_me_obj is not None:
        if dealer_obj is not None:
            position_conf = min(
                _field_confidence(player_me_obj, "spatial_conf"),
                _field_confidence(dealer_obj, "spatial_conf"),
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
            position_conf = _field_confidence(player_me_obj, "spatial_conf")
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
        dealer_conf = _field_confidence(dealer_obj, "spatial_conf")
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
    chip_candidates: list[tuple[int, float]] = []
    for obj in objects:
        if _object_class(obj) != "chip_stack":
            continue
        parsed_stack = _extract_int(obj.get("ocr_text"))
        if parsed_stack is None or parsed_stack <= 0:
            continue
        candidate_conf = _field_confidence(obj, "ocr_conf")
        if _is_accepted(candidate_conf):
            chip_candidates.append((parsed_stack, candidate_conf))

    if chip_candidates:
        chip_candidates.sort(key=lambda item: item[1], reverse=True)
        hero_stack = chip_candidates[0][0]
        warning = None
        if _confidence_band(chip_candidates[0][1]) == "usable":
            warning = "usable confidence; accepted with caution"
        diagnostics["hero_stack"] = _diag_entry(
            "chip_stack", chip_candidates[0][1], False, warning
        )
    else:
        diagnostics["hero_stack"] = _diag_entry(
            "fallback", 0.0, True, "chip stack OCR unavailable or low confidence"
        )

    pot = _DEFAULT_POT
    pot_sources = ("total_pot", "pot", "pot_bet")
    pot_selected = False
    for source_class in pot_sources:
        source_obj = next(
            (obj for obj in objects if _object_class(obj) == source_class),
            None,
        )
        if source_obj is None:
            continue
        parsed_pot = _extract_int(source_obj.get("ocr_text"))
        pot_conf = _field_confidence(source_obj, "ocr_conf")
        if parsed_pot is None or parsed_pot < 0 or not _is_accepted(pot_conf):
            continue
        pot = parsed_pot
        warning = None
        if _confidence_band(pot_conf) == "usable":
            warning = "usable confidence; accepted with caution"
        diagnostics["pot"] = _diag_entry(source_class, pot_conf, False, warning)
        pot_selected = True
        break
    if not pot_selected:
        diagnostics["pot"] = _diag_entry(
            "fallback", 0.0, True, "pot OCR unavailable or low confidence"
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

    hero_turn_controls = {
        "check_button",
        "check_fold_button",
        "fold_button",
        "raise_button",
        "bet_pot_button",
    }
    control_confidences: list[tuple[str, float]] = []
    for obj in objects:
        obj_class = _object_class(obj)
        if obj_class not in hero_turn_controls:
            continue
        control_confidences.append((obj_class, _field_confidence(obj, None)))

    is_hero_turn = True
    if control_confidences:
        control_confidences.sort(key=lambda item: item[1], reverse=True)
        best_control, best_control_conf = control_confidences[0]
        if _is_accepted(best_control_conf):
            diagnostics["is_hero_turn"] = _diag_entry(
                best_control,
                best_control_conf,
                False,
                None
                if _confidence_band(best_control_conf) == "trusted"
                else "usable confidence; accepted with caution",
            )
        else:
            diagnostics["is_hero_turn"] = _diag_entry(
                "fallback",
                best_control_conf,
                True,
                "hero controls found but below confidence threshold",
            )
    else:
        diagnostics["is_hero_turn"] = _diag_entry(
            "fallback",
            0.0,
            True,
            "no hero action controls detected",
        )

    hero_folded = False
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
            break

    if hero_folded:
        diagnostics["hero_folded"] = _diag_entry(
            "action_history",
            hero_fold_conf,
            False,
            None,
        )
    else:
        diagnostics["hero_folded"] = _diag_entry(
            "fallback",
            hero_fold_conf,
            True,
            "no confident hero fold evidence",
        )

    hand_state = {
        "hero_cards": hero_cards,
        "position": position,
        "big_blind": big_blind,
        "small_blind": small_blind,
        "hero_stack": hero_stack,
        "pot": pot,
        "amount_to_call": amount_to_call,
        "action_history": action_history,
        "is_hero_turn": is_hero_turn,
        "hero_folded": hero_folded,
    }
    return hand_state, diagnostics


def build_hand_state(enriched_payload: dict[str, Any]) -> dict[str, Any]:
    """Build a minimal HandState payload from enriched detections."""
    hand_state, _ = build_hand_state_with_diagnostics(enriched_payload)
    return hand_state
