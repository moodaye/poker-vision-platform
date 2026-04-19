from __future__ import annotations

import re
from typing import Any

_CARD_LABEL_RE = re.compile(r"^[2-9TJQKA][cdhs]$")
_FALLBACK_HERO_CARDS = ["Ah", "Kd"]
_DEFAULT_SMALL_BLIND = 50
_DEFAULT_BIG_BLIND = 100
_DEFAULT_HERO_STACK = 3000
_DEFAULT_POT = 150


def _object_class(obj: dict[str, Any]) -> str:
    return str(obj.get("class_name") or obj.get("class") or "unknown")


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


def build_hand_state(enriched_payload: dict[str, Any]) -> dict[str, Any]:
    """Build a minimal HandState payload from enriched detections.

    This parser is deliberately conservative for MVP wiring. It derives values
    from enriched detections when possible and falls back to explicit defaults
    for fields that the vision layer does not resolve yet.
    """
    raw_objects = enriched_payload.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("enriched payload must contain an 'objects' list")

    objects = [obj for obj in raw_objects if isinstance(obj, dict)]

    hero_cards = _collect_hero_cards(objects)
    small_blind = _DEFAULT_SMALL_BLIND
    big_blind = _DEFAULT_BIG_BLIND
    hero_stack = _DEFAULT_HERO_STACK
    pot = _DEFAULT_POT
    amount_to_call = 0
    is_hero_turn = True

    for obj in objects:
        obj_class = _object_class(obj)
        ocr_text = obj.get("ocr_text")

        if obj_class == "blinds":
            blinds = _extract_blinds(ocr_text)
            if blinds is not None:
                small_blind, big_blind = blinds
        elif obj_class == "chip_stack":
            parsed_stack = _extract_int(ocr_text)
            if parsed_stack is not None:
                hero_stack = parsed_stack
        elif obj_class in {"pot", "total_pot", "pot_bet"}:
            parsed_pot = _extract_int(ocr_text)
            if parsed_pot is not None:
                pot = parsed_pot
        elif obj_class in {"bet", "max_bet", "min_bet"}:
            parsed_call = _extract_int(ocr_text)
            if parsed_call is not None:
                amount_to_call = parsed_call
        elif obj_class in {
            "check_button",
            "check_fold_button",
            "fold_button",
            "raise_button",
            "bet_pot_button",
        }:
            is_hero_turn = True

    return {
        "hero_cards": hero_cards,
        "position": "BTN",
        "big_blind": big_blind,
        "small_blind": small_blind,
        "hero_stack": hero_stack,
        "pot": pot,
        "amount_to_call": amount_to_call,
        "action_history": [],
        "is_hero_turn": is_hero_turn,
        "hero_folded": False,
    }
