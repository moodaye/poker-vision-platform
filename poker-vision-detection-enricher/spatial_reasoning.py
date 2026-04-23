"""
Spatial Reasoning Module

This module resolves relationships between detected objects that cannot be
determined from a single bounding box in isolation — they require positional
reasoning across multiple detections.

The three spatial reasoning tasks are:

1. Dealer identification — dealer_button → nearest player_name
   The dealer is the player whose name label is spatially closest to the
   dealer_button object. "Closest" is measured as the Euclidean distance
   between bounding-box centres.

2. Chip stack → player association — chip_stack below player_name
   Each chip_stack belongs to the player_name object whose centre lies
   directly above it (smallest vertical offset, within a horizontal
   proximity threshold).

3. Bet amount → player association — bet/pot_bet in front of player
   Each bet or pot_bet belongs to the player_name object that is spatially
   nearest to it. This allows per-player bet tracking and action history
   reconstruction.

Entry points:
  resolve_spatial_relationships(enriched_objects)
    Called first — annotates dealer_button, chip_stack, bet/pot_bet with
    their associated player names.

  resolve_hero_position(enriched_objects)
    Called after resolve_spatial_relationships — uses dealer_button's
    spatial_info (populated by the first pass) plus clockwise seat ordering
    to determine the hero's position (BTN/SB/BB) and annotates player_me.

NOTE: Hero position is stable for the entire hand and is a candidate for
hand-scoped caching. See Architecture Considerations in README.
"""

import math
from typing import Any

# Maximum horizontal pixel offset for chip_stack → player_name association.
# A chip stack further than this from the player name centre is not matched.
_HORIZONTAL_THRESHOLD = 150.0


def _bbox_centre(obj: dict[str, Any]) -> tuple[float, float]:
    bbox = obj.get("bbox_xyxy") or obj.get("bbox")
    if not bbox or len(bbox) < 4:
        raise ValueError(f"Object has no valid bbox: {obj}")
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _nearest_by_euclidean(
    obj: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the candidate whose centre is nearest to obj by Euclidean distance."""
    if not candidates:
        return None
    centre = _bbox_centre(obj)
    return min(candidates, key=lambda c: _euclidean(centre, _bbox_centre(c)))


def _nearest_above(
    obj: dict[str, Any],
    candidates: list[dict[str, Any]],
    horizontal_threshold: float = _HORIZONTAL_THRESHOLD,
) -> dict[str, Any] | None:
    """Return the candidate directly above obj within horizontal_threshold pixels.

    'Above' means the candidate centre has a smaller y value (higher on screen).
    Among qualifying candidates, the one with the smallest vertical distance wins.
    """
    cx, cy = _bbox_centre(obj)
    best: dict[str, Any] | None = None
    best_vert_dist = float("inf")
    for candidate in candidates:
        pcx, pcy = _bbox_centre(candidate)
        if pcy >= cy:
            continue  # candidate must be above (smaller y)
        if abs(pcx - cx) > horizontal_threshold:
            continue  # too far left or right
        vert_dist = cy - pcy
        if vert_dist < best_vert_dist:
            best_vert_dist = vert_dist
            best = candidate
    return best


def resolve_spatial_relationships(
    enriched_objects: list[dict[str, Any]],
    default_conf: float = 0.70,
) -> None:
    """Resolve spatial relationships across the fully enriched object list.

    Mutates enriched_objects in-place, adding spatial_info and spatial_conf to:
    - dealer_button: annotated with the nearest player_name (the dealer)
    - chip_stack:    annotated with the player_name directly above it (the owner)
    - bet / pot_bet: annotated with the nearest player_name (who placed the bet)

    Must be called after all per-object enrichment so that player_name OCR text
    is populated and available as association labels.
    """
    player_names = [
        obj
        for obj in enriched_objects
        if (obj.get("class_name") or obj.get("class")) == "player_name"
    ]

    for obj in enriched_objects:
        cls = obj.get("class_name") or obj.get("class") or ""

        if cls == "dealer_button":
            nearest = _nearest_by_euclidean(obj, player_names)
            if nearest is not None:
                obj["spatial_info"] = {"dealer_player": nearest.get("ocr_text", "")}
                obj["spatial_conf"] = default_conf
            else:
                obj["spatial_info"] = {}
                obj["spatial_conf"] = 0.0

        elif cls == "chip_stack":
            nearest = _nearest_above(obj, player_names)
            if nearest is not None:
                obj["spatial_info"] = {"owner_player": nearest.get("ocr_text", "")}
                obj["spatial_conf"] = default_conf
            else:
                obj["spatial_info"] = {}
                obj["spatial_conf"] = 0.0

        elif cls in ("bet", "pot_bet"):
            nearest = _nearest_by_euclidean(obj, player_names)
            if nearest is not None:
                obj["spatial_info"] = {"owner_player": nearest.get("ocr_text", "")}
                obj["spatial_conf"] = default_conf
            else:
                obj["spatial_info"] = {}
                obj["spatial_conf"] = 0.0


def _clockwise_seat_order(
    player_names: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return player_name objects sorted clockwise from the bottom of the table.

    Uses the centroid of all player bbox centres as the reference point.
    'Clockwise from bottom' means the bottom-most player (largest y on screen)
    sorts first, then going right, top, left — matching the visual clockwise
    direction as seen on screen (where y increases downward).

    For a standard 3-player online poker layout this produces:
      [hero (bottom), opponent (top-right), opponent (top-left)]
    """
    if len(player_names) <= 1:
        return list(player_names)

    centres = [_bbox_centre(p) for p in player_names]
    cx_ref = sum(c[0] for c in centres) / len(centres)
    cy_ref = sum(c[1] for c in centres) / len(centres)

    def _key(obj: dict[str, Any]) -> float:
        cx, cy = _bbox_centre(obj)
        dx = cx - cx_ref
        dy = cy - cy_ref
        # atan2(dy, dx) in screen coords (y increases downward):
        #   right=0°, down=90°, left=180°, up=270°
        # Shift so bottom (90°) maps to 0° and clockwise increases from there.
        angle_deg = math.degrees(math.atan2(dy, dx)) % 360.0
        return (90.0 - angle_deg) % 360.0

    return sorted(player_names, key=_key)


def resolve_hero_position(
    enriched_objects: list[dict[str, Any]],
    default_conf: float = 0.70,
) -> None:
    """Determine the hero's table position (BTN/SB/BB) using clockwise seat ordering.

    Algorithm:
    1. Sort all player_name bboxes clockwise from the bottom of the screen to
       establish a canonical seat order.
    2. Find the dealer's seat index by matching dealer_button.spatial_info.dealer_player
       (case-insensitive) against the OCR text of each player_name.
    3. Find the hero's seat index as the player_name nearest to the player_me bbox.
    4. Compute the clockwise offset from dealer to hero and map to position:
         offset 0 → BTN,  offset 1 → SB,  offset 2 → BB
       Heads-up special case (2 players): offset 0 → BTN, offset 1 → BB.

    Annotates the player_me object in-place:
        spatial_info = {"position": "BTN" | "SB" | "BB"}
        spatial_conf = default_conf

    Must be called after resolve_spatial_relationships so that
    dealer_button.spatial_info.dealer_player is already populated.

    NOTE: Hero position is stable for the entire hand and is a candidate for
    hand-scoped caching — no need to recompute on every frame once resolved.
    See Architecture Considerations in README.
    """
    player_names = [
        obj
        for obj in enriched_objects
        if (obj.get("class_name") or obj.get("class")) == "player_name"
    ]
    player_me_obj = next(
        (
            obj
            for obj in enriched_objects
            if (obj.get("class_name") or obj.get("class")) == "player_me"
        ),
        None,
    )
    dealer_button_obj = next(
        (
            obj
            for obj in enriched_objects
            if (obj.get("class_name") or obj.get("class")) == "dealer_button"
        ),
        None,
    )

    if player_me_obj is None or dealer_button_obj is None or len(player_names) < 2:
        return

    dealer_player_name = (dealer_button_obj.get("spatial_info") or {}).get(
        "dealer_player", ""
    )
    if not dealer_player_name:
        return

    ordered = _clockwise_seat_order(player_names)
    num_players = len(ordered)

    # Find dealer seat index by OCR text match (case-insensitive)
    dealer_idx = next(
        (
            i
            for i, p in enumerate(ordered)
            if (p.get("ocr_text") or "").strip().lower()
            == dealer_player_name.strip().lower()
        ),
        None,
    )
    if dealer_idx is None:
        return

    # Find hero seat index: player_name nearest to the player_me bbox
    hero_seat = _nearest_by_euclidean(player_me_obj, ordered)
    if hero_seat is None:
        return
    hero_idx = ordered.index(hero_seat)

    offset = (hero_idx - dealer_idx) % num_players

    if num_players == 2:
        # Heads-up: dealer is BTN; the other player is BB
        position: str | None = "BTN" if offset == 0 else "BB"
    else:
        position = {0: "BTN", 1: "SB", 2: "BB"}.get(offset)

    if position is None:
        return  # > 3 players not yet supported

    player_me_obj["spatial_info"] = {"position": position}
    player_me_obj["spatial_conf"] = default_conf
