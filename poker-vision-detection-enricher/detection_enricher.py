"""
Detection Enricher Module

This module orchestrates the enrichment of object detector outputs by routing each detected object to the appropriate processing pipeline (classification, OCR, spatial reasoning).
"""

import base64
import io
import logging
import os
import re
import time
from typing import Any

import httpx
from ocr_module import run_ocr
from PIL import Image
from spatial_reasoning import resolve_hero_position, resolve_spatial_relationships

logger = logging.getLogger(__name__)

# Matches "All In" / "All-In" / "ALL IN" etc. from text-profile OCR fallback.
_ALL_IN_RE = re.compile(r"^all[\W_]*in$", re.IGNORECASE)


class DetectionEnricher:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.save_snips = config.get("save_snips", False)
        self.snip_dir = config.get("snip_dir", "snips/")
        self.default_classification_conf = float(
            config.get("default_classification_conf", 0.65)
        )
        self.default_spatial_conf = float(config.get("default_spatial_conf", 0.70))
        self.turn_halo_threshold = float(config.get("turn_halo_threshold", 0.18))
        self.turn_halo_ambiguity_delta = float(
            config.get("turn_halo_ambiguity_delta", 0.05)
        )
        self.classifier_url = str(config.get("classifier_url", "http://127.0.0.1:5001"))
        self.ocr_max_passes = int(config.get("ocr_max_passes", 1))
        os.makedirs(self.snip_dir, exist_ok=True)

    def _halo_score(self, image_crop: Image.Image) -> float:
        """Estimate turn-halo strength from avatar-ring glow inside player boxes.

        The poker client draws a halo on the player's icon ring (mostly top arc),
        not on the outer rectangle boundary. This scorer focuses on that avatar
        region inside `player_me` / `player_other` detections.
        """
        # Downsample large crops to keep runtime bounded in service mode.
        max_side = 128
        w0, h0 = image_crop.size
        if max(w0, h0) > max_side:
            scale = max_side / max(w0, h0)
            resized = image_crop.resize(
                (max(8, int(w0 * scale)), max(8, int(h0 * scale))),
                Image.Resampling.BILINEAR,
            )
        else:
            resized = image_crop

        hsv = resized.convert("HSV")
        w, h = hsv.size
        if w < 8 or h < 8:
            return 0.0

        # Avatar circle is near upper-middle of player bbox.
        cx = w * 0.50
        cy = h * 0.42
        radius = min(w * 0.28, h * 0.34)
        if radius < 3.0:
            return 0.0

        ring_inner = radius * 0.78
        ring_outer = radius * 1.08
        core_radius = radius * 0.62

        ring_count = 0
        ring_glow = 0
        ring_v_sum = 0
        core_count = 0
        core_glow = 0
        core_v_sum = 0

        pixels = hsv.load()
        if pixels is None:
            return 0.0
        for y in range(h):
            for x in range(w):
                dx = float(x) - cx
                dy = float(y) - cy
                dist = (dx * dx + dy * dy) ** 0.5

                # Halo is most visible on upper ~2/3 arc of the avatar ring.
                in_upper_arc = dy <= (radius * 0.35)
                if not in_upper_arc:
                    continue

                pixel = pixels[x, y]
                if not isinstance(pixel, tuple) or len(pixel) != 3:
                    continue
                _, s, v = pixel
                is_glow = (v >= 165 and s >= 55) or (v >= 205 and s >= 25)

                if ring_inner <= dist <= ring_outer:
                    ring_count += 1
                    ring_v_sum += int(v)
                    if is_glow:
                        ring_glow += 1
                elif dist <= core_radius:
                    core_count += 1
                    core_v_sum += int(v)
                    if is_glow:
                        core_glow += 1

        if ring_count == 0 or core_count == 0:
            return 0.0

        ring_glow_ratio = ring_glow / ring_count
        core_glow_ratio = core_glow / core_count
        ring_v_mean = ring_v_sum / ring_count
        core_v_mean = core_v_sum / core_count

        glow_component = max(0.0, ring_glow_ratio - core_glow_ratio)
        luminance_component = max(0.0, (ring_v_mean - core_v_mean) / 255.0)
        score = 0.75 * glow_component + 0.25 * luminance_component
        return round(max(0.0, min(1.0, score)), 4)

    def _object_class(self, det: dict[str, Any]) -> str:
        return str(det.get("class") or det.get("class_name") or "unknown")

    def _classify_snip(self, image_crop: Image.Image) -> tuple[str, float]:
        buf = io.BytesIO()
        image_crop.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        try:
            response = httpx.post(
                f"{self.classifier_url}/classify",
                json={"image": encoded},
                timeout=httpx.Timeout(connect=0.1, read=5.0, write=5.0, pool=0.1),
            )
            response.raise_for_status()
            data = response.json()
            label = str(data.get("label", ""))
            confidence = float(data.get("confidence", self.default_classification_conf))
            return label, confidence
        except Exception:
            logger.exception("Classifier call failed")
            return "", self.default_classification_conf

    def _ocr_profile_for_class(self, obj_class: str) -> str:
        if obj_class == "player_name":
            return "player_name"
        if obj_class == "blinds":
            return "blinds"
        if obj_class in {"total_pot", "pot"}:
            return "total_pot"
        return "numeric"

    def _bounded_confidence(self, value: Any, fallback: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = fallback
        return max(0.0, min(1.0, parsed))

    def _get_bbox_xyxy(self, det: dict[str, Any]) -> list[int]:
        if "bbox" in det and isinstance(det["bbox"], list) and len(det["bbox"]) == 4:
            return [int(v) for v in det["bbox"]]

        if (
            "bbox_xyxy" in det
            and isinstance(det["bbox_xyxy"], list)
            and len(det["bbox_xyxy"]) == 4
        ):
            return [int(v) for v in det["bbox_xyxy"]]

        if all(key in det for key in ("x", "y", "width", "height")):
            x = float(det["x"])
            y = float(det["y"])
            w = float(det["width"])
            h = float(det["height"])
            return [
                round(x - w / 2),
                round(y - h / 2),
                round(x + w / 2),
                round(y + h / 2),
            ]

        raise ValueError("Detection must include bbox, bbox_xyxy, or x/y/width/height")

    def enrich(
        self, image: Image.Image, detections: list[dict[str, Any]]
    ) -> dict[str, Any]:
        processing_map = self.config.get("processing", {})
        enriched: list[dict[str, Any]] = []
        player_candidates: list[dict[str, Any]] = []

        # Per-step timing accumulators (seconds)
        _timing: dict[str, float] = {
            "classify": 0.0,
            "ocr": 0.0,
            "halo": 0.0,
            "crop": 0.0,
            "spatial_resolve": 0.0,
            "spatial_hero": 0.0,
        }
        _counts: dict[str, int] = {"classify": 0, "ocr": 0, "halo": 0}

        t_overall = time.perf_counter()

        for det in detections:
            obj_class = self._object_class(det)

            t0 = time.perf_counter()
            bbox = self._get_bbox_xyxy(det)
            crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            _timing["crop"] += time.perf_counter() - t0

            detection_conf = self._bounded_confidence(det.get("confidence"), 1.0)
            result: dict[str, Any] = {
                "class": obj_class,
                "class_name": obj_class,
                "bbox": bbox,
                "bbox_xyxy": bbox,
                "confidence": detection_conf,
            }

            process_type = processing_map.get(obj_class)
            if process_type == "classify":
                t0 = time.perf_counter()
                label, conf = self._classify_snip(crop)
                elapsed = time.perf_counter() - t0
                _timing["classify"] += elapsed
                _counts["classify"] += 1
                result["classification"] = label
                result["classification_conf"] = conf
                logger.debug("  classify %-15s %.3fs", obj_class, elapsed)
            elif process_type == "ocr":
                t0 = time.perf_counter()
                ocr_profile = self._ocr_profile_for_class(obj_class)
                ocr_text, ocr_conf = run_ocr(
                    crop,
                    profile=ocr_profile,
                    max_passes=self.ocr_max_passes,
                )
                # chip_stack numeric OCR strips all letters, so "All In" returns
                # empty. Retry with player_name profile and normalise if matched.
                if obj_class == "chip_stack" and not ocr_text.strip():
                    text_fb, conf_fb = run_ocr(
                        crop,
                        profile="player_name",
                        max_passes=self.ocr_max_passes,
                    )
                    if _ALL_IN_RE.match(text_fb.strip()):
                        # Regex confirmed "All In" badge; guarantee usable confidence.
                        ocr_text, ocr_conf = "All In", max(conf_fb, 0.65)
                elapsed = time.perf_counter() - t0
                _timing["ocr"] += elapsed
                _counts["ocr"] += 1
                result["ocr_text"] = ocr_text
                result["ocr_conf"] = ocr_conf
                logger.debug("  ocr      %-15s %.3fs", obj_class, elapsed)
            elif process_type == "spatial":
                pass  # handled by resolve_spatial_relationships post-pass
            else:
                result["processing"] = "none"

            if obj_class in {"player_me", "player_other"}:
                t0 = time.perf_counter()
                halo_score = self._halo_score(crop)
                elapsed = time.perf_counter() - t0
                _timing["halo"] += elapsed
                _counts["halo"] += 1
                result["turn_halo_score"] = halo_score
                player_candidates.append(result)
                logger.debug("  halo     %-15s %.3fs", obj_class, elapsed)

            if self.save_snips:
                crop.save(
                    os.path.join(self.snip_dir, f"{obj_class}_{bbox[0]}_{bbox[1]}.png")
                )
            enriched.append(result)

        # Infer active-turn player from halo intensity among player bboxes.
        if player_candidates:
            sorted_candidates = sorted(
                player_candidates,
                key=lambda obj: float(obj.get("turn_halo_score", 0.0)),
                reverse=True,
            )
            best = sorted_candidates[0]
            best_score = float(best.get("turn_halo_score", 0.0))
            second_score = (
                float(sorted_candidates[1].get("turn_halo_score", 0.0))
                if len(sorted_candidates) > 1
                else 0.0
            )
            if (
                best_score >= self.turn_halo_threshold
                and (best_score - second_score) >= self.turn_halo_ambiguity_delta
            ):
                best["turn_active"] = True
            else:
                for candidate in sorted_candidates:
                    candidate["turn_active"] = False

        # Spatial post-pass
        t0 = time.perf_counter()
        resolve_spatial_relationships(enriched, default_conf=self.default_spatial_conf)
        _timing["spatial_resolve"] = time.perf_counter() - t0

        # Hero position pass
        t0 = time.perf_counter()
        resolve_hero_position(enriched, default_conf=self.default_spatial_conf)
        _timing["spatial_hero"] = time.perf_counter() - t0

        total = time.perf_counter() - t_overall
        logger.info(
            "[timing] enrich total=%.3fs  "
            "crop=%.3fs  "
            "classify=%.3fs(n=%d)  "
            "ocr=%.3fs(n=%d)  "
            "halo=%.3fs(n=%d)  "
            "spatial_resolve=%.3fs  "
            "spatial_hero=%.3fs",
            total,
            _timing["crop"],
            _timing["classify"],
            _counts["classify"],
            _timing["ocr"],
            _counts["ocr"],
            _timing["halo"],
            _counts["halo"],
            _timing["spatial_resolve"],
            _timing["spatial_hero"],
        )

        return {"objects": enriched}


# Example usage (for test script):
# config = {"processing": {"card": "classify", "chip_stack": "ocr", "dealer_button": "spatial"}, "save_snips": True}
# enricher = DetectionEnricher(config)
# result = enricher.enrich(image, detections)
