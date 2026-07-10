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
        self.turn_halo_threshold = float(config.get("turn_halo_threshold", 0.10))
        self.turn_halo_ambiguity_delta = float(
            config.get("turn_halo_ambiguity_delta", 0.03)
        )
        # Horizontal band fractions used by _halo_score().
        # top_band: rows [top_band_lo*h, top_band_hi*h) — where the halo arc
        #   crests above the card backs in player bbox crops.
        # card_band: rows [card_band_lo*h, card_band_hi*h) — the card-back
        #   surface, used as the brightness reference baseline.
        self.halo_top_band_lo = float(config.get("halo_top_band_lo", 0.12))
        self.halo_top_band_hi = float(config.get("halo_top_band_hi", 0.28))
        self.halo_card_band_lo = float(config.get("halo_card_band_lo", 0.30))
        self.halo_card_band_hi = float(config.get("halo_card_band_hi", 0.65))
        self.halo_brightness_threshold = int(
            config.get("halo_brightness_threshold", 200)
        )
        self.classifier_url = str(config.get("classifier_url", "http://127.0.0.1:5001"))
        self.ocr_max_passes = int(config.get("ocr_max_passes", 1))
        os.makedirs(self.snip_dir, exist_ok=True)

    def _halo_score(self, image_crop: Image.Image) -> float:
        """Estimate turn-halo strength from a player bbox crop.

        Uses a horizontal band comparison:
        - Top band (rows h×top_band_lo to h×top_band_hi): where the halo arc
          crests above the card backs.  Only this region is free of card-back
          contamination while still capturing the halo glow.
        - Card band (rows h×card_band_lo to h×card_band_hi): the card-back
          surface, used as the brightness reference baseline.

        Score = max(0, bright_ratio_top − bright_ratio_card), where
        bright_ratio = count(V > brightness_threshold) / pixels_in_band.

        This detects the white/silver ring this poker client renders for the
        active player.  A saturation requirement is intentionally omitted
        because the halo is achromatic (high V, low S).
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

        w, h = resized.size
        if w < 8 or h < 8:
            return 0.0

        pixels = resized.convert("HSV").load()
        thresh = self.halo_brightness_threshold

        top_lo = int(h * self.halo_top_band_lo)
        top_hi = int(h * self.halo_top_band_hi)
        card_lo = int(h * self.halo_card_band_lo)
        card_hi = int(h * self.halo_card_band_hi)

        top_bright = sum(
            1
            for y in range(top_lo, top_hi)
            for x in range(w)
            if pixels[x, y][2] > thresh
        )
        top_total = (top_hi - top_lo) * w

        card_bright = sum(
            1
            for y in range(card_lo, card_hi)
            for x in range(w)
            if pixels[x, y][2] > thresh
        )
        card_total = (card_hi - card_lo) * w

        if top_total == 0 or card_total == 0:
            return 0.0

        top_ratio = top_bright / top_total
        card_ratio = card_bright / card_total
        score = max(0.0, top_ratio - card_ratio)

        logger.debug(
            "[halo_score] top_band=%d/%d (%.3f) | card_band=%d/%d (%.3f) | score=%.4f",
            top_bright,
            top_total,
            top_ratio,
            card_bright,
            card_total,
            card_ratio,
            score,
        )

        return round(min(1.0, score), 4)

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

    def _classify_batch(
        self, image_crops: list[Image.Image]
    ) -> list[tuple[str, float]]:
        """Classify multiple card crops in a single batched HTTP call.

        Returns a list of (label, confidence) tuples, one per input crop.
        On failure, returns default values for all crops so the caller can
        continue without interruption.
        """
        if not image_crops:
            return []

        # Encode all crops as base64 PNG
        t_encode = time.perf_counter()
        encoded_images: list[str] = []
        for crop in image_crops:
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            encoded_images.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        t_encode = time.perf_counter() - t_encode

        t_http = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.classifier_url}/classify_batch",
                json={"images": encoded_images},
                timeout=httpx.Timeout(
                    connect=0.1,
                    read=30.0,
                    write=30.0,
                    pool=0.1,
                ),
            )
            response.raise_for_status()
            data = response.json()
            t_http = time.perf_counter() - t_http

            results = data.get("results", [])
            logger.info(
                "[timing] classify_batch  n=%d  encode=%.3fs  http=%.3fs",
                len(image_crops),
                t_encode,
                t_http,
            )
            return [
                (
                    str(r.get("label", "")),
                    float(r.get("confidence", self.default_classification_conf)),
                )
                for r in results
            ]
        except Exception:
            t_http = time.perf_counter() - t_http
            logger.exception(
                "[timing] classify_batch FAILED  n=%d  encode=%.3fs  http=%.3fs",
                len(image_crops),
                t_encode,
                t_http,
            )
            return [("", self.default_classification_conf) for _ in image_crops]

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

        # First pass: crop all detections, route to processing type, and collect
        # card crops for batched classification (instead of per-card HTTP calls).
        _classify_indices: list[int] = []  # indices into enriched[] for card results
        _classify_crops: list[Image.Image] = []  # parallel list of card crops

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
                # Defer classification — collect crop for batch call after the loop
                _classify_indices.append(len(enriched))
                _classify_crops.append(crop)
                _counts["classify"] += 1
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

        # Batch-classify all card crops in a single HTTP call (replaces N serial
        # per-card calls). Results are mapped back to the enriched objects by
        # the indices collected during the loop above.
        if _classify_crops:
            t0 = time.perf_counter()
            batch_results = self._classify_batch(_classify_crops)
            elapsed = time.perf_counter() - t0
            _timing["classify"] += elapsed
            for idx, (label, conf) in zip(
                _classify_indices, batch_results, strict=False
            ):
                enriched[idx]["classification"] = label
                enriched[idx]["classification_conf"] = conf
            logger.debug("  classify batch  n=%d  %.3fs", len(_classify_crops), elapsed)

        # Infer active-turn player from halo intensity among player bboxes.
        if player_candidates:
            # Log all player scores before sorting for debugging
            for candidate in player_candidates:
                c_class = candidate.get("class_name", "unknown")
                c_score = candidate.get("turn_halo_score", 0.0)
                c_bbox = candidate.get("bbox", [])
                logger.info(
                    "[halo] candidate | class=%s score=%.4f bbox=%s",
                    c_class,
                    c_score,
                    c_bbox[:2] if len(c_bbox) >= 2 else c_bbox,
                )

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
                for candidate in sorted_candidates:
                    candidate["turn_active"] = False
                best["turn_active"] = True
                # Extract identifying info from the player who got turn_active
                best_class = best.get("class_name", "unknown")
                best_spatial = best.get("spatial_info", {})
                best_seat = (
                    best_spatial.get("seat") if isinstance(best_spatial, dict) else None
                )
                best_player_name = (
                    best_spatial.get("hero_player")
                    if isinstance(best_spatial, dict) and best_class == "player_me"
                    else "unknown"
                )
                logger.info(
                    "[halo] ACTIVE detected | best=%.4f second=%.4f delta=%.4f | "
                    "threshold=%.2f ambig_delta=%.2f | player_class=%s seat=%s name=%s | all_scores=%s",
                    best_score,
                    second_score,
                    best_score - second_score,
                    self.turn_halo_threshold,
                    self.turn_halo_ambiguity_delta,
                    best_class,
                    best_seat,
                    best_player_name,
                    [f"{c.get('turn_halo_score', 0):.4f}" for c in sorted_candidates],
                )
            else:
                for candidate in sorted_candidates:
                    candidate["turn_active"] = False
                logger.info(
                    "[halo] no active (below threshold) | best=%.4f second=%.4f delta=%.4f | "
                    "threshold=%.2f ambig_delta=%.2f | all_scores=%s",
                    best_score,
                    second_score,
                    best_score - second_score,
                    self.turn_halo_threshold,
                    self.turn_halo_ambiguity_delta,
                    [f"{c.get('turn_halo_score', 0):.4f}" for c in sorted_candidates],
                )

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
