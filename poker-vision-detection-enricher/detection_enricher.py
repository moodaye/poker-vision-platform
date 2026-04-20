"""
Detection Enricher Module

This module orchestrates the enrichment of object detector outputs by routing each detected object to the appropriate processing pipeline (classification, OCR, spatial reasoning).
"""

import os
from typing import Any

from ocr_module import run_ocr
from PIL import Image
from spatial_reasoning import assign_dealer


class DetectionEnricher:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.save_snips = config.get("save_snips", False)
        self.snip_dir = config.get("snip_dir", "snips/")
        self.default_classification_conf = float(
            config.get("default_classification_conf", 0.65)
        )
        self.default_ocr_conf = float(config.get("default_ocr_conf", 0.60))
        self.default_spatial_conf = float(config.get("default_spatial_conf", 0.70))
        os.makedirs(self.snip_dir, exist_ok=True)

    def _object_class(self, det: dict[str, Any]) -> str:
        return str(det.get("class") or det.get("class_name") or "unknown")

    def _classify_snip(self, image_crop: Image.Image) -> str:
        # Placeholder classifier integration point.
        _ = image_crop
        return "<classification_result>"

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

        for det in detections:
            obj_class = self._object_class(det)
            bbox = self._get_bbox_xyxy(det)
            crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
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
                result["classification"] = self._classify_snip(crop)
                result["classification_conf"] = self.default_classification_conf
            elif process_type == "ocr":
                result["ocr_text"] = run_ocr(crop)
                result["ocr_conf"] = self.default_ocr_conf
            elif process_type == "spatial":
                result["spatial_info"] = assign_dealer(det, detections)
                result["spatial_conf"] = self.default_spatial_conf
            else:
                result["processing"] = "none"

            if self.save_snips:
                crop.save(
                    os.path.join(self.snip_dir, f"{obj_class}_{bbox[0]}_{bbox[1]}.png")
                )
            enriched.append(result)

        # Aggregate results into JSON for game state parser
        return {"objects": enriched}


# Example usage (for test script):
# config = {"processing": {"card": "classify", "chip_stack": "ocr", "dealer_button": "spatial"}, "save_snips": True}
# enricher = DetectionEnricher(config)
# result = enricher.enrich(image, detections)
