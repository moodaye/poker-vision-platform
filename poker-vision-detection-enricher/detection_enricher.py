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
        os.makedirs(self.snip_dir, exist_ok=True)

    def _classify_snip(self, image_crop: Image.Image) -> str:
        # Placeholder classifier integration point.
        _ = image_crop
        return "<classification_result>"

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
            obj_class = str(det.get("class", "unknown"))
            bbox = self._get_bbox_xyxy(det)
            crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            result: dict[str, Any] = {
                "class": obj_class,
                "bbox": bbox,
                "confidence": det.get("confidence"),
            }

            process_type = processing_map.get(obj_class)
            if process_type == "classify":
                result["classification"] = self._classify_snip(crop)
            elif process_type == "ocr":
                result["ocr_text"] = run_ocr(crop)
            elif process_type == "spatial":
                result["spatial_info"] = assign_dealer(det, detections)
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
