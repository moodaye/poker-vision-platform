"""
API service for detection enricher.

Run:
    uv run python .\\poker-vision-detection-enricher\\api.py

Endpoints:
    GET  /health
    POST /enrich
"""

from __future__ import annotations

import base64
import io
from typing import Any

from detection_enricher import DetectionEnricher
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

app = FastAPI(
    title="Poker Vision Detection Enricher",
    description="Routes detections to classification, OCR, and spatial reasoning.",
    version="0.1.0",
)


class EnrichRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description="Base64-encoded screenshot bytes. Supports optional data URI prefix.",
    )
    detections: list[dict[str, Any]] = Field(
        ..., description="Object detector predictions."
    )
    config: dict[str, Any] | None = Field(
        default=None,
        description="Optional processing config override for this request.",
    )


class EnrichResponse(BaseModel):
    objects: list[dict[str, Any]]


def _decode_image(image_base64: str) -> Image.Image:
    payload = image_base64
    if "," in payload and payload.strip().lower().startswith("data:"):
        payload = payload.split(",", 1)[1]

    try:
        raw = base64.b64decode(payload)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422, detail="Invalid image_base64 payload"
        ) from exc

    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=422, detail="Decoded bytes are not a valid image"
        ) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/enrich", response_model=EnrichResponse)
def enrich(payload: EnrichRequest) -> EnrichResponse:
    if not payload.detections:
        raise HTTPException(status_code=422, detail="detections cannot be empty")

    image = _decode_image(payload.image_base64)

    config = payload.config or {
        "processing": {
            "card": "classify",
            "flop_card": "classify",
            "turn_card": "classify",
            "river_card": "classify",
            "chip_stack": "ocr",
            "player_name": "ocr",
            "dealer_button": "spatial",
        },
        "save_snips": False,
        "snip_dir": "snips/",
    }

    enricher = DetectionEnricher(config)
    result = enricher.enrich(image, payload.detections)
    return EnrichResponse(**result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5004)
