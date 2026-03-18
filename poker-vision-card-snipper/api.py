"""
api.py
------
FastAPI service that accepts a poker-table screenshot and object-detection JSON,
then returns snipped images of every detected flop card.

Endpoints
---------
GET  /health          – liveness check
POST /snip            – snip and return results

POST /snip form fields
~~~~~~~~~~~~~~~~~~~~~~
image           : file   – PNG/JPG screenshot (required)
detections_json : string – Full detection payload JSON or bare detections list (required)
output_format   : string – "json" (default) | "zip"

JSON response schema
~~~~~~~~~~~~~~~~~~~~
{
  "source_image": "filename.png",
  "card_count": 5,
  "flop_cards": [
    {
      "index": 0,
      "confidence": 0.995522,
      "bbox_xyxy": [762, 388, 850, 513],
      "filename": "flop_card_00.png",
      "image_base64": "<base64-encoded PNG bytes>"
    },
    ...
  ]
}

ZIP response
~~~~~~~~~~~~
Returns a downloadable flop_cards.zip containing flop_card_00.png … flop_card_NN.png
sorted left-to-right across the table.
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from typing import Annotated

from card_snipper import snip_flop_cards
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError

app = FastAPI(
    title="Poker Card Snipper",
    description="Snip flop card images from a poker-table screenshot using object-detection bounding boxes.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_zip(card_snips) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for card in card_snips:
            zf.writestr(
                f"flop_card_{card.index:02d}.png", _image_to_png_bytes(card.image)
            )
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", summary="Liveness check")
def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/snip", summary="Snip flop card images from a screenshot", response_model=None
)
async def snip(
    image: Annotated[
        UploadFile,
        File(description="Poker-table screenshot (PNG or JPG)"),
    ],
    detections_json: Annotated[
        str,
        Form(
            description=(
                "Object-detection result JSON. Either the full payload "
                "(containing a 'detections' key) or a bare JSON array of detections."
            )
        ),
    ],
    output_format: Annotated[
        str,
        Form(description="Response format: 'json' (default) or 'zip'"),
    ] = "json",
) -> JSONResponse | StreamingResponse:
    # ---- validate output format -------------------------------------------
    if output_format not in ("json", "zip"):
        raise HTTPException(
            status_code=422,
            detail="output_format must be 'json' or 'zip'",
        )

    # ---- parse detections JSON --------------------------------------------
    try:
        payload = json.loads(detections_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {exc}") from exc

    if isinstance(payload, dict):
        detections = payload.get("detections")
        if detections is None:
            raise HTTPException(
                status_code=422,
                detail="JSON object must contain a 'detections' key",
            )
    elif isinstance(payload, list):
        detections = payload
    else:
        raise HTTPException(
            status_code=422,
            detail="detections_json must be a JSON object with a 'detections' key, or a JSON array",
        )

    # ---- load image -------------------------------------------------------
    raw_bytes = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=422, detail="Cannot decode image file") from exc

    # ---- snip -------------------------------------------------------------
    card_snips = snip_flop_cards(pil_image, detections)

    if not card_snips:
        return JSONResponse(
            content={
                "source_image": image.filename,
                "card_count": 0,
                "flop_cards": [],
                "message": "No flop_card detections found in the provided JSON",
            }
        )

    # ---- respond ----------------------------------------------------------
    if output_format == "zip":
        zip_buf = _build_zip(card_snips)
        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=flop_cards.zip"},
        )

    # JSON response with base64-encoded PNGs
    flop_cards = []
    for card in card_snips:
        png_bytes = _image_to_png_bytes(card.image)
        flop_cards.append(
            {
                "index": card.index,
                "confidence": round(card.confidence, 6),
                "bbox_xyxy": card.bbox_xyxy,
                "filename": f"flop_card_{card.index:02d}.png",
                "image_base64": base64.b64encode(png_bytes).decode(),
            }
        )

    return JSONResponse(
        content={
            "source_image": image.filename,
            "card_count": len(flop_cards),
            "flop_cards": flop_cards,
        }
    )
