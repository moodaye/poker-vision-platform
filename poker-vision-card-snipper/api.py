"""
api.py
------
FastAPI service that accepts a screenshot and object-detection JSON,
then returns snipped images of detected objects.

Endpoints
---------
GET  /health          – liveness check
POST /snip            – snip and return results

POST /snip form fields
~~~~~~~~~~~~~~~~~~~~~~
image           : file   – PNG/JPG screenshot (required)
detections_json : string – Full detection payload JSON or bare detections list (required)
target_classes  : string – Optional JSON array of class names to snip,
                           e.g. '["flop_card","holecard"]'.
                           If omitted, all detected classes are snipped.
output_format   : string – "json" (default) | "zip"

JSON response schema
~~~~~~~~~~~~~~~~~~~~
{
  "source_image": "filename.png",
  "snip_count": 5,
  "snips": [
    {
      "class_name": "flop_card",
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
Returns a downloadable snips.zip containing files named {class_name}_{index:02d}.png
sorted by class_name then left-to-right.
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from typing import Annotated

from card_snipper import snip_objects
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError

app = FastAPI(
    title="Poker Vision Snipper",
    description="Snip object regions from a screenshot using object-detection bounding boxes.",
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_zip(snips: list) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for snip in snips:
            zf.writestr(
                f"{snip.class_name}_{snip.index:02d}.png",
                _image_to_png_bytes(snip.image),
            )
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", summary="Liveness check")
def health() -> dict:
    return {"status": "ok"}


@app.post("/snip", summary="Snip object images from a screenshot", response_model=None)
async def snip(
    image: Annotated[
        UploadFile,
        File(description="Screenshot (PNG or JPG)"),
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
    target_classes: Annotated[
        str | None,
        Form(
            description=(
                "Optional JSON array of class names to snip, "
                'e.g. \'["flop_card","holecard"]\'. '
                "If omitted, all detected classes are snipped."
            )
        ),
    ] = None,
    output_format: Annotated[
        str,
        Form(description="Response format: 'json' (default) or 'zip'"),
    ] = "json",
) -> JSONResponse | StreamingResponse:
    # ---- validate output_format -------------------------------------------
    if output_format not in ("json", "zip"):
        raise HTTPException(
            status_code=422,
            detail="output_format must be 'json' or 'zip'",
        )

    # ---- parse target_classes ---------------------------------------------
    parsed_target_classes: list[str] | None = None
    if target_classes is not None:
        try:
            parsed = json.loads(target_classes)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid target_classes JSON: {exc}"
            ) from exc
        if not isinstance(parsed, list) or not all(isinstance(c, str) for c in parsed):
            raise HTTPException(
                status_code=422,
                detail="target_classes must be a JSON array of strings",
            )
        parsed_target_classes = parsed

    # ---- parse detections_json --------------------------------------------
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
    snips = snip_objects(pil_image, detections, target_classes=parsed_target_classes)

    if not snips:
        return JSONResponse(
            content={
                "source_image": image.filename,
                "snip_count": 0,
                "snips": [],
                "message": "No matching detections found",
            }
        )

    # ---- respond ----------------------------------------------------------
    if output_format == "zip":
        zip_buf = _build_zip(snips)
        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=snips.zip"},
        )

    snip_list = []
    for s in snips:
        png_bytes = _image_to_png_bytes(s.image)
        snip_list.append(
            {
                "class_name": s.class_name,
                "index": s.index,
                "confidence": round(s.confidence, 6),
                "bbox_xyxy": s.bbox_xyxy,
                "filename": f"{s.class_name}_{s.index:02d}.png",
                "image_base64": base64.b64encode(png_bytes).decode(),
            }
        )

    return JSONResponse(
        content={
            "source_image": image.filename,
            "snip_count": len(snip_list),
            "snips": snip_list,
        }
    )
