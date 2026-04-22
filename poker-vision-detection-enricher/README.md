# Detection Enricher

## Overview

The `detection_enricher` module processes the output of an object detector for poker table images. It routes each detected object to the appropriate processing pipeline (classification, OCR, or spatial reasoning), aggregates the results, and outputs a structured JSON suitable for the game state parser.

## Design

- **Modular:** Classification, OCR, and spatial reasoning are handled by separate modules/components.
- **Configurable:** Processing logic for each object type is determined by configuration.
- **Snipping:** Crops (snips) are passed in memory to the next module, with optional saving for debugging.
- **API-driven:** Designed for integration as an API or as a library.

## Usage

1. Configure the processing logic for each object type in a config dictionary.
2. Instantiate `DetectionEnricher` with the config.
3. Call `enrich(image, detections)` with a PIL image and a list of detection dicts.
4. The result is a JSON object ready for the game state parser.

## Example

```python
from PIL import Image
from detection_enricher import DetectionEnricher

config = {
    "processing": {
        "card": "classify",
        "chip_stack": "ocr",
        "dealer_button": "spatial"
    },
    "save_snips": True,
    "snip_dir": "snips/",
    "classifier_url": "http://localhost:5001"  # card classifier service
}
enricher = DetectionEnricher(config)
image = Image.open("example.png")
detections = [
    {"class": "card", "bbox": [10, 10, 60, 90]},
    {"class": "chip_stack", "bbox": [70, 10, 120, 60]},
    {"class": "dealer_button", "bbox": [130, 10, 180, 60]}
]
result = enricher.enrich(image, detections)
print(result)
```

## Testing

Unit and OCR tests (all fast — pytesseract has no model load):

```
uv run pytest poker-vision-detection-enricher/ -v
```

OCR audit against real screenshots (crops and OCR results saved to `snips/ocr_audit/`):

```
uv run python poker-vision-detection-enricher/audit_ocr.py
```

## API

Start the API service:

```
uv run python .\poker-vision-detection-enricher\api.py
```

Endpoints:

- `GET /health` returns `{ "status": "ok" }`
- `POST /enrich` accepts JSON with:
    - `image_base64`: base64-encoded screenshot image bytes
    - `detections`: detector predictions list
    - `config` (optional): processing config override

Example `POST /enrich` payload:

```json
{
    "image_base64": "<base64 image bytes>",
    "detections": [
        {"class": "card", "bbox": [10, 10, 60, 90], "confidence": 0.99},
        {"class": "chip_stack", "bbox": [70, 10, 120, 60], "confidence": 0.94},
        {"class": "dealer_button", "bbox": [130, 10, 180, 60], "confidence": 0.97}
    ]
}
```

## Performance

### OCR bottleneck — EasyOCR → pytesseract

**Problem:** The initial OCR implementation used [EasyOCR](https://github.com/JaidedAI/EasyOCR), a deep-learning OCR library. This introduced two performance problems:

1. **Cold-start latency (~20–60 s on CPU):** EasyOCR loads a neural network model into memory on the first call. This blocks the entire enricher response until the model is ready.
2. **Per-crop inference (~2–5 s on CPU):** Each chip stack or blind value crop is run through the model independently. With 4–8 numeric regions per screenshot, OCR alone accounts for 10–40 s of pipeline latency.

These figures make real-time poker assistance impractical — the full pipeline took ~60 s cold and ~30 s warm, against a target of <10 s.

**Root cause:** EasyOCR is a general-purpose deep-learning OCR model. Its power comes at the cost of model load time and neural inference overhead. For a narrow, well-defined task (reading digits and `/` from a poker HUD), this power is unnecessary.

**Fix:** Replace EasyOCR with [pytesseract](https://github.com/madmaze/pytesseract) — a Python wrapper around the Tesseract OCR binary (C library). Key differences:

| Property | EasyOCR | pytesseract |
|---|---|---|
| Mechanism | Deep neural network (LSTM) | Classical image analysis (C binary) |
| Cold start | 20–60 s (model load) | ~0 s (binary already in memory) |
| Per-crop latency | 2–5 s | 10–50 ms |
| Setup | `pip install easyocr` | `pip install pytesseract` + Tesseract binary |
| Config needed | Minimal | PSM mode + character whitelist |

For our use case (reading `"470"`, `"1/2"` etc.) pytesseract with `--psm 7 -c tessedit_char_whitelist=0123456789/` achieves comparable accuracy to EasyOCR with ~100× less latency per crop.

**Trade-off:** pytesseract requires the Tesseract binary to be installed on the host machine (not just a pip package), and needs explicit configuration (PSM mode, character whitelist) to work reliably on game UI crops. EasyOCR requires no configuration but is impractical for real-time use on CPU.

## Notes
- OCR uses [pytesseract](https://github.com/madmaze/pytesseract) with greyscale + contrast pre-processing. Requires Tesseract binary installed on host.
- The spatial reasoning module is implemented for `dealer_button` and `player_me`.
- Card classification calls the card classifier service (`POST /classify`) at `classifier_url` (default `http://localhost:5001`). On any connection or HTTP error the label falls back to `""` and the default confidence is used.
- `snips/` is gitignored — crop output from `audit_ocr.py` is local only.
