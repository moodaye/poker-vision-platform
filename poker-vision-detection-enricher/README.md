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
    "snip_dir": "snips/"
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

Unit tests (fast — OCR is mocked, no model load):

```
uv run pytest poker-vision-detection-enricher/ -v -m "not integration"
```

Integration tests (loads real EasyOCR model — slow on first run):

```
uv run pytest poker-vision-detection-enricher/test_ocr_module.py -v -m integration -s
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

## Notes
- OCR uses [EasyOCR](https://github.com/JaidedAI/EasyOCR) with greyscale + contrast pre-processing. Validated at ~92% on real poker screenshots.
- The spatial reasoning module is implemented for `dealer_button` and `player_me`.
- The card classifier integration in `_classify_snip` is a placeholder — connects to the card classifier service at runtime.
- `snips/` is gitignored — crop output from `audit_ocr.py` is local only.
