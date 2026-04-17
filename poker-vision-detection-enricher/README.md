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

Run the test script:

```
uv run python .\poker-vision-detection-enricher\test_detection_enricher.py
```

Run API smoke tests:

```
uv run python .\poker-vision-detection-enricher\test_api_detection_enricher.py
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
- The OCR and spatial reasoning modules are placeholders and should be implemented as needed.
- The game state parser API is TBD.

## TODO

1. Ensure all APIs are driven by Flask and have a consistent level of logging.
2. Ensure all submodules are using consistent libraries.
3. Ensure all submodules have adequate test coverage.
