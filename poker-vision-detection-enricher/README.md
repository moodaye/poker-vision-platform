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
    "classifier_url": "http://127.0.0.1:5001",  # card classifier service
    "ocr_max_passes": 1,  # single-pass OCR by default for speed
}
enricher = DetectionEnricher(config)
image = Image.open("example.png")
# Canonical format (object detector normalized output)
detections = [
    {"class_name": "card",          "bbox_xyxy": [10, 10, 60, 90],   "confidence": 0.99},
    {"class_name": "chip_stack",    "bbox_xyxy": [70, 10, 120, 60],  "confidence": 0.94},
    {"class_name": "dealer_button", "bbox_xyxy": [130, 10, 180, 60], "confidence": 0.97},
]
result = enricher.enrich(image, detections)
print(result)
```

> **Legacy format:** the enricher also accepts `"class"` instead of `"class_name"` and Roboflow-style `x`/`y`/`width`/`height` center-based coordinates. Prefer the canonical `class_name`/`bbox_xyxy` format for new code.

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
        {"class_name": "card",          "bbox_xyxy": [10, 10, 60, 90],   "confidence": 0.99},
        {"class_name": "chip_stack",    "bbox_xyxy": [70, 10, 120, 60],  "confidence": 0.94},
        {"class_name": "dealer_button", "bbox_xyxy": [130, 10, 180, 60], "confidence": 0.97}
    ]
}
```

> **Legacy format:** `"class"` is accepted in place of `"class_name"`, and Roboflow center-based `x`/`y`/`width`/`height` coordinates are accepted in place of `"bbox_xyxy"`.

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
| Per-crop latency | 2–5 s | ~1–1.2 s (Windows subprocess spawn overhead) |
| Setup | `pip install easyocr` | `pip install pytesseract` + Tesseract binary |
| Config needed | Minimal | PSM mode + character whitelist |

For our use case (reading `"470"`, `"1/2"` etc.) pytesseract with `--psm 7 -c tessedit_char_whitelist=0123456789/` achieves comparable accuracy to EasyOCR with significantly less latency per crop.

**Note (Windows):** On Windows, pytesseract spawns a new `tesseract.exe` process per call. Each subprocess startup costs ~1–1.2 s, making the naïve sequential approach slow (8 OCR fields × 1.2 s ≈ 9.6 s). The current default is `ocr_max_passes=1`, so only a single `psm 7` pass is used unless configured otherwise. This avoids the worst-case 4-pass path and keeps per-field OCR near 1 s.

**Fallback:** Legacy multi-pass OCR remains available in the codebase. Set `ocr_max_passes` to 0 to enable the full profile pass sequence (`psm 7` / `psm 6` with strong preprocessing fallback).

**Trade-off:** pytesseract requires the Tesseract binary to be installed on the host machine (not just a pip package), and needs explicit configuration (PSM mode, character whitelist) to work reliably on game UI crops. EasyOCR requires no configuration but is impractical for real-time use on CPU.

## Turn-Halo Detection

The enricher determines which player has the active turn by measuring a **white/silver glow ring** that this poker client renders around the avatar of the acting player.

### How it works

For every `player_me` and `player_other` detection, the enricher crops the bounding box from the screenshot and runs `_halo_score()`, producing a float in `[0.0, 1.0]` stored as `turn_halo_score` on the object.

The scorer uses a **horizontal band comparison**:

| Band | Rows (fraction of bbox height) | What it captures |
|---|---|---|
| **Top band** | `h×0.12` – `h×0.28` | The top arc of the halo, which crests above the card backs in the player bbox |
| **Card band** | `h×0.30` – `h×0.65` | The card-back surface — used as the brightness reference baseline |

A pixel is counted as bright when its HSV **V channel > 200** (brightness only; no saturation requirement, because the halo is white/silver — achromatic). The score is:

```
score = max(0, bright_ratio_top − bright_ratio_card)
```

where `bright_ratio = count(V > 200) / pixels_in_band`.

**Why horizontal bands instead of a radial ring?**
The player bbox includes the player name and stack at the bottom, so the card backs fill the full width of the bbox. A circular ring arc centred on the avatar descends into the card-back surface on both sides, contaminating the measurement. The top band exclusively captures the region above the cards where only halo glow or dark background are present — this gives a clean signal with no card-back noise.

### Winner selection

After scoring all player candidates, the enricher picks the active player:

```
best_score >= turn_halo_threshold        (default 0.10)
AND
(best_score − second_score) >= turn_halo_ambiguity_delta  (default 0.03)
```

If both conditions pass, the highest-scoring player receives `turn_active: True`. Otherwise all players get `turn_active: False` (ambiguous / no halo detected).

### Enricher output fields

| Field | Type | Meaning |
|---|---|---|
| `turn_halo_score` | float `[0.0, 1.0]` | Raw halo brightness score for this player |
| `turn_active` | boolean | `true` on the one player judged to hold the active turn |

### Configuration

All band fractions and thresholds are configurable via the config dict passed to `DetectionEnricher`:

| Key | Default | Meaning |
|---|---|---|
| `turn_halo_threshold` | `0.10` | Minimum score for the best candidate to be considered active |
| `turn_halo_ambiguity_delta` | `0.03` | Minimum gap between best and second-best scores |
| `halo_top_band_lo` | `0.12` | Top band start (fraction of bbox height) |
| `halo_top_band_hi` | `0.28` | Top band end |
| `halo_card_band_lo` | `0.30` | Card band start |
| `halo_card_band_hi` | `0.65` | Card band end |
| `halo_brightness_threshold` | `200` | V-channel threshold for counting a pixel as bright |

---

## Notes
- OCR uses [pytesseract](https://github.com/madmaze/pytesseract) with greyscale + contrast pre-processing. Requires Tesseract binary installed on host.
- The spatial reasoning module is implemented for `dealer_button` and `player_me`.
- Card classification calls the card classifier service (`POST /classify`) at `classifier_url` (default `http://127.0.0.1:5001`). On any connection or HTTP error the label falls back to `""` and the default confidence is used. Use `127.0.0.1` rather than `localhost` on Windows — `localhost` resolves to IPv6 (`::1`) first, which silently times out if the service only binds IPv4.
- `snips/` is gitignored — crop output from `audit_ocr.py` is local only.

### "All In" badge detection

When a player is all-in, the poker client renders an orange "All In" badge in place of a numeric chip stack value. The `numeric` OCR profile (`tessedit_char_whitelist=0123456789,./`) silently discards all letters, returning an empty string for this badge.

**Fallback:** After the numeric pass returns empty for a `chip_stack` crop, the enricher retries with the `player_name` OCR profile and matches the result against `_ALL_IN_RE` (`^all[\W_]*in$`, case-insensitive). If matched, `ocr_text` is normalised to `"All In"` and `ocr_conf` is set to `max(fallback_conf, 0.65)` — the floor guarantees the result passes the parser's 0.55 usable-confidence threshold even when tesseract reports low confidence for the stylised badge font.

This adds one extra subprocess spawn (~1–1.2 s) to any `chip_stack` crop whose numeric OCR returns empty.
