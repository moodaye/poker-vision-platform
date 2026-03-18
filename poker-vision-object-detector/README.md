# Poker Vision — Phase 1: Card Detection via Roboflow

Runs Roboflow hosted inference on local screenshot images, saving raw and normalized detections to disk.

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install requests pyyaml pillow python-dotenv pytest responses
```

---

## Configuration

### 3. Create your `.env` file

```bash
copy .env.example .env        # Windows
cp .env.example .env          # macOS / Linux
```

Then open `.env` and replace `your_api_key_here` with your actual Roboflow API key:

```
ROBOFLOW_API_KEY=your_actual_key
```

> **Never commit `.env` to version control.**

### 4. Edit `detect_config.yaml`

Open `detect_config.yaml` and set at minimum:

```yaml
input_dir: "screenshots"        # folder containing your .png/.jpg images
output_dir: "output"            # where results will be written
roboflow:
  project: "your-project-name" # your Roboflow project slug
  version: 1                   # model version number
```

---

## Running the CLI

```bash
python -m poker_vision.detect --config detect_config.yaml
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | *(required)* | Path to the YAML config file |
| `--env-file PATH` | `.env` | Path to the .env file |
| `--log-level LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### Output files

After a successful run, `output_dir` will contain:

```
output/
├── run_config.resolved.json          # Resolved config (secrets excluded)
├── run_summary.json                  # Aggregate stats + failures
├── predictions_raw/
│   └── <image_stem>.json            # Exact Roboflow API response
└── detections_normalized/
    └── <image_stem>.detections.json # Normalized detections schema
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success (partial per-image failures are tolerated) |
| `1` | All images failed |
| `2` | Invalid config or missing API key |

---

## Running tests

```bash
pytest -q
```

Tests are fully offline — HTTP is mocked using the `responses` library. No real API key is needed.

To run a specific test file:

```bash
pytest tests/test_bbox.py -v
pytest tests/test_normalize.py -v
pytest tests/test_runner_outputs.py -v
pytest tests/test_integration_mocked.py -v
```

---

## Project structure

```
poker_vision/
└── detect/
    ├── __init__.py
    ├── __main__.py       # python -m poker_vision.detect entry point
    ├── cli.py            # Argument parsing, dotenv loading, entrypoint
    ├── config.py         # Dataclasses, YAML loader, defaults, resolved config
    ├── client.py         # Roboflow HTTP upload client with retries
    ├── normalize.py      # Bbox conversion + detection normalization
    └── runner.py         # Directory scan, per-image processing, output writing

tests/
├── test_bbox.py               # Unit tests: bbox conversion
├── test_normalize.py          # Unit tests: normalize_response
├── test_runner_outputs.py     # Runner output structure tests (monkeypatched)
└── test_integration_mocked.py # Full pipeline integration tests (mocked HTTP)

detect_config.yaml   # Sample configuration
.env.example         # Example environment file (no real key)
```

---

## Normalized detection schema

Each `detections_normalized/<stem>.detections.json` file follows this schema:

```json
{
  "source_image": "/path/to/image.png",
  "image_width": 1280,
  "image_height": 720,
  "model": {
    "provider": "roboflow",
    "api_base": "https://detect.roboflow.com",
    "project": "your-project",
    "version": "1",
    "requested_confidence_threshold": 0.5,
    "requested_overlap_threshold": 0.5
  },
  "detections": [
    {
      "class_name": "Ace_Spades",
      "class_id": 0,
      "confidence": 0.92,
      "bbox_xywh_center": [100, 200, 40, 60],
      "bbox_xyxy": [80, 170, 120, 230]
    }
  ]
}
```
