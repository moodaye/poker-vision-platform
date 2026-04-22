"""Run the live-play pipeline against a saved screenshot.

Requires all services to be running:
    uv run python poker-vision-detection-enricher/api.py   (port 5004)
    uv run python poker-vision-hand-state-parser/api.py    (port 5003)
    uv run python poker-vision-decision-engine/api.py      (port 5002)
    uv run python orchestrator.py                          (port 5100)

Usage:
    uv run python orchestrate_pipeline.py [path/to/screenshot.png]

If no path is given, uses the default screenshot below.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ORCHESTRATOR_URL = "http://localhost:5100/decide"
DEFAULT_SCREENSHOT = Path(
    "./poker-vision-screenshot-archive/capture_20260219_175149_088638.png"
)


def main() -> None:
    screenshot_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SCREENSHOT

    if not screenshot_path.exists():
        print(f"Error: screenshot not found: {screenshot_path}")
        sys.exit(1)

    print(f"\nScreenshot: {screenshot_path}")
    image_bytes = screenshot_path.read_bytes()
    print(f"Loaded {len(image_bytes)} bytes")

    print(f"\nPOST {ORCHESTRATOR_URL} ...")
    response = requests.post(
        ORCHESTRATOR_URL,
        files={"image": (screenshot_path.name, image_bytes, "image/png")},
        timeout=60,
    )

    print(f"Status: {response.status_code}")
    print("\n=== Decision ===")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
