"""Run the live-play pipeline against a saved screenshot.

Requires all services to be running — use manage_services.py to start them:
    uv run python manage_services.py start

Services started:
    poker-vision-detection-enricher/api.py   (port 5004)
    poker-vision-hand-state-parser/api.py    (port 5003)
    poker-vision-decision-engine/api.py      (port 5002)
    orchestrator.py                          (port 5100)

Usage:
    uv run python orchestrate_pipeline.py [path/to/screenshot.png]

If no path is given, uses the default screenshot below.
Prints the decision JSON and speaks the action aloud using Windows TTS.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import requests

ORCHESTRATOR_URL = "http://localhost:5100/decide"
DEFAULT_SCREENSHOT = Path(
    "./poker-vision-screenshot-archive/capture_20260219_175149_088638.png"
)


def speak(action: str, amount: object) -> None:
    """Speak the decision using Windows built-in SAPI — no extra packages needed."""
    if action in ("watch", "wait"):
        return
    text = f"{action} {int(amount)}" if amount is not None else action
    try:
        no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=no_window,
        )
        proc.wait()  # block until speech finishes before the script exits
    except Exception:
        pass


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
    decision = response.json()
    print(json.dumps(decision, indent=2))

    speak(decision.get("action", ""), decision.get("amount"))


if __name__ == "__main__":
    main()
