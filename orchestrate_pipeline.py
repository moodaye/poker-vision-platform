"""Run the live-play pipeline against a saved screenshot.

Requires all services to be running — use manage_services.py to start them:
    uv run python manage_services.py start

Services started:
    poker-vision-detection-enricher/api.py   (port 5004)
    poker-vision-hand-state-parser/api.py    (port 5003)
    poker-vision-decision-engine/api.py      (port 5002)
    orchestrator.py                          (port 5100)

Usage:
    uv run python orchestrate_pipeline.py [path/to/screenshot.png] [--verbose]

If no path is given, uses the default screenshot below.
Prints the decision JSON and speaks the action aloud using Windows TTS.

--verbose mode bypasses the orchestrator and calls each service directly,
printing intermediate outputs at every pipeline stage:
  1. Raw detections from Roboflow (classes only, no bbox coordinates)
  2. Enriched objects — spatial reasoning, OCR values, card classifications
  3. Hand state JSON sent to the decision engine
  4. Decision
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / "poker-vision-object-detector" / ".env")

ORCHESTRATOR_URL = "http://localhost:5100/decide"
ENRICHER_URL = "http://127.0.0.1:5004/enrich"
HAND_STATE_PARSER_URL = "http://127.0.0.1:5003/parse"
DECISION_ENGINE_URL = "http://127.0.0.1:5002/decide"
ROBOFLOW_API_URL = os.environ.get(
    "ROBOFLOW_API_URL", "https://detect.roboflow.com/pokertabledetection/6"
)
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")

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


# ---------------------------------------------------------------------------
# Verbose mode helpers
# ---------------------------------------------------------------------------


def _call_object_detector(image_bytes: bytes) -> list[dict[str, Any]]:
    if not ROBOFLOW_API_KEY:
        raise RuntimeError("ROBOFLOW_API_KEY is not set")
    response = requests.post(
        ROBOFLOW_API_URL,
        params={"api_key": ROBOFLOW_API_KEY},
        files={"file": image_bytes},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("Object detector response did not contain predictions list")
    return predictions


def _call_enricher(
    image_bytes: bytes, detections: list[dict[str, Any]]
) -> dict[str, Any]:
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    response = requests.post(
        ENRICHER_URL,
        json={"image_base64": image_base64, "detections": detections},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _call_hand_state_parser(enriched_payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(HAND_STATE_PARSER_URL, json=enriched_payload, timeout=30)
    response.raise_for_status()
    return response.json()


def _call_decision_engine(hand_state: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(DECISION_ENGINE_URL, json=hand_state, timeout=30)
    response.raise_for_status()
    return response.json()


def _print_enriched_summary(objects: list[dict[str, Any]]) -> None:
    """Print a human-readable summary of enriched objects — no bbox coordinates."""
    # Collect chip stack owners for cross-referencing
    stack_by_owner: dict[str, str] = {}
    for obj in objects:
        cls = obj.get("class_name", "")
        if cls == "chip_stack":
            owner = (obj.get("spatial_info") or {}).get("owner_player", "")
            ocr = obj.get("ocr_text", "?")
            if owner:
                stack_by_owner[owner.lower()] = ocr

    for obj in objects:
        cls = obj.get("class_name", "")
        conf = obj.get("confidence", 0.0)

        if cls == "dealer_button":
            dealer = (obj.get("spatial_info") or {}).get("dealer_player", "?")
            print(f'  dealer_button   → dealer: "{dealer}"  (conf {conf:.2f})')

        elif cls == "player_me":
            position = (obj.get("spatial_info") or {}).get("position", "?")
            print(f"  player_me       → position: {position}  (conf {conf:.2f})")

        elif cls == "player_name":
            name = obj.get("ocr_text", "?")
            stack = stack_by_owner.get((name or "").lower(), "?")
            print(f'  player_name     → "{name}"  stack: {stack}  (conf {conf:.2f})')

        elif cls in ("holecard", "card"):
            label = obj.get("classification", "?")
            cls_conf = obj.get("classification_conf", 0.0)
            print(f"  {cls:<15} → {label}  (det {conf:.2f}, cls {cls_conf:.2f})")

        elif cls in ("flop_card", "turn_card", "river_card"):
            label = obj.get("classification", "?")
            cls_conf = obj.get("classification_conf", 0.0)
            print(f"  {cls:<15} → {label}  (det {conf:.2f}, cls {cls_conf:.2f})")

        elif cls == "chip_stack":
            owner = (obj.get("spatial_info") or {}).get("owner_player", "?")
            ocr = obj.get("ocr_text", "?")
            print(f'  chip_stack      → {ocr}  owner: "{owner}"  (conf {conf:.2f})')

        elif cls in (
            "blinds",
            "pot",
            "total_pot",
            "bet",
            "pot_bet",
            "max_bet",
            "min_bet",
        ):
            ocr = obj.get("ocr_text", "?")
            print(f"  {cls:<15} → {ocr}  (conf {conf:.2f})")

        else:
            print(f"  {cls:<15}  (conf {conf:.2f})")


def _run_verbose(screenshot_path: Path, image_bytes: bytes) -> dict[str, Any]:
    """Run the pipeline stage-by-stage with diagnostic output at each step."""
    print("\n--- Stage 1: Object Detector ---")
    detections = _call_object_detector(image_bytes)
    print(f"  {len(detections)} detections:")
    for d in detections:
        cls = d.get("class") or d.get("class_name", "?")
        c = d.get("confidence", 0.0)
        print(f"    {cls}  (conf {c:.2f})")

    print("\n--- Stage 2: Detection Enricher ---")
    enriched = _call_enricher(image_bytes, detections)
    _print_enriched_summary(enriched.get("objects", []))

    print("\n--- Stage 3: Hand State ---")
    hand_state = _call_hand_state_parser(enriched)
    print(json.dumps(hand_state, indent=2))

    print("\n--- Stage 4: Decision ---")
    decision = _call_decision_engine(hand_state)
    print(json.dumps(decision, indent=2))

    return decision


def main() -> None:
    args = sys.argv[1:]
    verbose = "--verbose" in args
    paths = [a for a in args if not a.startswith("--")]

    screenshot_path = Path(paths[0]) if paths else DEFAULT_SCREENSHOT

    if not screenshot_path.exists():
        print(f"Error: screenshot not found: {screenshot_path}")
        sys.exit(1)

    print(f"\nScreenshot: {screenshot_path}")
    image_bytes = screenshot_path.read_bytes()
    print(f"Loaded {len(image_bytes)} bytes")

    if verbose:
        decision = _run_verbose(screenshot_path, image_bytes)
    else:
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
