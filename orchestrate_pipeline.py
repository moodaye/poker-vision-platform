"""Run the live-play pipeline against a saved screenshot.

This script mirrors the orchestrator flow:
1. Load a screenshot from disk
2. Call the object detector
3. Call the detection enricher
4. Build a HandState payload
5. Call the decision engine
"""

from pathlib import Path

from hand_state_parser import build_hand_state
from orchestrator import (
    call_decision_engine,
    call_detection_enricher,
    call_object_detector,
)

SCREENSHOT_PATH = Path(
    "./poker-vision-screenshot-archive/capture_20260219_175149_088638.png"
)


def main() -> None:
    print("\n--- 1. Load screenshot ---")
    print(f"Screenshot path: {SCREENSHOT_PATH}")
    image_bytes = SCREENSHOT_PATH.read_bytes()
    print(f"Loaded {len(image_bytes)} bytes\n")

    print("--- 2. Call object detector ---")
    detections = call_object_detector(image_bytes)
    print(f"Detected {len(detections)} objects\n")

    print("--- 3. Call detection enricher ---")
    enriched_payload = call_detection_enricher(image_bytes, detections)
    print(f"Enriched objects: {len(enriched_payload['objects'])}\n")

    print("--- 4. Build hand state ---")
    hand_state = build_hand_state(enriched_payload)
    print(f"HandState: {hand_state}\n")

    print("--- 5. Call decision engine ---")
    decision = call_decision_engine(hand_state)
    print(f"Decision engine output: {decision}\n")

    print("=== Final decision ===")
    print(decision)


if __name__ == "__main__":
    main()
