"""
End-to-end Poker Vision Orchestrator

This script runs the full pipeline:
1. Loads a screenshot from the archive
2. Calls Roboflow object detector
3. Snips cards from the screenshot
4. Classifies each card
5. Calls the decision engine

Assumes all services are running locally and Roboflow API key/config is set up.
"""

import base64
import os
import sys
from io import BytesIO
from pathlib import Path

import requests
from card_snipper import snip_flop_cards

# Load .env file from poker-vision-object-detector
from dotenv import load_dotenv
from PIL import Image

load_dotenv(dotenv_path=Path(__file__).parent / "poker-vision-object-detector" / ".env")


PROJECT_ROOT = Path(__file__).parent.resolve()
SNIPPER_PATH = PROJECT_ROOT / "poker-vision-card-snipper"
if str(SNIPPER_PATH) not in sys.path:
    sys.path.insert(0, str(SNIPPER_PATH))


# --- Config ---
# Screenshot to use
SCREENSHOT_PATH = Path(
    "./poker-vision-screenshot-archive/capture_20260219_175149_088638.png"
)
# Roboflow config
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_API_URL = "https://detect.roboflow.com/pokertabledetection/6"
# Card classifier API
CARD_CLASSIFIER_URL = "http://localhost:5001/classify"
# Decision engine API
DECISION_ENGINE_URL = "http://localhost:5002/decide"


print("\n--- 1. Load screenshot ---")
print(f"Screenshot path: {SCREENSHOT_PATH}")
img = Image.open(SCREENSHOT_PATH)
print(f"Image size: {img.size}\n")


print("--- 2. Call Roboflow object detector ---")
with open(SCREENSHOT_PATH, "rb") as f:
    img_bytes = f.read()
print(f"Roboflow API URL: {ROBOFLOW_API_URL}")
response = requests.post(
    ROBOFLOW_API_URL,
    params={"api_key": ROBOFLOW_API_KEY},
    files={"file": img_bytes},
)
print(f"Roboflow request status: {response.status_code}")
response.raise_for_status()
detections = response.json()["predictions"]  # List of dicts
print(f"Object detector output (detections):\n{detections}\n")

# --- Formatted detection summary ---
print("Detections summary:")
for det in detections:
    print(
        f"  class={det.get('class', ''):12} conf={det.get('confidence', 0):.3f} "
        f"x={det.get('x', '')} y={det.get('y', '')} w={det.get('width', '')} h={det.get('height', '')}"
    )
print()

# Convert Roboflow bbox format if needed (xywh to xyxy)
for det in detections:
    if "x" in det and "y" in det and "width" in det and "height" in det:
        x, y, w, h = det["x"], det["y"], det["width"], det["height"]
        x1 = round(x - w / 2)
        y1 = round(y - h / 2)
        x2 = round(x + w / 2)
        y2 = round(y + h / 2)
        det["bbox_xyxy"] = [x1, y1, x2, y2]
        det["class_name"] = det.get("class", "flop_card")


print("--- 3. Snip cards ---")
print(f"Snipper input: detections (count={len(detections)})")
card_snips = snip_flop_cards(img, detections, target_class="flop_card")
print(f"Snipper output: {len(card_snips)} card snips\n")


print("--- 4. Classify each card ---")
classified_cards = []
for idx, snip in enumerate(card_snips):
    buffered = BytesIO()
    snip.image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    print(f"Classifier input (card {idx}): bbox={snip.bbox_xyxy}")
    resp = requests.post(CARD_CLASSIFIER_URL, json={"image": img_b64})
    print(f"Classifier response status: {resp.status_code}")
    resp.raise_for_status()
    result = resp.json()
    print(f"Classifier output (card {idx}): {result}\n")
    classified_cards.append(
        {
            "label": result["label"],
            "confidence": result["confidence"],
            "bbox_xyxy": snip.bbox_xyxy,
        }
    )


print("--- 5. Call decision engine ---")
# Assemble minimal HandState (mocking missing info)
hand_state = {
    "hero_cards": [
        c["label"] for c in classified_cards[:2]
    ],  # Use first 2 as hero cards
    "position": "BTN",
    "big_blind": 100,
    "small_blind": 50,
    "hero_stack": 3000,
    "pot": 150,
    "amount_to_call": 0,
    "action_history": [],
    "is_hero_turn": True,
    "hero_folded": False,
}
print(f"Decision engine input (HandState):\n{hand_state}\n")
resp = requests.post(DECISION_ENGINE_URL, json=hand_state)
print(f"Decision engine response status: {resp.status_code}")
resp.raise_for_status()
decision = resp.json()
print(f"Decision engine output: {decision}\n")

print("=== Final decision ===")
print(decision)
