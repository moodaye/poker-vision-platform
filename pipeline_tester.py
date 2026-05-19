"""Run the live-play pipeline: single test or batch test multiple screenshots.

Requires all services to be running — use manage_services.py to start them:
    uv run python manage_services.py start

Services started:
    poker-vision-card-classifier/api.py     (port 5001)
    poker-vision-detection-enricher/api.py   (port 5004)
    poker-vision-hand-state-parser/api.py    (port 5003)
    poker-vision-decision-engine/api.py      (port 5002)
    orchestrator.py                          (port 5100)

Usage:
    Single screenshot (calls orchestrator service):
        uv run python pipeline_tester.py [path/to/screenshot.png]

    Single screenshot with detailed output (calls each service directly):
        uv run python pipeline_tester.py [path/to/screenshot.png] --verbose

    Save card crops from the enricher for bbox inspection:
        uv run python pipeline_tester.py [path/to/screenshot.png] --verbose --save-snips
        (crops saved to poker-vision-detection-enricher/snips/)

    Batch test all 13 preflop screenshots (bypass orchestrator, get summary table):
        uv run python pipeline_tester.py --batch

If no path is given for single mode, uses the default screenshot below.
Prints the decision JSON and speaks the action aloud using Windows TTS.

--verbose mode bypasses the orchestrator and calls each service directly,
printing intermediate outputs at every pipeline stage:
  1. Raw detections from Roboflow (classes only, no bbox coordinates)
  2. Enriched objects — spatial reasoning, OCR values, card classifications
  3. Hand state JSON sent to the decision engine
  4. Decision

--batch mode tests ./test-screenshots/screenshot_preflop_1.png through _13.png,
extracts Stage 2/3/4 data, and prints a summary table showing card detections,
hero card visibility, stacks, positions, and decisions.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
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
    "ROBOFLOW_API_URL", "https://serverless.roboflow.com/pokertabledetection/7"
)
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")

DEFAULT_SCREENSHOT = Path(
    "./poker-vision-screenshot-archive/capture_20260219_175149_088638.png"
)


def speak(action: str, amount: object) -> None:
    """Speak the decision using Windows built-in SAPI — no extra packages needed."""
    text = f"{action} {int(amount)}" if amount is not None else action
    try:
        no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                (
                    f"Add-Type -AssemblyName System.Speech; "
                    f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"
                ),
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
    image_bytes: bytes,
    detections: list[dict[str, Any]],
    save_snips: bool = False,
) -> dict[str, Any]:
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    body: dict[str, Any] = {"image_base64": image_base64, "detections": detections}
    if save_snips:
        body["config"] = {"save_snips": True, "snip_dir": "snips/"}
    response = requests.post(ENRICHER_URL, json=body, timeout=60)
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

    def bbox_centre(obj):
        bbox = obj.get("bbox_xyxy") or obj.get("bbox")
        if not bbox or len(bbox) < 4:
            return None
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    for obj in objects:
        cls = obj.get("class_name", "")
        conf = obj.get("confidence", 0.0)

        if cls == "dealer_button":
            dealer = (obj.get("spatial_info") or {}).get("dealer_player", "?")
            center = bbox_centre(obj)
            center_str = (
                f"center=({center[0]:.1f},{center[1]:.1f})" if center else "center=?"
            )
            print(
                f'  dealer_button   → dealer: "{dealer}"  {center_str}  (conf {conf:.2f})'
            )

        elif cls == "player_me":
            position = (obj.get("spatial_info") or {}).get("position", "?")
            center = bbox_centre(obj)
            center_str = (
                f"center=({center[0]:.1f},{center[1]:.1f})" if center else "center=?"
            )
            print(
                f"  player_me       → position: {position}  {center_str}  (conf {conf:.2f})"
            )

        elif cls == "player_name":
            name = obj.get("ocr_text", "?")
            stack = stack_by_owner.get((name or "").lower(), "?")
            center = bbox_centre(obj)
            center_str = (
                f"center=({center[0]:.1f},{center[1]:.1f})" if center else "center=?"
            )
            print(
                f'  player_name     → "{name}"  stack: {stack}  {center_str}  (conf {conf:.2f})'
            )

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
            center = bbox_centre(obj)
            center_str = (
                f"center=({center[0]:.1f},{center[1]:.1f})" if center else "center=?"
            )
            print(
                f'  chip_stack      → {ocr}  owner: "{owner}"  {center_str}  (conf {conf:.2f})'
            )

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
            center = bbox_centre(obj)
            center_str = (
                f"center=({center[0]:.1f},{center[1]:.1f})" if center else "center=?"
            )
            print(f"  {cls:<15} → {ocr}  {center_str}  (conf {conf:.2f})")

        else:
            center = bbox_centre(obj)
            center_str = (
                f"center=({center[0]:.1f},{center[1]:.1f})" if center else "center=?"
            )
            print(f"  {cls:<15}  {center_str}  (conf {conf:.2f})")


def _run_verbose(
    screenshot_path: Path, image_bytes: bytes, save_snips: bool = False
) -> dict[str, Any]:
    """Run the pipeline stage-by-stage with diagnostic output at each step."""
    t_pipeline = time.perf_counter()

    print("\n--- Stage 1: Object Detector ---")
    t0 = time.perf_counter()
    detections = _call_object_detector(image_bytes)
    t1_elapsed = time.perf_counter() - t0
    print(f"  {len(detections)} detections:  [{t1_elapsed:.2f}s]")
    for d in detections:
        cls = d.get("class") or d.get("class_name", "?")
        c = d.get("confidence", 0.0)
        print(f"    {cls}  (conf {c:.2f})")

    print("\n--- Stage 2: Detection Enricher ---")
    t0 = time.perf_counter()
    enriched = _call_enricher(image_bytes, detections, save_snips=save_snips)
    t2_elapsed = time.perf_counter() - t0
    print(f"  [{t2_elapsed:.2f}s]")
    _print_enriched_summary(enriched.get("objects", []))

    print("\n--- Stage 3: Hand State ---")
    t0 = time.perf_counter()
    hand_state = _call_hand_state_parser(enriched)
    t3_elapsed = time.perf_counter() - t0
    print(f"  [{t3_elapsed:.2f}s]")
    print(json.dumps(hand_state, indent=2))

    print("\n--- Stage 4: Decision ---")
    t0 = time.perf_counter()
    decision = _call_decision_engine(hand_state)
    t4_elapsed = time.perf_counter() - t0
    print(f"  [{t4_elapsed:.2f}s]")
    print(json.dumps(decision, indent=2))

    t_total = time.perf_counter() - t_pipeline
    print("\n--- Pipeline Timing Summary ---")
    print(f"  Stage 1  object-detector   {t1_elapsed:6.2f}s")
    print(f"  Stage 2  detection-enricher {t2_elapsed:6.2f}s")
    print(f"  Stage 3  hand-state-parser  {t3_elapsed:6.2f}s")
    print(f"  Stage 4  decision-engine    {t4_elapsed:6.2f}s")
    print(f"  Total (end-to-end)          {t_total:6.2f}s  (target <5.00s)")

    return decision


# ---------------------------------------------------------------------------
# Batch testing mode helpers
# ---------------------------------------------------------------------------


def _extract_batch_results(verbose_output: str) -> dict[str, object]:
    """Extract Stage 2, 3, and 4 data from verbose pipeline output."""
    import re

    # Extract Stage 2 holecard lines (detection + classification)
    stage2_cards: list[dict[str, object]] = []
    holecard_matches = re.finditer(
        r"^\s*holecard\s+.*?\(det\s+([\d.]+),\s+cls\s+([\d.]+)\)",
        verbose_output,
        flags=re.MULTILINE,
    )
    for m in holecard_matches:
        line = m.group(0)
        label_match = re.search(r"holecard\s+.*?→\s*([^\(]*)\(det", line)
        label = ""
        if label_match is not None:
            label = label_match.group(1).strip()
        stage2_cards.append(
            {
                "label": label,
                "det": float(m.group(1)),
                "cls": float(m.group(2)),
            }
        )

    # Extract Stage 3 JSON (hand state)
    stage3 = {}
    stage3_match = re.search(
        r"--- Stage 3: Hand State ---\s*(\{.*?\})\s*--- Stage 4:",
        verbose_output,
        flags=re.DOTALL,
    )
    if stage3_match is not None:
        try:
            stage3 = json.loads(stage3_match.group(1))
        except json.JSONDecodeError:
            stage3 = {}

    # Extract Stage 4 JSON (decision)
    stage4 = {}
    stage4_match = re.search(
        r"--- Stage 4: Decision ---\s*(\{.*?\})\s*$",
        verbose_output,
        flags=re.DOTALL,
    )
    if stage4_match is not None:
        try:
            stage4 = json.loads(stage4_match.group(1))
        except json.JSONDecodeError:
            stage4 = {}

    return {
        "stage2_cards": stage2_cards,
        "stage3": stage3,
        "stage4": stage4,
    }


def _run_batch_tests() -> None:
    """Test all 13 preflop screenshots and print summary table."""
    test_dir = Path("./test-screenshots")
    results = []

    print("Processing 13 screenshots...\n", file=sys.stderr)

    for i in range(1, 14):
        fname = f"screenshot_preflop_{i}.png"
        fpath = test_dir / fname

        if not fpath.exists():
            print(f"[{i:2d}/13] SKIP  {fname} (not found)", file=sys.stderr)
            continue

        print(f"[{i:2d}/13] TEST  {fname}...", file=sys.stderr, end=" ")

        try:
            if not fpath.exists():
                print("ERROR: not found", file=sys.stderr)
                continue

            image_bytes = fpath.read_bytes()
            _run_verbose(fpath, image_bytes)
            print("OK", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print("ERROR: timeout", file=sys.stderr)
            continue
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            continue

    # Re-run with verbose mode to collect data for summary table
    print("\nCollecting detailed results...\n", file=sys.stderr)
    results = []

    for i in range(1, 14):
        fname = f"screenshot_preflop_{i}.png"
        fpath = test_dir / fname

        if not fpath.exists():
            continue

        print(f"[{i:2d}/13] COLLECT {fname}...", file=sys.stderr, end=" ")

        try:
            image_bytes = fpath.read_bytes()

            # Capture verbose output
            import io

            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            try:
                _run_verbose(fpath, image_bytes, save_snips=False)
                verbose_output = sys.stdout.getvalue() + sys.stderr.getvalue()
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            # Extract data from verbose output
            extracted = _extract_batch_results(verbose_output)
            stage2_cards = extracted.get("stage2_cards", [])
            stage3 = extracted.get("stage3", {})
            stage4 = extracted.get("stage4", {})
            hero_cards = stage3.get("hero_cards", [])
            hero_cards_visibility = str(stage3.get("hero_cards_visibility", "unknown"))
            hero_stack = stage3.get("hero_stack", "?")

            s2_c1_label = ""
            s2_c1_det = ""
            s2_c1_cls = ""
            s2_c2_label = ""
            s2_c2_det = ""
            s2_c2_cls = ""
            if len(stage2_cards) > 0:
                s2_c1_label = str(stage2_cards[0].get("label", ""))
                s2_c1_det = f"{stage2_cards[0].get('det', ''):.2f}"
                s2_c1_cls = f"{stage2_cards[0].get('cls', ''):.2f}"
            if len(stage2_cards) > 1:
                s2_c2_label = str(stage2_cards[1].get("label", ""))
                s2_c2_det = f"{stage2_cards[1].get('det', ''):.2f}"
                s2_c2_cls = f"{stage2_cards[1].get('cls', ''):.2f}"

            results.append(
                {
                    "file": fname,
                    "s2_card1": s2_c1_label,
                    "s2_det1": s2_c1_det,
                    "s2_cls1": s2_c1_cls,
                    "s2_card2": s2_c2_label,
                    "s2_det2": s2_c2_det,
                    "s2_cls2": s2_c2_cls,
                    "card1": hero_cards[0] if len(hero_cards) > 0 else "?",
                    "card2": hero_cards[1] if len(hero_cards) > 1 else "?",
                    "hero_stack": hero_stack,
                    "hero_cards_visibility": hero_cards_visibility,
                    "position": stage3.get("position", "?"),
                    "decision": stage4.get("action", "?"),
                    "decision_amount": stage4.get("amount", "?"),
                }
            )
            print("OK", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            continue

    # Print table
    print("\n" + "=" * 170)
    print(
        f"{'File':<26} {'S2 Card1':<10} {'Det1':<6} {'Cls1':<6} "
        f"{'S2 Card2':<10} {'Det2':<6} {'Cls2':<6} "
        f"{'Stage3 Cards':<14} {'HeroVis':<12} {'Stack':<7} {'Pos':<5} {'Decision':<8} {'Amt':<8}"
    )
    print("=" * 170)

    for row in results:
        stage3_cards = f"{row['card1']},{row['card2']}"
        print(
            f"{row['file']:<26} "
            f"{row['s2_card1']:<10} {row['s2_det1']:<6} {row['s2_cls1']:<6} "
            f"{row['s2_card2']:<10} {row['s2_det2']:<6} {row['s2_cls2']:<6} "
            f"{stage3_cards:<14} {str(row['hero_cards_visibility']):<12} {str(row['hero_stack']):<7} "
            f"{row['position']:<5} {str(row['decision']):<8} {str(row['decision_amount']):<8}"
        )

    print("=" * 170)
    print(f"\nTotal: {len(results)} screenshots processed")

    exposed_count = sum(
        1 for r in results if r.get("hero_cards_visibility") == "exposed"
    )
    not_exposed_count = sum(
        1 for r in results if r.get("hero_cards_visibility") == "not_exposed"
    )
    unknown_count = len(results) - exposed_count - not_exposed_count
    print(f"Hero cards exposed: {exposed_count}")
    print(f"Hero cards not exposed: {not_exposed_count}")
    print(f"Hero cards visibility unknown: {unknown_count}")


def main() -> None:
    args = sys.argv[1:]

    # Check for batch mode first
    if "--batch" in args:
        _run_batch_tests()
        return

    verbose = "--verbose" in args
    save_snips = "--save-snips" in args
    paths = [a for a in args if not a.startswith("--")]

    screenshot_path = Path(paths[0]) if paths else DEFAULT_SCREENSHOT

    if not screenshot_path.exists():
        print(f"Error: screenshot not found: {screenshot_path}")
        sys.exit(1)

    print(f"\nScreenshot: {screenshot_path}")
    image_bytes = screenshot_path.read_bytes()
    print(f"Loaded {len(image_bytes)} bytes")

    if verbose:
        decision = _run_verbose(screenshot_path, image_bytes, save_snips=save_snips)
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
