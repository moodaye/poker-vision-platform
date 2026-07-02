"""Capture canonical E2E fixtures by running the live pipeline once per scenario.

For each enabled scenario in tests/fixtures/e2e/scenarios.json, this script:
  1. Reads the screenshot.
  2. Calls detector (Roboflow) -> enricher -> hand-state parser -> decision engine.
  3. Saves the enricher, hand-state, and decision JSON outputs as canonical
     expected fixtures under tests/fixtures/e2e/<scenario_id>/.

The detector output is NOT saved as a fixture — it is non-deterministic
(bbox coordinates and confidence values vary per call). Only the downstream
stages (which are deterministic given a detector output) are captured.

Prerequisites:
    - Services running:  uv run python manage_services.py start
    - ROBOFLOW_API_KEY set in poker-vision-object-detector/.env

Usage:
    uv run python tests/e2e/capture_fixtures.py            # capture all enabled
    uv run python tests/e2e/capture_fixtures.py --smoke-only  # capture smoke scenario only
    uv run python tests/e2e/capture_fixtures.py --scenario raise_bb_limped_a9s
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Reuse the conftest's callers and config so there is a single source of truth.
# conftest.py is in the same package; import it directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest as e2e_conftest  # noqa: E402


def _run_pipeline_for_scenario(
    scenario: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run the full live pipeline for one scenario.

    Returns (detector_output, enricher_output, hand_state_output, decision_output).
    """
    screenshot = e2e_conftest.screenshot_path(scenario)
    image_bytes = screenshot.read_bytes()
    scenario_id = scenario["id"]

    print(f"\n[{scenario_id}] Running pipeline for {screenshot.name} ...")

    # Stage 1: detector (non-deterministic — logged only)
    print(f"[{scenario_id}]   Stage 1: detector ...", end=" ", flush=True)
    detections = e2e_conftest.call_detector(image_bytes)
    print(f"{len(detections)} predictions")

    # Stage 2: enricher
    print(f"[{scenario_id}]   Stage 2: enricher ...", end=" ", flush=True)
    enriched = e2e_conftest.call_enricher(image_bytes, detections)
    print(f"{len(enriched.get('objects', []))} objects")

    # Stage 3: hand-state parser
    print(f"[{scenario_id}]   Stage 3: hand-state parser ...", end=" ", flush=True)
    hand_state = e2e_conftest.call_hand_state_parser(enriched)
    print("ok")

    # Stage 4: decision engine
    print(f"[{scenario_id}]   Stage 4: decision engine ...", end=" ", flush=True)
    decision = e2e_conftest.call_decision_engine(hand_state)
    print(f"action={decision.get('action')} amount={decision.get('amount')}")

    return detections, enriched, hand_state, decision


def _save_fixture(scenario_id: str, stage: str, data: Any) -> Path:
    """Save a canonical fixture JSON for a scenario + stage."""
    out_dir = e2e_conftest.FIXTURES_DIR / scenario_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"expected_{stage}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    return out_path


def capture(scenarios: list[dict[str, Any]]) -> int:
    """Capture fixtures for all given scenarios. Returns count captured."""
    captured = 0
    for scenario in scenarios:
        scenario_id = scenario["id"]
        try:
            _detections, enriched, hand_state, decision = _run_pipeline_for_scenario(
                scenario
            )
        except Exception as exc:
            print(f"[{scenario_id}]   FAILED: {exc}", file=sys.stderr)
            continue

        _save_fixture(scenario_id, "enricher", enriched)
        _save_fixture(scenario_id, "hand_state", hand_state)
        _save_fixture(scenario_id, "decision", decision)
        captured += 1
        print(
            f"[{scenario_id}]   Saved fixtures to {e2e_conftest.FIXTURES_DIR / scenario_id}"
        )

    return captured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Capture only the scenario marked smoke=true.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Capture a single scenario by id.",
    )
    args = parser.parse_args()

    # Load detector env so ROBOFLOW_API_KEY is available.
    e2e_conftest._load_detector_env()

    # Precondition checks
    down = [
        name
        for name, url in e2e_conftest.HEALTH_URLS.items()
        if not e2e_conftest._is_healthy(url)
    ]
    if down:
        print(
            f"ERROR: services not running: {', '.join(down)}\n"
            f"Start them with:  uv run python manage_services.py start",
            file=sys.stderr,
        )
        return 1
    if not e2e_conftest.ROBOFLOW_API_KEY:
        print(
            f"ERROR: ROBOFLOW_API_KEY not set in {e2e_conftest._DETECTOR_ENV}",
            file=sys.stderr,
        )
        return 1

    scenarios = e2e_conftest.load_scenarios()
    if args.scenario:
        scenarios = [s for s in scenarios if s["id"] == args.scenario]
        if not scenarios:
            print(
                f"ERROR: scenario {args.scenario!r} not found in manifest",
                file=sys.stderr,
            )
            return 1
    elif args.smoke_only:
        scenarios = [s for s in scenarios if s.get("smoke")]
        if not scenarios:
            print("ERROR: no smoke scenario found in manifest", file=sys.stderr)
            return 1

    print(f"Capturing fixtures for {len(scenarios)} scenario(s) ...")
    captured = capture(scenarios)
    print(f"\nDone. Captured {captured}/{len(scenarios)} scenario(s).")
    return 0 if captured == len(scenarios) else 1


if __name__ == "__main__":
    raise SystemExit(main())
