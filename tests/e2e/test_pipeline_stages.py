"""E2E stage-contract tests: screenshot -> detector -> enricher -> parser -> decision.

Each scenario starts from a screenshot and validates the expected JSON at
each deterministic stage (enricher, hand-state parser, decision engine).
The detector stage is non-deterministic (Roboflow bbox/confidence varies per
call) so it is logged but not asserted — only its downstream effects are checked.

Failures are stage-specific: each stage assertion names the exact stage that
regressed, so you can pinpoint a break without reading the full trace.

Prerequisites:
    uv run python manage_services.py start

Run:
    Full E2E:   uv run pytest tests/e2e -v
    Smoke only: uv run pytest tests/e2e -m smoke -v
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import (
    call_decision_engine,
    call_detector,
    call_enricher,
    call_hand_state_parser,
    load_expected_fixture,
    normalize_decision,
    normalize_enricher,
    normalize_hand_state,
    screenshot_path,
)


def _assert_stage(
    stage: str, scenario_id: str, actual: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Assert actual == expected with a stage-specific failure message."""
    assert actual == expected, (
        f"\n[STAGE: {stage}] Scenario {scenario_id!r} regressed.\n"
        f"Expected: {expected}\n"
        f"Actual:   {actual}"
    )


# ---------------------------------------------------------------------------
# Parametrization: one test per scenario, each running all stages
# ---------------------------------------------------------------------------


def _scenario_ids() -> list[dict[str, Any]]:
    from conftest import load_scenarios

    return load_scenarios()


@pytest.mark.parametrize(
    "scenario",
    _scenario_ids(),
    ids=lambda s: str(s["id"]),
    indirect=True,
)
def test_e2e_pipeline_stages(scenario: dict[str, Any]) -> None:
    """Run the full pipeline from screenshot and assert each stage's JSON."""
    scenario_id = scenario["id"]
    screenshot = screenshot_path(scenario)
    image_bytes = screenshot.read_bytes()

    # --- Stage 1: detector (non-deterministic — log only, no assertion) ---
    detections = call_detector(image_bytes)
    print(
        f"\n[{scenario_id}] detector: {len(detections)} predictions "
        f"(non-deterministic, not asserted)"
    )

    # --- Stage 2: enricher ---
    enriched = call_enricher(image_bytes, detections)
    expected_enricher = load_expected_fixture(scenario_id, "enricher")
    _assert_stage(
        "enricher",
        scenario_id,
        normalize_enricher(enriched),
        normalize_enricher(expected_enricher),
    )

    # --- Stage 3: hand-state parser ---
    hand_state = call_hand_state_parser(enriched)
    expected_hand_state = load_expected_fixture(scenario_id, "hand_state")
    _assert_stage(
        "hand_state",
        scenario_id,
        normalize_hand_state(hand_state),
        normalize_hand_state(expected_hand_state),
    )

    # --- Stage 4: decision engine ---
    decision = call_decision_engine(hand_state)
    expected_decision = load_expected_fixture(scenario_id, "decision")
    _assert_stage(
        "decision",
        scenario_id,
        normalize_decision(decision),
        normalize_decision(expected_decision),
    )
