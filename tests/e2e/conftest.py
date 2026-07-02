"""E2E test configuration: service health preconditions + fixture loading.

E2E tests in this suite start from a screenshot and exercise the full live
pipeline via HTTP:
    detector (Roboflow) -> enricher -> hand-state parser -> decision engine

Services must be started out-of-band:
    uv run python manage_services.py start

The suite fails fast if any required service is unreachable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import requests

# ---------------------------------------------------------------------------
# Path setup — reuse root conftest's sys.path additions for any in-process
# fallback imports (not used in the live-HTTP path, but keeps imports valid).
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Service URLs (must match orchestrator.py / manage_services.py)
# ---------------------------------------------------------------------------
ENRICHER_URL = "http://127.0.0.1:5004/enrich"
HAND_STATE_PARSER_URL = "http://127.0.0.1:5003/parse"
DECISION_ENGINE_URL = "http://127.0.0.1:5002/decide"

HEALTH_URLS = {
    "detection-enricher": "http://127.0.0.1:5004/health",
    "hand-state-parser": "http://127.0.0.1:5003/health",
    "decision-engine": "http://127.0.0.1:5002/health",
}

# Roboflow detector config (read from the object-detector .env, same as
# orchestrator.py).
_DETECTOR_ENV = ROOT / "poker-vision-object-detector" / ".env"
ROBOFLOW_API_URL = os.environ.get(
    "ROBOFLOW_API_URL", "https://serverless.roboflow.com/pokertabledetection/7"
)
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")

# ---------------------------------------------------------------------------
# Fixture directories
# ---------------------------------------------------------------------------
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "e2e"
SCENARIOS_MANIFEST = FIXTURES_DIR / "scenarios.json"
SCREENSHOTS_ROOT = ROOT / "test-screenshots"

REQUEST_TIMEOUT_SECONDS = 60


# ---------------------------------------------------------------------------
# Session-scoped precondition: fail fast if services are down
# ---------------------------------------------------------------------------


def _is_healthy(url: str) -> bool:
    try:
        resp = requests.get(url, timeout=2)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _load_detector_env() -> None:
    """Load ROBOFLOW_API_KEY from the object-detector .env if not already set."""
    global ROBOFLOW_API_KEY
    if ROBOFLOW_API_KEY:
        return
    if not _DETECTOR_ENV.exists():
        return
    for line in _DETECTOR_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key == "ROBOFLOW_API_KEY" and value:
                os.environ[key] = value
                ROBOFLOW_API_KEY = value


@pytest.fixture(scope="session", autouse=True)
def require_live_services() -> None:
    """Fail fast with a clear message if any required service is unreachable."""
    _load_detector_env()

    down = [name for name, url in HEALTH_URLS.items() if not _is_healthy(url)]
    if down:
        pytest.exit(
            "E2E services not running: " + ", ".join(down) + ".\n"
            "Start them with:  uv run python manage_services.py start",
            returncode=3,
        )

    if not ROBOFLOW_API_KEY:
        pytest.exit(
            "ROBOFLOW_API_KEY is not set. Configure it in "
            f"{_DETECTOR_ENV.relative_to(ROOT)}",
            returncode=3,
        )


# ---------------------------------------------------------------------------
# Scenario / fixture loading
# ---------------------------------------------------------------------------


def load_scenarios() -> list[dict[str, Any]]:
    """Load the E2E scenarios manifest."""
    with SCENARIOS_MANIFEST.open("r", encoding="utf-8") as f:
        scenarios = json.load(f)
    return [s for s in scenarios if s.get("enabled", True)]


def load_expected_fixture(scenario_id: str, stage: str) -> dict[str, Any]:
    """Load a canonical expected-output fixture for a given scenario + stage.

    stage is one of: "enricher", "hand_state", "decision".
    """
    path = FIXTURES_DIR / scenario_id / f"expected_{stage}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def screenshot_path(scenario: dict[str, Any]) -> Path:
    """Resolve the screenshot path for a scenario to an absolute Path."""
    rel = scenario["screenshot_path"]
    p = (ROOT / rel).resolve()
    if not p.exists():
        pytest.fail(f"Screenshot not found for scenario {scenario['id']}: {p}")
    return p


# ---------------------------------------------------------------------------
# Pipeline stage callers (live HTTP) — shared with capture_fixtures.py
# ---------------------------------------------------------------------------


def call_detector(image_bytes: bytes) -> list[dict[str, Any]]:
    """Call the Roboflow object detector. Non-deterministic — log only."""
    response = requests.post(
        ROBOFLOW_API_URL,
        params={"api_key": ROBOFLOW_API_KEY},
        files={"file": image_bytes},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("Detector response did not contain a predictions list")
    return predictions


def call_enricher(
    image_bytes: bytes, detections: list[dict[str, Any]]
) -> dict[str, Any]:
    import base64

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    response = requests.post(
        ENRICHER_URL,
        json={"image_base64": image_base64, "detections": detections},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def call_hand_state_parser(enriched: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        HAND_STATE_PARSER_URL,
        json=enriched,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def call_decision_engine(hand_state: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        DECISION_ENGINE_URL,
        json=hand_state,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Normalization helpers — keep assertions stable across minor variations
# ---------------------------------------------------------------------------


def normalize_enricher(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip non-deterministic fields (confidence, bbox coords) from enricher
    output so that fixture comparisons are stable.

    Keeps: class_name, classification, ocr_text, spatial_info, owner_player,
    dealer_player, hero_position, position.
    """
    objects = []
    for obj in payload.get("objects", []):
        normalized: dict[str, Any] = {"class_name": obj.get("class_name")}
        if "classification" in obj:
            normalized["classification"] = obj["classification"]
        if "ocr_text" in obj:
            normalized["ocr_text"] = obj["ocr_text"]
        spatial = obj.get("spatial_info") or {}
        if spatial:
            normalized["spatial_info"] = {
                k: v
                for k, v in spatial.items()
                if k
                in (
                    "owner_player",
                    "dealer_player",
                    "hero_position",
                    "position",
                )
            }
        objects.append(normalized)
    return {"objects": objects}


def normalize_hand_state(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize hand-state JSON for stable comparison.

    Drops schema_version (may differ by revision) and sorts action_history
    to avoid ordering sensitivity.
    """
    out = dict(payload)
    out.pop("schema_version", None)
    history = out.get("action_history")
    if isinstance(history, list):
        out["action_history"] = sorted(
            history,
            key=lambda e: (
                e.get("player", ""),
                e.get("action", ""),
                str(e.get("amount", "")),
            ),
        )
    return out


def normalize_decision(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize decision JSON. Keeps action + amount; drops reason wording
    (asserted via substring, not exact match)."""
    return {
        "action": payload.get("action"),
        "amount": payload.get("amount"),
    }


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def scenarios() -> list[dict[str, Any]]:
    return load_scenarios()


@pytest.fixture
def scenario(request) -> dict[str, Any]:
    return request.param


# ---------------------------------------------------------------------------
# Smoke marker — applied to scenarios flagged smoke=true in the manifest
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):  # noqa: ANN001
    """Mark scenarios with smoke=true so `-m smoke` selects only them."""
    for item in items:
        if not hasattr(item, "callspec"):
            continue
        scenario = item.callspec.params.get("scenario")
        if isinstance(scenario, dict) and scenario.get("smoke"):
            item.add_marker(pytest.mark.smoke)
