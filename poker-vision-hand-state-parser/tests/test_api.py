from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import sys

PARSER_DIR = Path(__file__).resolve().parents[1]
API_PATH = PARSER_DIR / "api.py"
spec = importlib.util.spec_from_file_location("hand_state_parser_api", API_PATH)
assert spec is not None and spec.loader is not None
api = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api
spec.loader.exec_module(api)



def test_parse_logs_diagnostics_when_enabled(monkeypatch, caplog) -> None:
    monkeypatch.setenv("HAND_STATE_PARSER_LOG_DIAGNOSTICS", "true")

    expected_hand_state = {
        "schema_version": "2.2.0",
        "position": "BTN",
        "hero_seat": "BTN",
        "hero_cards": ["Ah", "Kd"],
    }
    expected_diagnostics = {
        "hero_cards": {
            "source": "holecard+holecard",
            "field_conf": 0.95,
            "band": "trusted",
            "fallback_used": False,
            "warning": None,
        }
    }
    calls = {"simple": 0, "diagnostic": 0}

    def fake_build_hand_state(data):
        calls["simple"] += 1
        raise AssertionError("build_hand_state should not be used when diagnostics logging is enabled")

    def fake_build_hand_state_with_diagnostics(data):
        calls["diagnostic"] += 1
        return expected_hand_state, expected_diagnostics

    monkeypatch.setattr(api, "build_hand_state", fake_build_hand_state)
    monkeypatch.setattr(api, "build_hand_state_with_diagnostics", fake_build_hand_state_with_diagnostics)

    client = api.app.test_client()
    with caplog.at_level(logging.INFO):
        response = client.post("/parse", json={"objects": [{"class_name": "holecard"}]})

    assert response.status_code == 200
    assert response.get_json() == expected_hand_state
    assert calls == {"simple": 0, "diagnostic": 1}
    assert "Parsed hand state:" in caplog.text
    assert "Hand state diagnostics:" in caplog.text
    assert '"schema_version": "2.2.0"' in caplog.text
    assert '"source": "holecard+holecard"' in caplog.text



def test_parse_uses_simple_builder_when_diagnostics_disabled(monkeypatch, caplog) -> None:
    monkeypatch.delenv("HAND_STATE_PARSER_LOG_DIAGNOSTICS", raising=False)

    expected_hand_state = {
        "schema_version": "2.2.0",
        "position": "SB",
        "hero_seat": "SB",
        "hero_cards": ["Qs", "Jh"],
    }
    calls = {"simple": 0, "diagnostic": 0}

    def fake_build_hand_state(data):
        calls["simple"] += 1
        return expected_hand_state

    def fake_build_hand_state_with_diagnostics(data):
        calls["diagnostic"] += 1
        raise AssertionError("build_hand_state_with_diagnostics should not be used when diagnostics logging is disabled")

    monkeypatch.setattr(api, "build_hand_state", fake_build_hand_state)
    monkeypatch.setattr(api, "build_hand_state_with_diagnostics", fake_build_hand_state_with_diagnostics)

    client = api.app.test_client()
    with caplog.at_level(logging.INFO):
        response = client.post("/parse", json={"objects": [{"class_name": "holecard"}]})

    assert response.status_code == 200
    assert response.get_json() == expected_hand_state
    assert calls == {"simple": 1, "diagnostic": 0}
    assert "Parsed hand state:" not in caplog.text
    assert "Hand state diagnostics:" not in caplog.text
