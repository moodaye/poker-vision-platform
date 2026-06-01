"""Tests for upload_to_roboflow.py — labelmap construction and upload logic."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from upload_to_roboflow import load_labelmap, upload_image

# ---------------------------------------------------------------------------
# load_labelmap — must produce string-keyed dict from data.yaml
# ---------------------------------------------------------------------------


class TestLoadLabelmap:
    """
    load_labelmap() reads data.yaml and must return {"0": "class_name", ...}.
    This dict is JSON-serialised and passed to Roboflow as the labelmap param.
    All keys must be strings (not ints) so the JSON is {"0": ..., "1": ...}.
    """

    def _write_data_yaml(self, tmp_path: Path, names: list[str]) -> Path:
        data = {
            "path": str(tmp_path),
            "train": "images",
            "val": "images",
            "names": names,
        }
        p = tmp_path / "data.yaml"
        with open(p, "w") as f:
            yaml.dump(data, f)
        return p

    def test_keys_are_strings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        p = self._write_data_yaml(
            tmp_path, ["bet_pot_button", "check_button", "player_me"]
        )
        import upload_to_roboflow as m

        monkeypatch.setattr(m, "DATA_YAML", p)
        labelmap = load_labelmap()
        for k in labelmap:
            assert isinstance(k, str), (
                f"key {k!r} must be a string, not {type(k).__name__}"
            )

    def test_values_are_class_names(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        names = ["bet_pot_button", "check_button", "player_me"]
        p = self._write_data_yaml(tmp_path, names)
        import upload_to_roboflow as m

        monkeypatch.setattr(m, "DATA_YAML", p)
        labelmap = load_labelmap()
        assert labelmap == {
            "0": "bet_pot_button",
            "1": "check_button",
            "2": "player_me",
        }

    def test_json_serialisable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The labelmap must serialise to valid JSON without errors."""
        p = self._write_data_yaml(tmp_path, ["chip_stack", "dealer_button"])
        import upload_to_roboflow as m

        monkeypatch.setattr(m, "DATA_YAML", p)
        labelmap = load_labelmap()
        serialised = json.dumps(labelmap)
        parsed = json.loads(serialised)
        assert parsed == labelmap

    def test_all_18_classes_present(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression: must load all 18 expected classes without truncation."""
        names = [
            "bet_pot_button",
            "check_button",
            "check_fold_button",
            "chip_stack",
            "dealer_button",
            "flop_card",
            "fold_button",
            "holecard",
            "level",
            "nextblinds",
            "player_me",
            "player_name",
            "player_other",
            "poker-table",
            "prizepool",
            "total_pot",
            "win_card",
            "win_chips",
        ]
        p = self._write_data_yaml(tmp_path, names)
        import upload_to_roboflow as m

        monkeypatch.setattr(m, "DATA_YAML", p)
        labelmap = load_labelmap()
        assert len(labelmap) == 18
        assert labelmap["10"] == "player_me"
        assert labelmap["13"] == "poker-table"


# ---------------------------------------------------------------------------
# upload_image — duplicate response must not silently return None
# ---------------------------------------------------------------------------


class TestUploadImageDuplicate:
    """
    When Roboflow returns {"duplicate": true} (image already uploaded),
    the current code does data.get("id") which returns None.
    This causes the annotation upload to be skipped silently.
    This test documents the bug so it is visible and caught if/when fixed.
    """

    def _mock_post(self, response_json: dict[str, object]) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_json
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_fresh_upload_returns_id(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)  # minimal fake PNG

        import upload_to_roboflow as m

        monkeypatch.setattr(m, "ROBOFLOW_API_KEY", "test-key")

        with patch("upload_to_roboflow.requests.post") as mock_post:
            mock_post.return_value = self._mock_post({"success": True, "id": "abc123"})
            result = upload_image(img)

        assert result == "abc123"

    def test_duplicate_upload_returns_none_bug(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        BUG: when a duplicate image is uploaded, Roboflow returns
        {"duplicate": true} with no "id" field. upload_image() returns None,
        which causes the annotation to be silently skipped.

        This test documents the current (buggy) behaviour. When the bug is
        fixed (e.g. by querying Roboflow for the existing image id), this
        test should be updated to assert the returned id is a non-empty string.
        """
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        import upload_to_roboflow as m

        monkeypatch.setattr(m, "ROBOFLOW_API_KEY", "test-key")

        with patch("upload_to_roboflow.requests.post") as mock_post:
            mock_post.return_value = self._mock_post({"duplicate": True})
            result = upload_image(img)

        # Documents the bug: None means the annotation upload will be skipped
        assert result is None, (
            "If this assertion fails, the duplicate-handling bug has been fixed. "
            "Update this test to assert result is a valid image id string."
        )
