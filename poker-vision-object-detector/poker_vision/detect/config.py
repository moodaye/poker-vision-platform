"""
Configuration dataclasses, YAML loading, defaults, and resolved config output.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RoboflowConfig:
    api_base: str = "https://detect.roboflow.com"
    project: str = ""
    version: int | str = 1
    timeout_seconds: int = 30
    max_retries: int = 3
    backoff_seconds: float = 1.0
    confidence_threshold: float = 0.50
    overlap_threshold: float = 0.50


@dataclass
class IOConfig:
    recursive_input: bool = True
    image_extensions: list[str] = field(
        default_factory=lambda: [".png", ".jpg", ".jpeg"]
    )


@dataclass
class NormalizationConfig:
    bbox_rounding: str = "round"


@dataclass
class RunConfig:
    run_id: str | None = None
    save_raw_predictions: bool = True
    save_normalized_detections: bool = True
    save_run_summary: bool = True


@dataclass
class DetectConfig:
    input_dir: str = ""
    output_dir: str = ""
    roboflow: RoboflowConfig = field(default_factory=RoboflowConfig)
    io: IOConfig = field(default_factory=IOConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    run: RunConfig = field(default_factory=RunConfig)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def _merge_dataclass(dc_instance: Any, data: dict[str, Any]) -> None:
    """Recursively update a dataclass instance from a dict, keeping defaults for missing keys."""
    for key, value in data.items():
        if not hasattr(dc_instance, key):
            continue
        current = getattr(dc_instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(dc_instance, key, value)


def load_config(path: str | Path) -> DetectConfig:
    """Load and validate config from a YAML file. Raises ValueError on invalid config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file must be a YAML mapping, got: {type(raw).__name__}"
        )

    cfg = DetectConfig()
    _merge_dataclass(cfg, raw)

    _validate_config(cfg)

    # Generate run_id if not provided
    if not cfg.run.run_id:
        cfg.run.run_id = datetime.datetime.now(tz=datetime.timezone.utc).strftime(
            "run_%Y%m%dT%H%M%SZ"
        )

    return cfg


def _validate_config(cfg: DetectConfig) -> None:
    """Raise ValueError with a descriptive message if the config is invalid."""
    errors = []
    if not cfg.input_dir:
        errors.append("input_dir is required")
    if not cfg.output_dir:
        errors.append("output_dir is required")
    if not cfg.roboflow.project:
        errors.append("roboflow.project is required")
    if not str(cfg.roboflow.version):
        errors.append("roboflow.version is required")
    if not (0.0 <= cfg.roboflow.confidence_threshold <= 1.0):
        errors.append("roboflow.confidence_threshold must be between 0 and 1")
    if not (0.0 <= cfg.roboflow.overlap_threshold <= 1.0):
        errors.append("roboflow.overlap_threshold must be between 0 and 1")
    if cfg.normalization.bbox_rounding not in ("round",):
        errors.append(
            f"normalization.bbox_rounding must be 'round', got '{cfg.normalization.bbox_rounding}'"
        )
    if errors:
        raise ValueError(
            "Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors)
        )


# ---------------------------------------------------------------------------
# Resolved config output (secrets excluded)
# ---------------------------------------------------------------------------


def resolved_config_dict(cfg: DetectConfig) -> dict:
    """Return a JSON-serialisable dict of the resolved config, excluding secrets."""
    d = asdict(cfg)
    # Remove any api_key that might be present (defensive; it is not in the dataclass,
    # but guard against future additions here)
    d.get("roboflow", {}).pop("api_key", None)
    return d


def save_resolved_config(cfg: DetectConfig, output_dir: str | Path) -> Path:
    """Write run_config.resolved.json under output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "run_config.resolved.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(resolved_config_dict(cfg), fh, indent=2)
    return out_path
