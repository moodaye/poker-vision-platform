"""
CLI entry point for poker_vision.detect.

Usage:
    python -m poker_vision.detect --config detect_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import load_config
from .runner import run

logger = logging.getLogger(__name__)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="poker_vision.detect",
        description="Run Roboflow hosted inference on local images.",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the YAML config file (e.g. detect_config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        metavar="PATH",
        help="Path to the .env file to load (default: .env in current directory)",
    )

    args = parser.parse_args(argv)
    _setup_logging(args.log_level)

    # Load .env file — override=False so already-set env vars are respected
    env_path = Path(args.env_file)
    loaded = load_dotenv(dotenv_path=env_path, override=False)
    if loaded:
        logger.debug("Loaded environment from %s", env_path)
    else:
        logger.debug(
            "No .env file found at %s (continuing with existing env)", env_path
        )

    # Load config
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Configuration error: %s", exc)
        return 2

    # Get API key — never log or print it
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        logger.error(
            "ROBOFLOW_API_KEY is not set. "
            "Add it to your .env file or set it as an environment variable."
        )
        return 2

    try:
        exit_code = run(cfg, api_key)
    except FileNotFoundError as exc:
        logger.error("Runtime error: %s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error: %s", exc)
        return 2

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
