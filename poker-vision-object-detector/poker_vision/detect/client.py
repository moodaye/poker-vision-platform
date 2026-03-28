"""
Roboflow hosted inference HTTP client with retry + exponential backoff.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any, cast

import requests

from .config import RoboflowConfig

logger = logging.getLogger(__name__)

# HTTP status codes that are worth retrying
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class RoboflowAPIError(Exception):
    """Raised when the Roboflow API returns an unrecoverable error."""

    def __init__(self, message: str, http_status: int | None = None):
        super().__init__(message)
        self.http_status = http_status


class RoboflowClient:
    """
    Thin HTTP client for the Roboflow hosted inference API.

    Uploads image bytes as base64-encoded JSON body and returns the
    parsed JSON prediction dict.
    """

    def __init__(self, cfg: RoboflowConfig, api_key: str) -> None:
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY is not set")
        self._cfg = cfg
        self._api_key = api_key
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def predict(self, image_path: Path) -> dict[str, Any]:
        """
        Upload *image_path* to Roboflow and return the raw JSON response dict.

        Retries up to cfg.max_retries on transient failures with exponential backoff.
        Raises RoboflowAPIError on non-retryable failure or exhausted retries.
        """
        url = self._build_url()
        image_b64 = self._encode_image(image_path)

        last_exc: Exception | None = None
        last_status: int | None = None

        for attempt in range(self._cfg.max_retries + 1):
            if attempt > 0:
                sleep_time = self._cfg.backoff_seconds * (2 ** (attempt - 1))
                logger.debug(
                    "Retry %d/%d after %.1fs sleep",
                    attempt,
                    self._cfg.max_retries,
                    sleep_time,
                )
                time.sleep(sleep_time)

            try:
                response = self._session.post(
                    url,
                    params=self._query_params(),
                    data=image_b64,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=self._cfg.timeout_seconds,
                )
                last_status = response.status_code

                if response.status_code == 200:
                    return cast(dict[str, Any], response.json())

                if response.status_code not in _RETRYABLE_STATUS:
                    raise RoboflowAPIError(
                        f"Roboflow API returned HTTP {response.status_code}: {response.text[:200]}",
                        http_status=response.status_code,
                    )

                # Retryable — keep looping
                last_exc = RoboflowAPIError(
                    f"Roboflow API transient error HTTP {response.status_code}",
                    http_status=response.status_code,
                )
                logger.warning(
                    "Transient API error HTTP %d on attempt %d for %s",
                    response.status_code,
                    attempt,
                    image_path.name,
                )

            except requests.exceptions.Timeout as exc:
                last_exc = exc
                logger.warning(
                    "Request timed out on attempt %d for %s", attempt, image_path.name
                )
            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                logger.warning(
                    "Connection error on attempt %d for %s: %s",
                    attempt,
                    image_path.name,
                    exc,
                )
            except RoboflowAPIError:
                raise  # Non-retryable — propagate immediately

        # Exhausted retries
        msg = f"All {self._cfg.max_retries + 1} attempts failed for {image_path.name}"
        if last_status is not None:
            raise RoboflowAPIError(msg, http_status=last_status) from last_exc
        raise RoboflowAPIError(msg) from last_exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_url(self) -> str:
        base = self._cfg.api_base.rstrip("/")
        project = self._cfg.project
        version = self._cfg.version
        return f"{base}/{project}/{version}"

    def _query_params(self) -> dict[str, Any]:
        return {
            "api_key": self._api_key,
            "confidence": self._cfg.confidence_threshold,
            "overlap": self._cfg.overlap_threshold,
        }

    @staticmethod
    def _encode_image(image_path: Path) -> bytes:
        """Read image file and return base64-encoded bytes suitable for the request body."""
        with image_path.open("rb") as fh:
            raw = fh.read()
        return base64.b64encode(raw)
