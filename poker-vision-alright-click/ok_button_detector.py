"""
OK Button Detection Module
Handles detection of OK buttons on screen using the Windows UI Automation API
(ctypes) as the primary method, with visual template matching as a fallback.
"""

import ctypes
import ctypes.wintypes
import logging
from typing import Any, cast

import config
import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import ImageFont

ImageArray = NDArray[np.uint8]

# Windows API constants
GW_HWNDNEXT = 2
GW_CHILD = 5
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
BM_CLICK = 0xF5

user32 = ctypes.windll.user32


def _enum_child_windows(parent_hwnd: int) -> list[int]:
    """Return all child window handles for a given parent."""
    children: list[int] = []

    EnumChildProc = ctypes.WINFUNCTYPE(
        ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
    )

    @EnumChildProc
    def callback(hwnd: int, _l_param: int) -> bool:
        children.append(hwnd)
        return True

    user32.EnumChildWindows(parent_hwnd, callback, 0)
    return children


def _get_window_text(hwnd: int) -> str:
    """Return the text/caption of a window handle."""
    length = user32.GetWindowTextLengthW(hwnd) + 1
    buf = ctypes.create_unicode_buffer(length)
    user32.GetWindowTextW(hwnd, buf, length)
    return buf.value


def _get_window_rect(hwnd: int) -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) screen rect for a window handle."""
    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.left, rect.top, rect.right, rect.bottom


class OKButtonDetector:
    """Detect OK buttons using Windows UI API (primary) and template matching (fallback)."""

    def __init__(
        self, confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD
    ) -> None:
        self.confidence_threshold = max(
            config.MIN_CONFIDENCE_THRESHOLD,
            min(config.MAX_CONFIDENCE_THRESHOLD, confidence_threshold),
        )
        self._pyautogui: Any | None = None
        logging.info(
            f"OK Button Detector initialized with confidence threshold: {self.confidence_threshold}"
        )

    def _get_pyautogui(self) -> Any:
        if self._pyautogui is None:
            import pyautogui

            self._pyautogui = pyautogui
            self._pyautogui.FAILSAFE = True
            self._pyautogui.PAUSE = config.CLICK_DELAY
        return self._pyautogui

    # ------------------------------------------------------------------
    # Primary detection: Windows UI Automation via ctypes
    # ------------------------------------------------------------------

    def find_ok_buttons_via_windows_api(self) -> list[tuple[int, int, float]]:
        """
        Walk all top-level and child windows looking for a button whose text
        is "OK" (case-insensitive).  Returns a list of (center_x, center_y,
        confidence=1.0) tuples for every match found.
        """
        results: list[tuple[int, int, float]] = []

        EnumWindowsProc = ctypes.WINFUNCTYPE(
            ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
        )

        @EnumWindowsProc
        def enum_callback(hwnd: int, _l_param: int) -> bool:
            if not user32.IsWindowVisible(hwnd):
                return True
            for child in _enum_child_windows(hwnd):
                text = _get_window_text(child).strip()
                if text.upper() in ("OK", "&OK"):
                    try:
                        left, top, right, bottom = _get_window_rect(child)
                        cx = (left + right) // 2
                        cy = (top + bottom) // 2
                        # Sanity-check: button must be on screen
                        if cx > 0 and cy > 0:
                            logging.info(
                                f"Windows API: OK button found at ({cx}, {cy}), hwnd={child}"
                            )
                            results.append((cx, cy, 1.0))
                    except Exception as exc:
                        logging.warning(f"Could not get rect for hwnd {child}: {exc}")
            return True

        user32.EnumWindows(enum_callback, 0)
        return results

    # ------------------------------------------------------------------
    # Fallback detection: screenshot + template matching
    # ------------------------------------------------------------------

    def capture_screen(self) -> ImageArray:
        """Capture the current screen as a BGR numpy array."""
        try:
            pyautogui = self._get_pyautogui()
            screenshot = pyautogui.screenshot()
            return cast(
                ImageArray, cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            )
        except Exception as e:
            logging.error(f"Failed to capture screen: {e}")
            raise

    def _build_templates(self) -> list[ImageArray]:
        """
        Build OK-button templates that resemble actual Windows buttons by
        rendering with a system font (Segoe UI) via PIL.
        """
        from PIL import Image, ImageDraw

        templates: list[ImageArray] = []

        sizes = [(75, 23), (75, 28), (90, 26), (90, 30)]
        labels = ["OK"]
        font_name = "segoeuib.ttf"  # Windows built-in
        font_sizes = [11, 12, 13]

        for label in labels:
            for w, h in sizes:
                for fsize in font_sizes:
                    for bg in (240, 255, 220):
                        img = Image.new("RGB", (w, h), color=(bg, bg, bg))
                        draw = ImageDraw.Draw(img)
                        # Outer border
                        draw.rectangle([0, 0, w - 1, h - 1], outline=(100, 100, 100))
                        # Try system font; fall back to default
                        try:
                            font: ImageFont.FreeTypeFont | ImageFont.ImageFont = (
                                ImageFont.truetype(font_name, fsize)
                            )
                        except OSError:
                            font = ImageFont.load_default()
                        bbox = draw.textbbox((0, 0), label, font=font)
                        tw = bbox[2] - bbox[0]
                        th = bbox[3] - bbox[1]
                        tx = (w - tw) // 2
                        ty = (h - th) // 2
                        draw.text((tx, ty), label, fill=(0, 0, 0), font=font)
                        gray = cast(
                            ImageArray, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                        )
                        templates.append(gray)

        return templates

    def find_ok_buttons_via_vision(self) -> list[tuple[int, int, float]]:
        """Template-matching fallback on a fresh screenshot."""
        screenshot = self.capture_screen()
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        detected: list[tuple[int, int, float]] = []

        for template in self._build_templates():
            try:
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                locs = np.where(result >= self.confidence_threshold)
                th, tw = template.shape
                for pt in zip(*locs[::-1], strict=False):
                    x, y = pt
                    confidence = float(result[y, x])
                    cx = x + tw // 2
                    cy = y + th // 2
                    if all(
                        np.sqrt((cx - ex) ** 2 + (cy - ey) ** 2) >= 30
                        for ex, ey, _ in detected
                    ):
                        detected.append((cx, cy, confidence))
                        logging.info(
                            f"Vision: OK button at ({cx},{cy}) conf={confidence:.3f}"
                        )
            except Exception as exc:
                logging.warning(f"Template match error: {exc}")

        detected.sort(key=lambda t: t[2], reverse=True)
        return detected

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect_ok_buttons(
        self,
        screenshot: ImageArray | None = None,
    ) -> list[tuple[int, int, float]]:
        """
        Detect OK buttons.  Uses Windows API first (most reliable), falls
        back to visual template matching.

        Args:
            screenshot: Ignored (kept for API compatibility). Always uses a
                        fresh capture for visual fallback.

        Returns:
            List of (center_x, center_y, confidence) sorted by confidence desc.
        """
        # Primary: Windows accessibility API
        api_results = self.find_ok_buttons_via_windows_api()
        if api_results:
            logging.info(f"Windows API found {len(api_results)} OK button(s)")
            return api_results

        # Fallback: visual template matching
        logging.info("Windows API found nothing; trying visual detection")
        return self.find_ok_buttons_via_vision()

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update the confidence threshold used by visual detection."""
        self.confidence_threshold = max(
            config.MIN_CONFIDENCE_THRESHOLD,
            min(config.MAX_CONFIDENCE_THRESHOLD, threshold),
        )
        logging.info(f"Confidence threshold updated to: {self.confidence_threshold}")
