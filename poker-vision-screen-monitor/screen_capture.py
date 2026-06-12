import base64
import json
import logging
import os
import threading
import time
from datetime import datetime
from io import BytesIO
from typing import Any, cast

import requests
from PIL import Image, ImageDraw, ImageFont

# Screen capture library selection
CAPTURE_LIBRARY: str | None
try:
    import mss

    CAPTURE_LIBRARY = "mss"
except ImportError:
    try:
        import pyautogui

        CAPTURE_LIBRARY = "pyautogui"
    except ImportError:
        CAPTURE_LIBRARY = None

logger = logging.getLogger(__name__)


class ScreenCaptureService:
    def __init__(self) -> None:
        self._capturing = False
        self._capture_thread: threading.Thread | None = None
        self._latest_image: Image.Image | None = None
        self._last_capture_time: str | None = None
        self._last_error: str | None = None
        self._stats: dict[str, Any] = {
            "total_captures": 0,
            "failed_captures": 0,
            "start_time": None,
        }
        self._config: dict[str, Any] = {
            "interval": 1.0,  # seconds
            "quality": 85,  # JPEG quality
            "resize_factor": 0.5,  # scale factor for resizing (reduced from 1.0)
            "add_timestamp": True,  # add timestamp to image
            "monitor": 0,  # monitor index for multi-monitor setups
            "capture_mode": "interval",  # 'interval' or 'manual'
            "webhook_urls": ["http://127.0.0.1:5100/decide"],  # URLs to send captured images to
            "send_to_external": False,  # enable/disable external sending
            "external_format": "multipart",  # 'base64' or 'multipart'
            "webhook_quality": 60,  # Quality for webhook images (lower than display quality)
            "webhook_max_width": 1280,  # Maximum width for webhook images
            "webhook_max_height": 720,  # Maximum height for webhook images
            "webhook_timeout": 40,  # Timeout in seconds for webhook requests
            "transform_enabled": False,  # enable/disable preprocessing transforms
            "transport_format": "png",  # 'png' or 'jpeg' for webhook transport
            "transport_optimize_enabled": False,  # enable webhook transport optimization
            "save_local": False,  # Save captures to local folder
            "save_path": os.path.abspath("captures"),  # Folder path for local saves
            "save_format": "png",  # 'png' (lossless) or 'jpg'
            "save_processed": False,  # Save processed image instead of raw
        }

        # Check if screen capture is available
        if CAPTURE_LIBRARY is None:
            logger.error(
                "No screen capture library available. Install 'mss' or 'pyautogui'"
            )
        else:
            logger.info(f"Using {CAPTURE_LIBRARY} for screen capture")

    def is_capturing(self) -> bool:
        """Check if currently capturing"""
        return self._capturing

    def get_config(self) -> dict[str, Any]:
        """Get current configuration"""
        return self._config.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get capture statistics"""
        stats = self._stats.copy()
        if stats["start_time"] and self._capturing:
            stats["uptime_seconds"] = (
                datetime.now() - stats["start_time"]
            ).total_seconds()
        return stats

    def get_last_error(self) -> str | None:
        """Get last error message"""
        return self._last_error

    def get_latest_image(self) -> Image.Image | None:
        """Get the latest captured image"""
        return self._latest_image

    def get_last_capture_time(self) -> str | None:
        """Get timestamp of last capture"""
        return self._last_capture_time

    def feed_external_image(self, image: Image.Image) -> bool:
        """Accept an external image and process it as if it was captured"""
        try:
            self._handle_captured_image(image)
            return True
        except Exception as e:
            logger.error(f"Error feeding external image: {str(e)}")
            self._last_error = str(e)
            self._stats["failed_captures"] += 1
            return False

    def update_config(self, new_config: dict[str, Any]) -> bool:
        """Update configuration"""
        try:
            # Validate configuration values
            if "interval" in new_config:
                interval = float(new_config["interval"])
                if interval <= 0:
                    raise ValueError("Interval must be positive")
                self._config["interval"] = interval

            if "quality" in new_config:
                quality = int(new_config["quality"])
                if not 1 <= quality <= 100:
                    raise ValueError("Quality must be between 1 and 100")
                self._config["quality"] = quality

            if "resize_factor" in new_config:
                resize_factor = float(new_config["resize_factor"])
                if resize_factor <= 0:
                    raise ValueError("Resize factor must be positive")
                self._config["resize_factor"] = resize_factor

            if "add_timestamp" in new_config:
                self._config["add_timestamp"] = bool(new_config["add_timestamp"])

            if "monitor" in new_config:
                monitor = int(new_config["monitor"])
                if monitor < 0:
                    raise ValueError("Monitor index must be non-negative")
                self._config["monitor"] = monitor

            if "save_local" in new_config:
                self._config["save_local"] = bool(new_config["save_local"])

            if "save_path" in new_config:
                save_path = str(new_config["save_path"]).strip()
                if not save_path:
                    raise ValueError("Save path cannot be empty")
                self._config["save_path"] = os.path.abspath(
                    os.path.expanduser(save_path)
                )

            if "save_format" in new_config:
                save_format = str(new_config["save_format"]).lower().strip()
                if save_format not in ["png", "jpg", "jpeg"]:
                    raise ValueError("Save format must be 'png' or 'jpg'")
                self._config["save_format"] = (
                    "jpg" if save_format == "jpeg" else save_format
                )

            if "save_processed" in new_config:
                self._config["save_processed"] = bool(new_config["save_processed"])

            if "webhook_timeout" in new_config:
                webhook_timeout = float(new_config["webhook_timeout"])
                if webhook_timeout <= 0:
                    raise ValueError("Webhook timeout must be positive")
                self._config["webhook_timeout"] = webhook_timeout

            if "transform_enabled" in new_config:
                self._config["transform_enabled"] = bool(new_config["transform_enabled"])

            if "transport_format" in new_config:
                transport_format = str(new_config["transport_format"]).strip().lower()
                if transport_format not in {"png", "jpeg"}:
                    raise ValueError("transport_format must be 'png' or 'jpeg'")
                self._config["transport_format"] = transport_format

            if "transport_optimize_enabled" in new_config:
                self._config["transport_optimize_enabled"] = bool(
                    new_config["transport_optimize_enabled"]
                )

            if (
                not self._config["transform_enabled"]
                and self._config["transport_format"] == "jpeg"
            ):
                raise ValueError(
                    "transport_format 'jpeg' requires transform_enabled to be true"
                )

            if "capture_mode" in new_config:
                capture_mode = str(new_config["capture_mode"]).strip().lower()
                if capture_mode not in {"interval", "manual"}:
                    raise ValueError("capture_mode must be 'interval' or 'manual'")
                self._config["capture_mode"] = capture_mode

            logger.info(f"Configuration updated: {self._config}")
            return True

        except (ValueError, TypeError) as e:
            logger.error(f"Invalid configuration: {str(e)}")
            self._last_error = f"Invalid configuration: {str(e)}"
            return False

    def start_capture(
        self, interval: float | None = None, quality: int | None = None
    ) -> bool:
        """Start screen capture"""
        if CAPTURE_LIBRARY is None:
            self._last_error = "No screen capture library available"
            return False

        if self._capturing:
            logger.warning("Capture already running")
            return True

        # Update config if provided
        if interval is not None:
            self._config["interval"] = float(interval)
        if quality is not None:
            self._config["quality"] = int(quality)

        try:
            self._capturing = True
            self._stats["start_time"] = datetime.now()
            self._stats["total_captures"] = 0
            self._stats["failed_captures"] = 0
            self._last_error = None

            # Start capture thread
            self._capture_thread = threading.Thread(
                target=self._capture_loop, daemon=True
            )
            self._capture_thread.start()

            logger.info("Screen capture started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start capture: {str(e)}")
            self._last_error = str(e)
            self._capturing = False
            return False

    def stop_capture(self) -> None:
        """Stop screen capture"""
        if not self._capturing:
            logger.warning("Capture not running")
            return

        self._capturing = False

        # Wait for thread to finish
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5.0)

        logger.info("Screen capture stopped")

    def capture_once(self) -> bool:
        """Capture one screenshot and optionally send it to external webhooks."""
        try:
            image = self._capture_screenshot()

            if not image:
                self._stats["failed_captures"] += 1
                logger.warning("Failed to capture screenshot")
                return False

            self._handle_captured_image(image)
            logger.info("Manual capture taken")
            return True

        except Exception as e:
            logger.error(f"Error capturing once: {str(e)}")
            self._last_error = str(e)
            self._stats["failed_captures"] += 1
            return False

    def _handle_captured_image(self, raw_image: Image.Image) -> None:
        """Apply preprocessing, save, send and update latest preview."""
        outbound_image = (
            self._process_image(raw_image)
            if self._config.get("transform_enabled", False)
            else raw_image
        )

        self._latest_image = outbound_image
        self._last_capture_time = datetime.now().isoformat()
        self._stats["total_captures"] += 1

        if self._config.get("save_local", False):
            self._save_image(raw_image, outbound_image)

        if self._config.get("send_to_external", False):
            self._send_to_external_systems(outbound_image)

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread"""
        logger.info("Starting capture loop")

        while self._capturing:
            try:
                # Capture screenshot
                image = self._capture_screenshot()

                if self._config.get("capture_mode", "interval") == "interval":
                    if image:
                        self._handle_captured_image(image)
                    else:
                        self._stats["failed_captures"] += 1
                        logger.warning("Failed to capture screenshot")
                else:
                    logger.debug("Manual capture mode active; skipping interval capture")

            except Exception as e:
                logger.error(f"Error in capture loop: {str(e)}")
                self._last_error = str(e)
                self._stats["failed_captures"] += 1

            # Wait for next capture
            time.sleep(self._config["interval"])

        logger.info("Capture loop ended")

    def _save_image(self, raw_image: Image.Image, processed_image: Image.Image) -> None:
        """Save captured image to local folder if enabled"""
        try:
            save_path = self._config.get("save_path", "captures")
            save_format = self._config.get("save_format", "png")
            use_processed = self._config.get("save_processed", False)

            os.makedirs(save_path, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"capture_{timestamp}.{save_format}"
            full_path = os.path.join(save_path, filename)

            image_to_save = processed_image if use_processed else raw_image

            if save_format == "jpg":
                quality = int(self._config.get("quality", 85))
                image_to_save.save(full_path, format="JPEG", quality=quality)
            else:
                image_to_save.save(full_path, format="PNG")

        except Exception as e:
            logger.error(f"Failed to save image: {str(e)}")

    def _capture_screenshot(self) -> Image.Image:
        """Capture screenshot using available library"""
        try:
            if CAPTURE_LIBRARY == "mss":
                # Check if we're in a headless environment (Linux/Mac only check)
                import os
                import platform

                is_headless = False

                # Only check for headless on non-Windows systems
                if platform.system() != "Windows":
                    if not os.environ.get("DISPLAY") and not os.environ.get(
                        "WAYLAND_DISPLAY"
                    ):
                        is_headless = True

                if is_headless:
                    # Create a simple test image for demonstration in headless environment
                    logger.warning("No display available - creating test image")
                    return self._create_test_image()

                with mss.mss() as sct:
                    # Get monitor info
                    monitors = sct.monitors
                    monitor_index = min(self._config["monitor"], len(monitors) - 1)
                    monitor = (
                        monitors[monitor_index + 1]
                        if monitor_index < len(monitors) - 1
                        else monitors[1]
                    )

                    # Capture screenshot
                    screenshot = sct.grab(monitor)
                    image = Image.frombytes(
                        "RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"
                    )
                    return image

            elif CAPTURE_LIBRARY == "pyautogui":
                # Check if we're in a headless environment (Linux/Mac only check)
                import os
                import platform

                is_headless = False

                # Only check for headless on non-Windows systems
                if platform.system() != "Windows":
                    if not os.environ.get("DISPLAY") and not os.environ.get(
                        "WAYLAND_DISPLAY"
                    ):
                        is_headless = True

                if is_headless:
                    # Create a simple test image for demonstration in headless environment
                    logger.warning("No display available - creating test image")
                    return self._create_test_image()

                # pyautogui doesn't support monitor selection easily
                screenshot = pyautogui.screenshot()
                return cast(Image.Image, screenshot)

            else:
                raise RuntimeError("No capture library available")

        except Exception as e:
            logger.error(f"Screenshot capture failed: {str(e)}")
            # If screen capture fails, create a test image to demonstrate functionality
            logger.info("Creating test image due to capture failure")
            return self._create_test_image()

    def _create_test_image(self) -> Image.Image:
        """Create a test image for demonstration purposes"""
        from datetime import datetime

        # Create a 800x600 test image
        image = Image.new(
            "RGB", (800, 600), color=(45, 55, 72)
        )  # Dark blue-gray background
        draw = ImageDraw.Draw(image)

        # Draw some test content
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # Draw title
        title = "Screen Monitor - Test Mode"
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(
            ((800 - title_width) // 2, 200), title, fill=(255, 255, 255), font=font
        )

        # Draw timestamp
        timestamp = datetime.now().strftime("Captured: %Y-%m-%d %H:%M:%S")
        timestamp_bbox = draw.textbbox((0, 0), timestamp, font=font)
        timestamp_width = timestamp_bbox[2] - timestamp_bbox[0]
        draw.text(
            ((800 - timestamp_width) // 2, 250),
            timestamp,
            fill=(200, 200, 200),
            font=font,
        )

        # Draw info message
        info = "Running in headless environment - showing test image"
        info_bbox = draw.textbbox((0, 0), info, font=font)
        info_width = info_bbox[2] - info_bbox[0]
        draw.text(((800 - info_width) // 2, 300), info, fill=(150, 150, 150), font=font)

        # Draw some geometric shapes for visual interest
        draw.rectangle(
            [100, 400, 200, 500], fill=(52, 152, 219), outline=(255, 255, 255)
        )  # Blue square
        draw.ellipse(
            [250, 400, 350, 500], fill=(231, 76, 60), outline=(255, 255, 255)
        )  # Red circle
        draw.polygon(
            [(450, 400), (500, 450), (450, 500), (400, 450)],
            fill=(46, 204, 113),
            outline=(255, 255, 255),
        )  # Green diamond
        draw.rectangle(
            [550, 400, 650, 500], fill=(155, 89, 182), outline=(255, 255, 255)
        )  # Purple square

        return image

    def _process_image(self, image: Image.Image) -> Image.Image:
        """Process captured image"""
        try:
            # Resize if needed
            if self._config["resize_factor"] != 1.0:
                new_size = (
                    int(image.width * self._config["resize_factor"]),
                    int(image.height * self._config["resize_factor"]),
                )
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Add timestamp if enabled
            if self._config["add_timestamp"]:
                image = self._add_timestamp(image)

            return image

        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            return image  # Return original image if processing fails

    def _add_timestamp(self, image: Image.Image) -> Image.Image:
        """Add timestamp to image"""
        try:
            # Create a copy to avoid modifying original
            img_copy = image.copy()
            draw = ImageDraw.Draw(img_copy)

            # Get timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Try to use a default font, fallback to default if not available
            try:
                font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = (
                    ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20
                    )
                )
            except Exception:
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None

            # Position timestamp at bottom-right
            text_bbox = draw.textbbox((0, 0), timestamp, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            x = image.width - text_width - 10
            y = image.height - text_height - 10

            # Draw background rectangle
            draw.rectangle(
                [x - 5, y - 5, x + text_width + 5, y + text_height + 5],
                fill=(0, 0, 0, 128),
            )

            # Draw timestamp text
            draw.text((x, y), timestamp, fill=(255, 255, 255), font=font)

            return img_copy

        except Exception as e:
            logger.error(f"Failed to add timestamp: {str(e)}")
            return image  # Return original if timestamp addition fails

    def _optimize_image_for_webhook(self, image: Image.Image) -> Image.Image:
        """Optimize image specifically for webhook transmission"""
        max_width = self._config.get("webhook_max_width", 1280)
        max_height = self._config.get("webhook_max_height", 720)

        # Calculate resize factor to fit within max dimensions
        width_factor = max_width / image.width if image.width > max_width else 1.0
        height_factor = max_height / image.height if image.height > max_height else 1.0
        resize_factor = min(width_factor, height_factor)

        # Only resize if needed
        if resize_factor < 1.0:
            new_width = int(image.width * resize_factor)
            new_height = int(image.height * resize_factor)
            optimized_image = image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )
            logger.debug(
                f"Optimized image for webhook: {image.size} -> {optimized_image.size}"
            )
            return optimized_image

        return image

    def _send_to_external_systems(self, image: Image.Image) -> None:
        """Send captured image to configured external systems"""
        webhook_urls = self._config.get("webhook_urls", [])
        if not webhook_urls:
            logger.debug("No webhook URLs configured - skipping external send")
            return

        logger.info(f"Sending image to {len(webhook_urls)} webhook(s)")

        optimized_image = image
        if self._config.get("transport_optimize_enabled", False):
            optimized_image = self._optimize_image_for_webhook(image)

        def send_async() -> None:
            for url in webhook_urls:
                try:
                    logger.info(f"Attempting to send image to: {url}")
                    self._send_image_to_url(optimized_image, url)
                    logger.info(f"Successfully sent image to: {url}")
                except Exception as e:
                    logger.error(f"Failed to send image to {url}: {str(e)}")
                    self._last_error = f"Webhook send failed to {url}: {str(e)}"

        # Send in background thread to avoid blocking capture
        threading.Thread(target=send_async, daemon=True).start()

    def _send_image_to_url(self, image: Image.Image, url: str) -> None:
        """Send image to a specific URL"""
        format_type = self._config.get("external_format", "base64")
        transport_format = self._config.get("transport_format", "png")
        optimize_transport = self._config.get("transport_optimize_enabled", False)
        logger.debug(
            f"Sending image to {url} in {format_type} format, transport={transport_format}, optimize={optimize_transport}"
        )

        if format_type == "base64":
            if transport_format == "png":
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()
                payload = {
                    "image": img_str,
                    "format": "png",
                    "metadata": {"source": "ScreenStream"},
                }
                response = requests.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self._config.get("webhook_timeout", 40),
                )
                response.raise_for_status()
                logger.info(f"Successfully sent image to {url} (base64 png)")
                self._handle_decision_response(response)
                return

            # JPEG transport path
            webhook_quality = self._config.get("quality", 85)
            if optimize_transport:
                max_attempts = 3
                quality_levels = [
                    webhook_quality,
                    max(webhook_quality - 20, 30),
                    max(webhook_quality - 40, 20),
                ]
                resize_factors = [1.0, 0.8, 0.6]
            else:
                max_attempts = 1
                quality_levels = [webhook_quality]
                resize_factors = [1.0]

            for attempt in range(max_attempts):
                current_quality = quality_levels[attempt]
                current_resize = resize_factors[attempt]
                working_image = image

                if current_resize < 1.0:
                    new_size = (
                        int(image.width * current_resize),
                        int(image.height * current_resize),
                    )
                    working_image = image.resize(new_size, Image.Resampling.LANCZOS)
                    logger.debug(
                        f"Attempt {attempt + 1}: Resized to {working_image.size}"
                    )

                buffer = BytesIO()
                working_image.save(buffer, format="JPEG", quality=current_quality)
                img_str = base64.b64encode(buffer.getvalue()).decode()
                payload = {
                    "image": img_str,
                    "format": "jpeg",
                    "metadata": {"source": "ScreenStream"},
                }
                response = requests.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self._config.get("webhook_timeout", 40),
                )

                logger.debug(f"Response status: {response.status_code}")
                logger.debug(f"Response headers: {dict(response.headers)}")
                if response.text:
                    logger.debug(f"Response body: {response.text[:200]}...")

                if response.status_code == 413 and attempt < max_attempts - 1:
                    logger.warning(
                        f"Attempt {attempt + 1}: Payload too large (413), trying smaller size/quality..."
                    )
                    continue

                response.raise_for_status()
                logger.info(
                    f"Successfully sent image to {url} (base64 jpeg) - Quality: {current_quality}%, Size: {working_image.size}"
                )
                self._handle_decision_response(response)
                return

            raise Exception(
                f"Failed to send image after {max_attempts} attempts with different sizes"
            )

        elif format_type == "multipart":
            buffer = BytesIO()
            if transport_format == "png":
                image.save(buffer, format="PNG")
                filename = "screenshot.png"
                content_type = "image/png"
            else:
                image.save(buffer, format="JPEG", quality=self._config.get("quality", 85))
                filename = "screenshot.jpg"
                content_type = "image/jpeg"

            buffer.seek(0)
            files = {"image": (filename, buffer, content_type)}
            data = {
                "timestamp": self._last_capture_time,
                "size": f"{image.size[0]}x{image.size[1]}",
                "source": "screen_monitor",
                "quality": str(self._config.get("quality", 85)),
            }

            logger.debug(f"Multipart data: {data}")

            response = requests.post(
                url,
                files=files,
                data=data,
                timeout=self._config.get("webhook_timeout", 40),
            )

            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            if response.text:
                logger.debug(f"Response body: {response.text[:200]}...")

            response.raise_for_status()
            logger.info(f"Successfully sent image to {url} (multipart)")
            self._handle_decision_response(response)

    def _handle_decision_response(self, response: requests.Response) -> None:
        """Log and voice a decision returned by the orchestrator."""
        logger.info(
            "Orchestrator response [%d]: %s", response.status_code, response.text[:500]
        )

        try:
            payload = response.json()
        except Exception as exc:
            logger.warning("Could not parse orchestrator response as JSON: %s", exc)
            return

        action = payload.get("action")
        if not action:
            logger.warning("Orchestrator response missing 'action' field: %s", payload)
            return

        amount = payload.get("amount")
        reason = payload.get("reason", "")

        amount_str = f" {amount:g}" if amount is not None else ""
        logger.info("*** DECISION: %s%s — %s ***", action.upper(), amount_str, reason)

        self._speak_decision(action, amount)

    def _speak_decision(self, action: str, amount: object) -> None:
        """Speak the decision using Windows built-in SAPI — no extra packages needed."""
        amount_int: int | None = None
        if isinstance(amount, int):
            amount_int = amount
        elif isinstance(amount, float):
            amount_int = int(amount)
        elif isinstance(amount, str):
            try:
                amount_int = int(float(amount))
            except ValueError:
                amount_int = None
        text = f"{action} {amount_int}" if amount_int is not None else action
        logger.info("Speaking: %r", text)

        try:
            import subprocess

            no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=no_window,
            )
        except Exception as exc:
            logger.warning("TTS failed: %s", exc)

    def add_webhook_url(self, url: str) -> bool:
        """Add a webhook URL for sending images"""
        if url not in self._config.get("webhook_urls", []):
            if "webhook_urls" not in self._config:
                self._config["webhook_urls"] = []
            self._config["webhook_urls"].append(url)
            logger.info(f"Added webhook URL: {url}")
            return True
        return False

    def remove_webhook_url(self, url: str) -> bool:
        """Remove a webhook URL"""
        if url in self._config.get("webhook_urls", []):
            self._config["webhook_urls"].remove(url)
            logger.info(f"Removed webhook URL: {url}")
            return True
        return False

    def get_webhook_urls(self) -> list[str]:
        """Get list of configured webhook URLs"""
        webhooks = self._config.get("webhook_urls", [])
        return [str(url) for url in webhooks] if isinstance(webhooks, list) else []

    def enable_external_sending(self, enabled: bool = True) -> None:
        """Enable or disable sending to external systems"""
        self._config["send_to_external"] = enabled
        logger.info(f"External sending {'enabled' if enabled else 'disabled'}")

    def set_external_format(self, format_type: str) -> bool:
        """Set the format for external sending ('base64' or 'multipart')"""
        if format_type in ["base64", "multipart"]:
            self._config["external_format"] = format_type
            logger.info(f"External format set to: {format_type}")
            return True
        return False
