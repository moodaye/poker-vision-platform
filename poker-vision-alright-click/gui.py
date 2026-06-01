"""
GUI Module for OK Button Detector
Provides a simple Tkinter interface for controlling the detection and clicking.
"""

import logging
import threading
import time
import tkinter as tk
from tkinter import scrolledtext, ttk

import config
from mouse_controller import MouseController
from ok_button_detector import OKButtonDetector


class OKButtonDetectorGUI:
    """Main GUI class for the OK Button Detector application."""

    def __init__(self, root: tk.Tk):
        """
        Initialize the GUI.

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.detector = OKButtonDetector()
        self.mouse_controller = MouseController()

        self.is_running = False
        self.detection_thread: threading.Thread | None = None

        self._setup_window()
        self._create_widgets()
        self._setup_logging_handler()

        logging.info("GUI initialized")

    def _setup_window(self) -> None:
        """Setup the main window properties."""
        self.root.title(config.WINDOW_TITLE)
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.root.resizable(True, True)

        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (config.WINDOW_WIDTH // 2)
        y = (self.root.winfo_screenheight() // 2) - (config.WINDOW_HEIGHT // 2)
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}+{x}+{y}")

    def _create_widgets(self) -> None:
        """Create and arrange GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame, text="OK Button Auto-Clicker", font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 20))

        # Controls frame
        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        controls_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        controls_frame.columnconfigure(1, weight=1)

        # Confidence threshold
        ttk.Label(controls_frame, text="Confidence Threshold:").grid(
            row=0, column=0, sticky=tk.W
        )
        self.confidence_var = tk.DoubleVar(value=config.DEFAULT_CONFIDENCE_THRESHOLD)
        confidence_scale = ttk.Scale(
            controls_frame,
            from_=config.MIN_CONFIDENCE_THRESHOLD,
            to=config.MAX_CONFIDENCE_THRESHOLD,
            variable=self.confidence_var,
            orient=tk.HORIZONTAL,
        )
        confidence_scale.grid(row=0, column=1, sticky="ew", padx=(10, 0))

        # Confidence value label
        self.confidence_label = ttk.Label(
            controls_frame, text=f"{self.confidence_var.get():.2f}"
        )
        self.confidence_label.grid(row=0, column=2, padx=(5, 0))

        # Bind scale change
        confidence_scale.configure(command=self._on_confidence_change)

        # Button frame
        button_frame = ttk.Frame(controls_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=(10, 0))

        # Start/Stop button
        self.start_stop_button = ttk.Button(
            button_frame, text="Start Detection", command=self._toggle_detection
        )
        self.start_stop_button.pack(side=tk.LEFT, padx=(0, 10))

        # Test detection button
        self.test_button = ttk.Button(
            button_frame, text="Test Detection", command=self._test_detection
        )
        self.test_button.pack(side=tk.LEFT)

        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        status_frame.columnconfigure(1, weight=1)

        # Status label
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W)
        self.status_label = ttk.Label(status_frame, text="Stopped", foreground="red")
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        # Statistics
        ttk.Label(status_frame, text="Detections:").grid(row=1, column=0, sticky=tk.W)
        self.detection_count_label = ttk.Label(status_frame, text="0")
        self.detection_count_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(status_frame, text="Clicks:").grid(row=2, column=0, sticky=tk.W)
        self.click_count_label = ttk.Label(status_frame, text="0")
        self.click_count_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))

        # Log frame
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        # Log text area
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=8, state=tk.DISABLED
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        # Clear log button
        clear_log_button = ttk.Button(
            log_frame, text="Clear Log", command=self._clear_log
        )
        clear_log_button.grid(row=1, column=0, pady=(5, 0))

        # Initialize counters
        self.detection_count = 0
        self.click_count = 0

    def _on_confidence_change(self, value: str) -> None:
        """Handle confidence threshold scale change."""
        confidence = float(value)
        self.confidence_label.config(text=f"{confidence:.2f}")
        self.detector.set_confidence_threshold(confidence)

    def _toggle_detection(self) -> None:
        """Start or stop the detection process."""
        if self.is_running:
            self.stop_detection()
        else:
            self.start_detection()

    def start_detection(self) -> None:
        """Start the detection process in a separate thread."""
        if self.is_running:
            return

        self.is_running = True
        self.start_stop_button.config(text="Stop Detection")
        self.test_button.config(state=tk.DISABLED)
        self.status_label.config(text="Running", foreground="green")

        # Start detection thread
        self.detection_thread = threading.Thread(
            target=self._detection_loop, daemon=True
        )
        self.detection_thread.start()

        self._log_message("Detection started")

    def stop_detection(self) -> None:
        """Stop the detection process."""
        if not self.is_running:
            return

        self.is_running = False
        self.start_stop_button.config(text="Start Detection")
        self.test_button.config(state=tk.NORMAL)
        self.status_label.config(text="Stopped", foreground="red")

        self._log_message("Detection stopped")

    def _detection_loop(self) -> None:
        """Main detection loop that runs in a separate thread."""
        while self.is_running:
            try:
                # Detect OK buttons
                detections = self.detector.detect_ok_buttons()

                if detections:
                    self.detection_count += len(detections)
                    self._update_detection_count()

                    # Click the best detection (highest confidence)
                    best_detection = detections[0]
                    x, y, confidence = best_detection

                    if self.mouse_controller.click_button(x, y):
                        self.click_count += 1
                        self._update_click_count()
                        self._log_message(
                            f"Clicked OK button at ({x}, {y}) with confidence {confidence:.3f}"
                        )
                    else:
                        self._log_message("Failed to click detected button")
                else:
                    self._log_message("No OK buttons detected")

                # Wait before next detection
                time.sleep(config.SCREEN_CAPTURE_DELAY)

            except Exception as e:
                self._log_message(f"Detection error: {e}")
                logging.error(f"Detection loop error: {e}")
                time.sleep(1)  # Wait a bit before retrying

    def _test_detection(self) -> None:
        """Test detection without clicking."""
        try:
            self._log_message("Testing detection...")
            detections = self.detector.detect_ok_buttons()

            if detections:
                self._log_message(f"Found {len(detections)} OK button(s):")
                for i, (x, y, confidence) in enumerate(detections[:5]):  # Show max 5
                    self._log_message(
                        f"  {i + 1}. Position: ({x}, {y}), Confidence: {confidence:.3f}"
                    )
            else:
                self._log_message("No OK buttons detected in current screen")

        except Exception as e:
            self._log_message(f"Test detection failed: {e}")
            logging.error(f"Test detection error: {e}")

    def _update_detection_count(self) -> None:
        """Update the detection count in the GUI."""
        self.root.after(
            0, lambda: self.detection_count_label.config(text=str(self.detection_count))
        )

    def _update_click_count(self) -> None:
        """Update the click count in the GUI."""
        self.root.after(
            0, lambda: self.click_count_label.config(text=str(self.click_count))
        )

    def _log_message(self, message: str) -> None:
        """Add a message to the log display."""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"

        def update_log() -> None:
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, formatted_message)
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)

        self.root.after(0, update_log)

    def _clear_log(self) -> None:
        """Clear the log display."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _setup_logging_handler(self) -> None:
        """Setup logging handler to display logs in the GUI."""

        class GUILogHandler(logging.Handler):
            def __init__(self, gui_instance: OKButtonDetectorGUI) -> None:
                super().__init__()
                self.gui = gui_instance

            def emit(self, record: logging.LogRecord) -> None:
                message = self.format(record)
                self.gui._log_message(message)

        # Add GUI handler to root logger
        gui_handler = GUILogHandler(self)
        gui_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        gui_handler.setFormatter(formatter)

        # Get the root logger and add our handler
        root_logger = logging.getLogger()
        root_logger.addHandler(gui_handler)
