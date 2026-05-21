"""
Configuration settings for the OK Button Detector application.
"""

# Detection settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.8
MIN_CONFIDENCE_THRESHOLD = 0.5
MAX_CONFIDENCE_THRESHOLD = 0.99

# Screen capture settings
SCREEN_CAPTURE_DELAY = 0.5  # Seconds between screen captures
CLICK_DELAY = 0.1  # Seconds to wait after clicking

# Safety settings
MAX_CLICKS_PER_MINUTE = 60  # Prevent excessive clicking
CLICK_COOLDOWN = 1.0  # Minimum seconds between clicks

# GUI settings
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 300
WINDOW_TITLE = "OK Button Auto-Clicker"

# Template matching settings
TEMPLATE_MATCH_METHOD = 'cv2.TM_CCOEFF_NORMED'

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FILE = 'ok_detector.log'
