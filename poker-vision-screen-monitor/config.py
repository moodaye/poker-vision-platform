import os


class Config:
    """Configuration settings for the screen monitor application"""

    # Flask settings
    SECRET_KEY = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
    DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() == "true"

    # Screen capture settings
    DEFAULT_CAPTURE_INTERVAL = float(os.environ.get("CAPTURE_INTERVAL", "1.0"))
    DEFAULT_IMAGE_QUALITY = int(os.environ.get("IMAGE_QUALITY", "85"))
    DEFAULT_RESIZE_FACTOR = float(os.environ.get("RESIZE_FACTOR", "1.0"))

    # API settings
    MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", "1920x1080").split("x")[0])

    # Logging settings
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG")

    @staticmethod
    def get_default_config():
        """Get default configuration dictionary"""
        return {
            "interval": Config.DEFAULT_CAPTURE_INTERVAL,
            "quality": Config.DEFAULT_IMAGE_QUALITY,
            "resize_factor": Config.DEFAULT_RESIZE_FACTOR,
            "add_timestamp": True,
            "monitor": 0,
        }
