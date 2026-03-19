#!/usr/bin/env python3
"""Test script to verify screen capture functionality"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time

from screen_capture import ScreenCaptureService


def test_screen_capture():
    print("Testing Screen Capture Service...")

    # Create service
    service = ScreenCaptureService()

    # Check configuration
    config = service.get_config()
    print(f"Configuration: {config}")

    # Start capture
    print("Starting capture...")
    if service.start_capture():
        print("Capture started successfully!")

        # Wait a moment for first capture
        time.sleep(2)

        # Check if we got a real image
        image = service.get_latest_image()
        if image:
            print(f"Latest image captured: {image.size} pixels")
            print(f"Last capture time: {service.get_last_capture_time()}")

            # Save test image
            image.save("test_capture.jpg", "JPEG", optimize=True, quality=85)
            print("Test image saved as 'test_capture.jpg'")

        else:
            print("No image captured yet")

        # Get stats
        stats = service.get_stats()
        print(f"Capture stats: {stats}")

        # Stop capture
        service.stop_capture()
        print("Capture stopped")

    else:
        print("Failed to start capture")
        error = service.get_last_error()
        if error:
            print(f"Error: {error}")


if __name__ == "__main__":
    test_screen_capture()
