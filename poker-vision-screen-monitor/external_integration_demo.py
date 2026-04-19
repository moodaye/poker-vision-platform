#!/usr/bin/env python3
"""
External Integration Demo for Screen Monitor
Demonstrates how external systems can:
1. Feed images into the screen monitor
2. Retrieve images from the screen monitor
3. Stream real-time data from the monitor
"""

import base64
import time
from io import BytesIO

import requests
from PIL import Image, ImageDraw, ImageFont

BASE_URL = "http://localhost:5000"


def create_sample_image(text="Sample Image", color=(100, 150, 200)):
    """Create a sample image for demonstration"""
    img = Image.new("RGB", (400, 300), color=color)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Draw text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    x = (400 - text_width) // 2
    y = (300 - text_height) // 2

    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    # Add timestamp
    timestamp = time.strftime("%H:%M:%S")
    draw.text((10, 10), timestamp, fill=(255, 255, 255), font=font)

    return img


def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def demo_feed_image():
    """Demonstrate feeding an external image to the monitor"""
    print("=== Demo: Feeding External Image ===")

    # Create a sample image
    sample_img = create_sample_image("External System Image", (200, 100, 50))
    img_base64 = image_to_base64(sample_img)

    # Feed the image to the monitor
    response = requests.post(f"{BASE_URL}/api/image/feed", json={"image": img_base64})

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Image fed successfully: {result['message']}")
        print(f"  Image size: {result['size']}")
    else:
        print(f"✗ Failed to feed image: {response.text}")


def demo_retrieve_image():
    """Demonstrate retrieving the latest image from monitor"""
    print("\n=== Demo: Retrieving Latest Image ===")

    # Get latest image as JSON
    response = requests.get(f"{BASE_URL}/api/image/latest")

    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            print("✓ Retrieved image successfully")
            print(f"  Size: {result['size']}")
            print(f"  Timestamp: {result['timestamp']}")
            # Image data is in result['image'] as base64
        else:
            print(f"✗ No image available: {result['message']}")
    else:
        print(f"✗ Failed to retrieve image: {response.text}")


def demo_raw_image():
    """Demonstrate getting raw image bytes"""
    print("\n=== Demo: Getting Raw Image ===")

    response = requests.get(f"{BASE_URL}/api/image/raw")

    if response.status_code == 200:
        print("✓ Retrieved raw image successfully")
        print(f"  Content-Type: {response.headers.get('Content-Type')}")
        print(f"  Size: {len(response.content)} bytes")

        # Save the image
        with open("retrieved_image.jpg", "wb") as f:
            f.write(response.content)
        print("  Saved as: retrieved_image.jpg")
    else:
        print(f"✗ Failed to retrieve raw image: {response.text}")


def demo_status_monitoring():
    """Demonstrate monitoring system status"""
    print("\n=== Demo: Status Monitoring ===")

    response = requests.get(f"{BASE_URL}/api/status")

    if response.status_code == 200:
        status = response.json()
        print("✓ System Status:")
        print(f"  Capturing: {status['is_capturing']}")
        print(f"  Total captures: {status['stats']['total_captures']}")
        print(f"  Failed captures: {status['stats']['failed_captures']}")
        print(f"  Quality: {status['config']['quality']}%")
        print(f"  Interval: {status['config']['interval']}s")
    else:
        print(f"✗ Failed to get status: {response.text}")


def demo_start_capture():
    """Demonstrate starting capture with custom config"""
    print("\n=== Demo: Starting Capture ===")

    config = {"interval": 3.0, "quality": 95}

    response = requests.post(f"{BASE_URL}/api/start", json=config)

    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            print(f"✓ Capture started: {result['message']}")
        else:
            print(f"✗ Failed to start: {result['message']}")
    else:
        print(f"✗ Failed to start capture: {response.text}")


def main():
    """Run all demonstrations"""
    print("Screen Monitor - External Integration Demo")
    print("=========================================")

    try:
        # Check if the monitor is running
        response = requests.get(f"{BASE_URL}/api/status", timeout=5)
        if response.status_code != 200:
            print("✗ Screen Monitor is not running. Start the application first.")
            return

        print("✓ Screen Monitor is running\n")

        # Run demonstrations
        demo_status_monitoring()
        demo_start_capture()
        time.sleep(2)  # Wait for capture to start
        demo_feed_image()
        time.sleep(1)
        demo_retrieve_image()
        demo_raw_image()

        print("\n=== Integration Examples ===")
        print("1. Feed images: POST /api/image/feed with base64 image data")
        print("2. Get latest image: GET /api/image/latest (JSON with base64)")
        print("3. Get raw image: GET /api/image/raw (JPEG bytes)")
        print("4. Stream images: GET /api/image/stream (multipart stream)")
        print("5. Monitor status: GET /api/status")
        print("6. Control capture: POST /api/start, POST /api/stop")

    except requests.ConnectionError:
        print(
            "✗ Cannot connect to Screen Monitor. Make sure it's running on port 5000."
        )
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    main()
