#!/usr/bin/env python3
"""
Quick Webhook Diagnostic
Simple script to test webhook connectivity and payload format
"""

import base64
import json
from datetime import datetime
from io import BytesIO

import requests
from PIL import Image, ImageDraw


def create_simple_test_image():
    """Create a basic test image"""
    img = Image.new("RGB", (200, 150), color=(0, 128, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 50), "Test Image", fill=(255, 255, 255))
    draw.text((10, 70), datetime.now().strftime("%H:%M:%S"), fill=(255, 255, 255))
    return img


def test_webhook_quick(webhook_url, format_type="base64"):
    """Quick webhook test"""
    print(f"🔍 Testing webhook: {webhook_url}")
    print(f"📊 Format: {format_type}")

    test_image = create_simple_test_image()

    try:
        if format_type == "base64":
            # Base64 JSON format
            buffer = BytesIO()
            test_image.save(buffer, format="JPEG", quality=85)
            img_str = base64.b64encode(buffer.getvalue()).decode()

            payload = {
                "image": img_str,  # Just the base64 string, no data URL prefix
                "format": "jpeg",  # Image format
                "metadata": {"source": "ScreenStream"},
            }

            print(f"📦 Sending JSON payload ({len(json.dumps(payload))} bytes)...")

            response = requests.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

        else:  # multipart
            # Multipart form data
            buffer = BytesIO()
            test_image.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)

            files = {"image": ("test.jpg", buffer, "image/jpeg")}
            data = {
                "timestamp": datetime.now().isoformat(),
                "size": f"{test_image.size[0]}x{test_image.size[1]}",
                "source": "diagnostic_test",
                "quality": "85",
            }

            print("📦 Sending multipart data...")

            response = requests.post(webhook_url, files=files, data=data, timeout=10)

        print(f"📨 Response: {response.status_code}")
        print(f"📨 Headers: {dict(response.headers)}")
        if response.text:
            print(f"📨 Body: {response.text[:100]}...")

        if response.status_code == 200:
            print("✅ SUCCESS! Webhook is working!")
            return True
        else:
            print(f"❌ FAILED! Status code: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Cannot connect to webhook URL")
        print("   - Check if the receiving server is running")
        print("   - Verify the URL is correct")
        return False
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT ERROR: Webhook server didn't respond in time")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


if __name__ == "__main__":
    print("🔧 Quick Webhook Diagnostic Tool")
    print("=" * 40)

    # Get webhook URL
    url = input("Enter webhook URL: ").strip()
    if not url:
        print("No URL provided. Exiting.")
        exit(1)

    # Determine format
    if "multipart" in url.lower():
        format_type = "multipart"
    else:
        format_type = "base64"

    # Run test
    success = test_webhook_quick(url, format_type)

    if not success:
        print("\n🔧 Troubleshooting Tips:")
        print("1. Make sure your webhook server is running")
        print("2. Check firewall settings")
        print("3. Verify the URL endpoint exists")
        print("4. Try testing with a tool like curl or Postman first")
        print("5. Check server logs for error messages")
