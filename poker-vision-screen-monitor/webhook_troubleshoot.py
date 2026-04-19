#!/usr/bin/env python3
"""
Webhook Troubleshooting Script
Tests webhook functionality with detailed logging and error reporting.
"""

import base64
import json
import os
import sys
import time
from datetime import datetime
from io import BytesIO

import requests
from PIL import Image, ImageDraw, ImageFont

# Add the current directory to path to import screen_capture
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_test_image():
    """Create a simple test image for webhook testing"""
    image = Image.new("RGB", (400, 300), color=(100, 150, 200))
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Draw test content
    draw.text((50, 100), "Webhook Test Image", fill=(255, 255, 255), font=font)
    draw.text(
        (50, 130),
        f"Generated: {datetime.now().strftime('%H:%M:%S')}",
        fill=(200, 200, 200),
        font=font,
    )
    draw.rectangle([50, 200, 150, 250], fill=(255, 0, 0), outline=(255, 255, 255))
    draw.ellipse([200, 200, 300, 250], fill=(0, 255, 0), outline=(255, 255, 255))

    return image


def test_webhook_connectivity(url):
    """Test if the webhook URL is reachable"""
    print(f"\n🔗 Testing connectivity to: {url}")

    try:
        # Try a simple GET request first to check if the server is running
        response = requests.get(url, timeout=5)
        print(f"   ✅ Server is reachable (Status: {response.status_code})")
        return True
    except requests.exceptions.ConnectRefused:
        print("   ❌ Connection refused - is the webhook server running?")
        return False
    except requests.exceptions.Timeout:
        print("   ❌ Connection timeout - server may be slow or unreachable")
        return False
    except requests.exceptions.RequestException as e:
        print(f"   ⚠️  Connection test returned: {e}")
        return True  # May still work for POST requests
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False


def test_base64_webhook(url, image):
    """Test sending image as base64 JSON payload"""
    print(f"\n📤 Testing BASE64 webhook to: {url}")

    try:
        # Convert image to base64
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()

        payload = {
            "image": img_str,  # Just the base64 string, no data URL prefix
            "format": "jpeg",  # Image format
            "metadata": {"source": "ScreenStream"},
        }

        print(f"   📊 Payload size: {len(json.dumps(payload))} bytes")
        print(f"   📊 Image size: {image.size}")

        response = requests.post(
            url, json=payload, headers={"Content-Type": "application/json"}, timeout=10
        )

        print(f"   📨 Response status: {response.status_code}")
        print(f"   📨 Response headers: {dict(response.headers)}")

        if response.text:
            print(f"   📨 Response body: {response.text[:200]}...")

        if response.status_code == 200:
            print("   ✅ BASE64 webhook test SUCCESSFUL!")
            return True
        else:
            print(
                f"   ❌ BASE64 webhook test FAILED with status {response.status_code}"
            )
            return False

    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False


def test_multipart_webhook(url, image):
    """Test sending image as multipart form data"""
    print(f"\n📤 Testing MULTIPART webhook to: {url}")

    try:
        # Convert image to bytes
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        files = {"image": ("test_screenshot.jpg", buffer, "image/jpeg")}
        data = {
            "timestamp": datetime.now().isoformat(),
            "size": f"{image.size[0]}x{image.size[1]}",
            "source": "webhook_test",
            "quality": "85",
        }

        print(f"   📊 Image size: {image.size}")
        print(f"   📊 Form data: {data}")

        response = requests.post(url, files=files, data=data, timeout=10)

        print(f"   📨 Response status: {response.status_code}")
        print(f"   📨 Response headers: {dict(response.headers)}")

        if response.text:
            print(f"   📨 Response body: {response.text[:200]}...")

        if response.status_code == 200:
            print("   ✅ MULTIPART webhook test SUCCESSFUL!")
            return True
        else:
            print(
                f"   ❌ MULTIPART webhook test FAILED with status {response.status_code}"
            )
            return False

    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False


def test_screen_capture_service_webhook():
    """Test the actual screen capture service webhook functionality"""
    print("\n🔧 Testing ScreenCaptureService webhook integration...")

    try:
        from screen_capture import ScreenCaptureService

        service = ScreenCaptureService()

        # Create test webhook URL
        test_url = "http://localhost:8000/webhook/base64"

        # Add webhook URL
        service.add_webhook_url(test_url)
        service.enable_external_sending(True)
        service.set_external_format("base64")

        print(f"   📋 Configured webhooks: {service.get_webhook_urls()}")
        print(
            f"   📋 External sending enabled: {service.get_config()['send_to_external']}"
        )
        print(f"   📋 External format: {service.get_config()['external_format']}")

        # Feed a test image to trigger webhook
        test_image = create_test_image()

        print("   🎯 Feeding test image to service...")
        if service.feed_external_image(test_image):
            print("   ✅ Test image fed successfully!")

            # Wait a moment for webhook to be sent
            time.sleep(2)

            return True
        else:
            print("   ❌ Failed to feed test image")
            error = service.get_last_error()
            if error:
                print(f"   ❌ Error: {error}")
            return False

    except ImportError as e:
        print(f"   ❌ Failed to import ScreenCaptureService: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False


def main():
    """Main troubleshooting function"""
    print("🔍 Webhook Troubleshooting Script")
    print("=" * 50)

    # Get webhook URL from user
    webhook_url = input(
        "Enter your webhook URL (or press Enter for default http://localhost:8000/webhook/base64): "
    ).strip()

    if not webhook_url:
        webhook_url = "http://localhost:8000/webhook/base64"

    print(f"\n🎯 Testing webhook URL: {webhook_url}")

    # Create test image
    print("\n🎨 Creating test image...")
    test_image = create_test_image()
    test_image.save("webhook_test_image.jpg", "JPEG", quality=85)
    print("   ✅ Test image saved as 'webhook_test_image.jpg'")

    # Run tests
    connectivity_ok = test_webhook_connectivity(webhook_url)

    if connectivity_ok:
        base64_ok = test_base64_webhook(webhook_url, test_image)

        # Try multipart if base64 failed or if URL suggests multipart
        if not base64_ok or "multipart" in webhook_url.lower():
            multipart_url = webhook_url.replace("/base64", "/multipart")
            multipart_ok = test_multipart_webhook(multipart_url, test_image)
        else:
            multipart_ok = False
    else:
        base64_ok = False
        multipart_ok = False

    # Test actual service integration
    service_ok = test_screen_capture_service_webhook()

    # Summary
    print("\n" + "=" * 50)
    print("📋 TROUBLESHOOTING SUMMARY:")
    print(f"   🔗 Connectivity: {'✅ OK' if connectivity_ok else '❌ FAILED'}")
    print(f"   📤 Base64 Format: {'✅ OK' if base64_ok else '❌ FAILED'}")
    print(f"   📤 Multipart Format: {'✅ OK' if multipart_ok else '❌ FAILED'}")
    print(f"   🔧 Service Integration: {'✅ OK' if service_ok else '❌ FAILED'}")

    if not any([connectivity_ok, base64_ok, multipart_ok]):
        print("\n⚠️  RECOMMENDATIONS:")
        print("   1. Ensure your webhook server is running")
        print("   2. Check the webhook URL is correct")
        print("   3. Verify firewall/network settings")
        print("   4. Try running the webhook_receiver_demo.py first")
    elif not service_ok:
        print("\n⚠️  SERVICE INTEGRATION ISSUE:")
        print("   1. Check screen capture service configuration")
        print("   2. Verify webhook URLs are added correctly")
        print("   3. Ensure external sending is enabled")


if __name__ == "__main__":
    main()
