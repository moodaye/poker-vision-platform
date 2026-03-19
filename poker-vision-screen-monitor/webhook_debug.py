#!/usr/bin/env python3
"""
Webhook Debug Script
Step-by-step debugging to identify why webhooks aren't working
"""

import base64
import os
import sys
import time
from io import BytesIO

import requests
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_debug_image():
    """Create a simple debug image"""
    img = Image.new("RGB", (300, 200), color=(255, 0, 0))
    return img


def test_webhook_manual(webhook_url):
    """Manually test the webhook with a simple payload"""
    print(f"🔍 Manual webhook test to: {webhook_url}")

    try:
        # Create test image
        test_img = create_debug_image()
        buffer = BytesIO()
        test_img.save(buffer, format="JPEG", quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()

        payload = {
            "image": img_str,
            "format": "jpeg",
            "metadata": {"source": "ScreenStream"},
        }

        print(f"📊 Payload size: {len(str(payload))} characters")
        print(f"📊 Base64 image size: {len(img_str)} characters")

        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        print(f"📨 Response status: {response.status_code}")
        print(f"📨 Response headers: {dict(response.headers)}")
        print(f"📨 Response body: {response.text[:200]}...")

        if response.status_code == 200:
            print("✅ Manual webhook test SUCCESSFUL!")
            return True
        else:
            print("❌ Manual webhook test FAILED!")
            return False

    except Exception as e:
        print(f"❌ Manual test error: {e}")
        return False


def debug_screen_capture_service():
    """Debug the screen capture service configuration"""
    print("\n🔧 Debugging ScreenCaptureService...")

    try:
        from screen_capture import ScreenCaptureService

        service = ScreenCaptureService()

        # Check current configuration
        config = service.get_config()
        print(f"📋 Current config: {config}")

        # Check if external sending is enabled
        external_enabled = config.get("send_to_external", False)
        print(f"📤 External sending enabled: {external_enabled}")

        # Check webhook URLs
        webhook_urls = config.get("webhook_urls", [])
        print(f"🔗 Webhook URLs: {webhook_urls}")

        # Check if any images have been captured
        latest_image = service.get_latest_image()
        print(f"🖼️  Latest image available: {latest_image is not None}")

        # Get stats
        stats = service.get_stats()
        print(f"📊 Stats: {stats}")

        # Check for errors
        last_error = service.get_last_error()
        if last_error:
            print(f"⚠️  Last error: {last_error}")

        # Recommendations
        if not external_enabled:
            print("❌ ISSUE: External sending is DISABLED!")
            print("   Fix: service.enable_external_sending(True)")

        if not webhook_urls:
            print("❌ ISSUE: No webhook URLs configured!")
            print("   Fix: service.add_webhook_url('your-webhook-url')")

        if external_enabled and webhook_urls:
            print("✅ Configuration looks correct for webhook sending")

        return service

    except Exception as e:
        print(f"❌ Error debugging service: {e}")
        return None


def test_service_webhook_integration(service, webhook_url=None):
    """Test the service's webhook integration"""
    print("\n🧪 Testing service webhook integration...")

    if not service:
        print("❌ No service available")
        return False

    # Configure webhook if provided
    if webhook_url:
        print(f"🔧 Configuring webhook: {webhook_url}")
        service.add_webhook_url(webhook_url)
        service.enable_external_sending(True)
        service.set_external_format("base64")

    # Feed a test image to trigger webhook
    test_img = create_debug_image()
    print("🎯 Feeding test image to service...")

    success = service.feed_external_image(test_img)
    if success:
        print("✅ Test image fed successfully!")
        print("⏳ Waiting 3 seconds for webhook to be sent...")
        time.sleep(3)

        # Check for any errors
        error = service.get_last_error()
        if error:
            print(f"⚠️  Service error after feeding image: {error}")
        else:
            print("✅ No errors reported by service")

        return True
    else:
        print("❌ Failed to feed test image")
        error = service.get_last_error()
        if error:
            print(f"❌ Error: {error}")
        return False


def main():
    """Main debug function"""
    print("🔍 WEBHOOK DEBUG TOOL")
    print("=" * 50)

    # Get webhook URL
    webhook_url = input(
        "Enter your webhook URL (press Enter to skip manual test): "
    ).strip()

    # Step 1: Manual webhook test
    if webhook_url:
        print("\n" + "=" * 30)
        print("STEP 1: Manual Webhook Test")
        print("=" * 30)
        manual_success = test_webhook_manual(webhook_url)
    else:
        manual_success = None
        print("⏭️  Skipping manual webhook test")

    # Step 2: Debug service configuration
    print("\n" + "=" * 30)
    print("STEP 2: Service Configuration Debug")
    print("=" * 30)
    service = debug_screen_capture_service()

    # Step 3: Test service integration
    if service and webhook_url:
        print("\n" + "=" * 30)
        print("STEP 3: Service Integration Test")
        print("=" * 30)
        integration_success = test_service_webhook_integration(service, webhook_url)
    else:
        integration_success = None
        print("⏭️  Skipping service integration test")

    # Summary
    print("\n" + "=" * 50)
    print("🎯 DEBUG SUMMARY")
    print("=" * 50)

    if manual_success is True:
        print("✅ Manual webhook test: SUCCESS")
    elif manual_success is False:
        print("❌ Manual webhook test: FAILED")
    else:
        print("⏭️  Manual webhook test: SKIPPED")

    if service:
        print("✅ Service configuration: LOADED")
    else:
        print("❌ Service configuration: FAILED")

    if integration_success is True:
        print("✅ Service integration: SUCCESS")
    elif integration_success is False:
        print("❌ Service integration: FAILED")
    else:
        print("⏭️  Service integration: SKIPPED")

    print("\n💡 NEXT STEPS:")
    if manual_success is False:
        print("1. Fix webhook URL or receiving server")
        print("2. Check if receiving server is running")
        print("3. Verify network connectivity")
    elif manual_success is True and integration_success is False:
        print("1. Check service configuration")
        print("2. Enable external sending")
        print("3. Add webhook URLs to service")
    elif manual_success is True and integration_success is True:
        print("1. Start screen capture: python main.py")
        print("2. Enable capture in the web interface")
        print("3. Check server logs for webhook activity")
    else:
        print("1. Provide webhook URL for complete testing")
        print("2. Ensure receiving server is running")


if __name__ == "__main__":
    main()
