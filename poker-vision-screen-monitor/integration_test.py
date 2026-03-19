#!/usr/bin/env python3
"""
Complete Webhook Integration Test
Tests the entire webhook flow step by step
"""

import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_flask_app_webhooks():
    """Test webhook configuration through Flask app API"""
    print("🌐 Testing Flask App Webhook API...")

    base_url = "http://localhost:5000"

    try:
        # Test if Flask app is running
        response = requests.get(f"{base_url}/api/status", timeout=5)
        print(f"✅ Flask app is running (status: {response.status_code})")
    except requests.exceptions.ConnectionError:
        print("❌ Flask app is not running!")
        print("   Start it with: python main.py")
        return False
    except Exception as e:
        print(f"❌ Error checking Flask app: {e}")
        return False

    # Get current webhook configuration
    try:
        response = requests.get(f"{base_url}/api/webhooks")
        if response.status_code == 200:
            config = response.json()
            print(f"📋 Current webhook config: {json.dumps(config, indent=2)}")

            webhooks = config.get("webhooks", [])
            external_enabled = config.get("external_sending_enabled", False)

            if not webhooks:
                print("⚠️  No webhook URLs configured in Flask app!")

            if not external_enabled:
                print("⚠️  External sending is disabled in Flask app!")

            return config
        else:
            print(f"❌ Failed to get webhook config: {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ Error getting webhook config: {e}")
        return None


def configure_webhook_via_api(webhook_url):
    """Configure webhook through Flask API"""
    print(f"🔧 Configuring webhook via API: {webhook_url}")

    base_url = "http://localhost:5000"

    try:
        # Add webhook URL
        response = requests.post(
            f"{base_url}/api/webhooks", json={"add_url": webhook_url}
        )

        if response.status_code == 200:
            print("✅ Webhook URL added successfully")
        else:
            print(f"⚠️  Add webhook response: {response.status_code} - {response.text}")

        # Enable external sending
        response = requests.post(
            f"{base_url}/api/webhooks", json={"enable_external": True}
        )

        if response.status_code == 200:
            print("✅ External sending enabled successfully")
        else:
            print(
                f"⚠️  Enable external response: {response.status_code} - {response.text}"
            )

        # Set format to base64
        response = requests.post(
            f"{base_url}/api/webhooks", json={"external_format": "base64"}
        )

        if response.status_code == 200:
            print("✅ External format set to base64 successfully")
        else:
            print(f"⚠️  Set format response: {response.status_code} - {response.text}")

        return True

    except Exception as e:
        print(f"❌ Error configuring webhook via API: {e}")
        return False


def test_capture_status():
    """Test if screen capture is running"""
    print("📷 Testing screen capture status...")

    try:
        response = requests.get("http://localhost:5000/api/status")
        if response.status_code == 200:
            status = response.json()
            print(f"📊 Capture status: {json.dumps(status, indent=2)}")

            is_capturing = status.get("capturing", False)
            if not is_capturing:
                print("⚠️  Screen capture is not running!")
                print("   Start it through the web interface or API")
            else:
                print("✅ Screen capture is active")

            return status
        else:
            print(f"❌ Failed to get status: {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ Error getting capture status: {e}")
        return None


def start_screen_capture():
    """Start screen capture via API"""
    print("▶️  Starting screen capture...")

    try:
        response = requests.post("http://localhost:5000/api/capture/start")
        if response.status_code == 200:
            print("✅ Screen capture started successfully")
            return True
        else:
            print(
                f"❌ Failed to start capture: {response.status_code} - {response.text}"
            )
            return False

    except Exception as e:
        print(f"❌ Error starting capture: {e}")
        return False


def monitor_webhooks_for_activity(duration=30):
    """Monitor for webhook activity by checking stats"""
    print(f"👀 Monitoring webhook activity for {duration} seconds...")

    start_time = time.time()
    last_capture_count = 0

    while time.time() - start_time < duration:
        try:
            response = requests.get("http://localhost:5000/api/status")
            if response.status_code == 200:
                status = response.json()
                stats = status.get("stats", {})
                capture_count = stats.get("total_captures", 0)

                if capture_count > last_capture_count:
                    print(f"📸 New capture detected! Total: {capture_count}")
                    last_capture_count = capture_count

                # Check for errors
                last_error = status.get("last_error")
                if last_error:
                    print(f"⚠️  Error detected: {last_error}")

            time.sleep(2)

        except Exception as e:
            print(f"❌ Error monitoring: {e}")
            break

    print(f"⏹️  Monitoring complete. Final capture count: {last_capture_count}")


def main():
    """Main integration test"""
    print("🔍 COMPLETE WEBHOOK INTEGRATION TEST")
    print("=" * 60)

    # Get webhook URL
    webhook_url = input("Enter your webhook URL: ").strip()
    if not webhook_url:
        print("❌ No webhook URL provided. Exiting.")
        return

    print(f"\n🎯 Testing webhook integration with: {webhook_url}")

    # Step 1: Test Flask app
    print("\n" + "=" * 40)
    print("STEP 1: Flask App Webhook API Test")
    print("=" * 40)
    config = test_flask_app_webhooks()
    if not config:
        return

    # Step 2: Configure webhook
    print("\n" + "=" * 40)
    print("STEP 2: Configure Webhook via API")
    print("=" * 40)
    if not configure_webhook_via_api(webhook_url):
        return

    # Step 3: Check capture status
    print("\n" + "=" * 40)
    print("STEP 3: Check Screen Capture Status")
    print("=" * 40)
    status = test_capture_status()

    # Step 4: Start capture if needed
    if status and not status.get("capturing", False):
        print("\n" + "=" * 40)
        print("STEP 4: Start Screen Capture")
        print("=" * 40)
        if not start_screen_capture():
            return

    # Step 5: Monitor for activity
    print("\n" + "=" * 40)
    print("STEP 5: Monitor Webhook Activity")
    print("=" * 40)
    print("📝 Watch your webhook receiver for incoming requests...")
    monitor_webhooks_for_activity(30)

    # Final status check
    print("\n" + "=" * 60)
    print("🎯 FINAL STATUS CHECK")
    print("=" * 60)
    final_config = test_flask_app_webhooks()
    final_status = test_capture_status()

    if final_config and final_status:
        webhooks = final_config.get("webhooks", [])
        external_enabled = final_config.get("external_sending_enabled", False)
        is_capturing = final_status.get("capturing", False)

        print("\n📋 FINAL SUMMARY:")
        print(f"   🔗 Webhooks configured: {len(webhooks)}")
        print(
            f"   📤 External sending: {'✅ ENABLED' if external_enabled else '❌ DISABLED'}"
        )
        print(f"   📷 Screen capture: {'✅ RUNNING' if is_capturing else '❌ STOPPED'}")

        if webhooks and external_enabled and is_capturing:
            print("\n✅ Everything looks configured correctly!")
            print("If webhooks still aren't working:")
            print("1. Check your webhook receiver server logs")
            print("2. Verify the webhook URL is correct")
            print("3. Check firewall/network settings")
            print("4. Look at Flask app logs for error messages")
        else:
            print("\n❌ Configuration issues detected - see above for details")


if __name__ == "__main__":
    main()
