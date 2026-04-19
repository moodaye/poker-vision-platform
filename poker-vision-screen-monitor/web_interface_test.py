#!/usr/bin/env python3
"""
Web Interface Simulation Test
Simulates exactly what happens when you use the web interface to add webhooks
"""

import time

import requests


def test_web_interface_flow(webhook_url):
    """Test the exact same flow as the web interface"""
    base_url = "http://localhost:5000"

    print("🌐 Testing Web Interface Webhook Flow")
    print("=" * 50)

    # Step 1: Check current status
    print("\n1️⃣ Checking initial status...")
    response = requests.get(f"{base_url}/api/webhooks")
    if response.status_code == 200:
        initial_state = response.json()
        print(f"   📋 Initial webhooks: {initial_state.get('webhooks', [])}")
        print(
            f"   📤 External sending: {initial_state.get('external_sending_enabled', False)}"
        )
    else:
        print(f"   ❌ Failed to get initial status: {response.status_code}")
        return False

    # Step 2: Add webhook URL (exactly like web interface)
    print(f"\n2️⃣ Adding webhook URL: {webhook_url}")
    response = requests.post(f"{base_url}/api/webhooks", json={"add_url": webhook_url})

    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ {result.get('message', 'Success')}")
        print(f"   📋 Updated webhooks: {result.get('webhooks', [])}")
    else:
        print(f"   ❌ Failed to add webhook: {response.status_code} - {response.text}")
        return False

    # Step 3: Enable external sending (this is the critical step!)
    print("\n3️⃣ Enabling external sending...")
    response = requests.post(f"{base_url}/api/webhooks", json={"enable_external": True})

    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ {result.get('message', 'Success')}")
    else:
        print(
            f"   ❌ Failed to enable external sending: {response.status_code} - {response.text}"
        )
        return False

    # Step 4: Set format to base64
    print("\n4️⃣ Setting format to base64...")
    response = requests.post(
        f"{base_url}/api/webhooks", json={"external_format": "base64"}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ {result.get('message', 'Success')}")
    else:
        print(f"   ❌ Failed to set format: {response.status_code} - {response.text}")
        return False

    # Step 5: Verify final configuration
    print("\n5️⃣ Verifying final configuration...")
    response = requests.get(f"{base_url}/api/webhooks")
    if response.status_code == 200:
        final_state = response.json()
        print(f"   📋 Final webhooks: {final_state.get('webhooks', [])}")
        print(
            f"   📤 External sending: {final_state.get('external_sending_enabled', False)}"
        )
        print(f"   📨 Format: {final_state.get('external_format', 'unknown')}")

        # Check if everything is configured correctly
        webhooks = final_state.get("webhooks", [])
        external_enabled = final_state.get("external_sending_enabled", False)

        if webhook_url in webhooks and external_enabled:
            print("   ✅ Configuration is CORRECT!")
            return True
        else:
            print("   ❌ Configuration is INCOMPLETE!")
            if webhook_url not in webhooks:
                print("      - Webhook URL not in list")
            if not external_enabled:
                print("      - External sending not enabled")
            return False
    else:
        print(f"   ❌ Failed to verify configuration: {response.status_code}")
        return False


def test_capture_status():
    """Check if screen capture is running"""
    base_url = "http://localhost:5000"

    print("\n6️⃣ Checking capture status...")
    response = requests.get(f"{base_url}/api/status")
    if response.status_code == 200:
        status = response.json()
        is_capturing = status.get("capturing", False)
        print(f"   📷 Screen capture running: {is_capturing}")

        if not is_capturing:
            print("   ⚠️  Screen capture is not running!")
            print("      You need to start it for webhooks to work")

            # Offer to start it
            start = input("   🤔 Start screen capture now? (y/n): ").strip().lower()
            if start == "y":
                print("   ▶️  Starting screen capture...")
                response = requests.post(f"{base_url}/api/capture/start")
                if response.status_code == 200:
                    print("   ✅ Screen capture started!")
                    return True
                else:
                    print(f"   ❌ Failed to start capture: {response.status_code}")
                    return False
        else:
            print("   ✅ Screen capture is active!")
            return True
    else:
        print(f"   ❌ Failed to check status: {response.status_code}")
        return False


def monitor_webhook_activity(duration=20):
    """Monitor for webhook activity"""
    print(f"\n7️⃣ Monitoring webhook activity for {duration} seconds...")
    print("   👀 Watch your webhook receiver for incoming requests...")

    base_url = "http://localhost:5000"
    start_time = time.time()
    last_captures = 0

    while time.time() - start_time < duration:
        try:
            response = requests.get(f"{base_url}/api/status")
            if response.status_code == 200:
                status = response.json()
                stats = status.get("stats", {})
                total_captures = stats.get("total_captures", 0)

                if total_captures > last_captures:
                    print(
                        f"   📸 New capture #{total_captures} - webhook should be sent!"
                    )
                    last_captures = total_captures

                # Check for errors
                last_error = status.get("last_error")
                if last_error and "webhook" in last_error.lower():
                    print(f"   ⚠️  Webhook error detected: {last_error}")

            time.sleep(2)
            print(".", end="", flush=True)

        except KeyboardInterrupt:
            print("\n   ⏹️  Monitoring stopped by user")
            break
        except Exception:
            pass

    print("\n   ⏹️  Monitoring complete")


def main():
    """Main test function"""
    print("🔍 WEB INTERFACE WEBHOOK SIMULATION")
    print("This script simulates exactly what the web interface does")
    print("=" * 60)

    # Check if Flask app is running
    try:
        requests.get("http://localhost:5000/api/status", timeout=3)
        print("✅ Flask app is running")
    except Exception:
        print("❌ Flask app is not running!")
        print("   Start it with: python main.py")
        return

    # Get webhook URL
    webhook_url = input("\nEnter your webhook URL: ").strip()
    if not webhook_url:
        print("❌ No webhook URL provided")
        return

    # Run the test
    config_success = test_web_interface_flow(webhook_url)

    if config_success:
        capture_success = test_capture_status()

        if capture_success:
            monitor_webhook_activity(20)

            print("\n" + "=" * 60)
            print("🎯 FINAL DIAGNOSIS")
            print("=" * 60)
            print("✅ Webhook configuration: CORRECT")
            print("✅ Screen capture: RUNNING")
            print("✅ Setup should be working!")
            print("\nIf webhooks still don't work:")
            print("1. Check your webhook receiver logs")
            print("2. Verify the webhook URL is accessible")
            print("3. Check for firewall/network issues")
            print("4. Look at Flask app console for webhook errors")
        else:
            print("\n❌ Screen capture issue - webhook won't work until this is fixed")
    else:
        print("\n❌ Configuration issue - webhook won't work until this is fixed")


if __name__ == "__main__":
    main()
