#!/usr/bin/env python3
"""
Current State Checker
Quickly check the current state of the running application
"""

import requests


def check_current_state():
    """Check the current state of the application"""
    base_url = "http://localhost:5000"

    print("🔍 CURRENT APPLICATION STATE")
    print("=" * 40)

    try:
        # Check if app is running
        response = requests.get(f"{base_url}/api/status", timeout=3)
        if response.status_code != 200:
            print("❌ Flask app is not responding properly")
            return

        status = response.json()
        print("✅ Flask app is running")
        print(
            f"📷 Screen capture: {'✅ RUNNING' if status.get('capturing') else '❌ STOPPED'}"
        )

    except Exception as e:
        print("❌ Flask app is not running or not accessible")
        print(f"   Error: {e}")
        print("   Start with: python main.py")
        return

    # Check webhook configuration
    try:
        response = requests.get(f"{base_url}/api/webhooks")
        if response.status_code == 200:
            webhook_config = response.json()

            webhooks = webhook_config.get("webhooks", [])
            external_enabled = webhook_config.get("external_sending_enabled", False)
            format_type = webhook_config.get("external_format", "unknown")

            print("\n🔗 WEBHOOK CONFIGURATION:")
            print(f"   📋 Configured URLs: {len(webhooks)}")
            for i, url in enumerate(webhooks, 1):
                print(f"      {i}. {url}")

            print(
                f"   📤 External sending: {'✅ ENABLED' if external_enabled else '❌ DISABLED'}"
            )
            print(f"   📨 Format: {format_type}")

            # Show what needs to be done
            print("\n💡 STATUS SUMMARY:")
            if not webhooks:
                print("   ⚠️  No webhook URLs configured")
                print("      → Add webhook URLs through web interface")

            if not external_enabled:
                print("   ⚠️  External sending is disabled")
                print("      → Enable 'Send to External Systems' toggle")

            if not status.get("capturing"):
                print("   ⚠️  Screen capture is not running")
                print("      → Click 'Start Capture' button")

            if webhooks and external_enabled and status.get("capturing"):
                print("   ✅ Everything looks configured correctly!")
                print("   📊 Recent stats:")
                stats = status.get("stats", {})
                print(f"      Total captures: {stats.get('total_captures', 0)}")
                print(f"      Failed captures: {stats.get('failed_captures', 0)}")

                last_error = status.get("last_error")
                if last_error:
                    print(f"      Last error: {last_error}")

        else:
            print(f"❌ Failed to get webhook config: {response.status_code}")

    except Exception as e:
        print(f"❌ Error checking webhook config: {e}")


if __name__ == "__main__":
    check_current_state()
