#!/usr/bin/env python3
"""
Screen Monitor Configuration Checker
Helps verify webhook configuration and settings
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_screen_capture_config():
    """Check the current screen capture service configuration"""
    print("🔧 Screen Monitor Configuration Check")
    print("=" * 50)

    try:
        from screen_capture import ScreenCaptureService

        service = ScreenCaptureService()
        config = service.get_config()

        print("📋 Current Configuration:")
        print(f"   📊 Capture interval: {config.get('interval', 'Not set')} seconds")
        print(f"   🖼️  Image quality: {config.get('quality', 'Not set')}%")
        print(f"   📏 Resize factor: {config.get('resize_factor', 'Not set')}")
        print(f"   🕒 Add timestamp: {config.get('add_timestamp', 'Not set')}")
        print(f"   🖥️  Monitor index: {config.get('monitor', 'Not set')}")

        print("\n🌐 Webhook Configuration:")
        print(
            f"   📤 External sending enabled: {config.get('send_to_external', False)}"
        )
        print(f"   📨 External format: {config.get('external_format', 'Not set')}")

        webhook_urls = config.get("webhook_urls", [])
        if webhook_urls:
            print(f"   🔗 Configured webhook URLs ({len(webhook_urls)}):")
            for i, url in enumerate(webhook_urls, 1):
                print(f"      {i}. {url}")
        else:
            print("   ⚠️  No webhook URLs configured!")

        print("\n📈 Service Status:")
        print(f"   🔄 Currently capturing: {service.is_capturing()}")

        stats = service.get_stats()
        print(f"   📊 Total captures: {stats.get('total_captures', 0)}")
        print(f"   ❌ Failed captures: {stats.get('failed_captures', 0)}")

        last_error = service.get_last_error()
        if last_error:
            print(f"   ⚠️  Last error: {last_error}")
        else:
            print("   ✅ No recent errors")

        latest_image = service.get_latest_image()
        if latest_image:
            print(f"   🖼️  Latest image: {latest_image.size} pixels")
            print(f"   🕒 Last capture: {service.get_last_capture_time()}")
        else:
            print("   ⚠️  No images captured yet")

        # Configuration recommendations
        print("\n💡 Recommendations:")

        if not config.get("send_to_external", False):
            print("   ⚠️  External sending is disabled - enable it to send webhooks")

        if not webhook_urls:
            print("   ⚠️  No webhook URLs configured - add URLs to receive images")

        if config.get("send_to_external", False) and webhook_urls:
            print("   ✅ Webhook configuration looks good!")

        return True

    except ImportError as e:
        print(f"❌ Failed to import ScreenCaptureService: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking configuration: {e}")
        return False


def interactive_webhook_config():
    """Interactive webhook configuration"""
    print("\n🛠️  Interactive Webhook Configuration")
    print("-" * 40)

    try:
        from screen_capture import ScreenCaptureService

        service = ScreenCaptureService()

        # Add webhook URL
        add_url = input("Add a webhook URL (press Enter to skip): ").strip()
        if add_url:
            if service.add_webhook_url(add_url):
                print(f"✅ Added webhook URL: {add_url}")
            else:
                print(f"⚠️  URL already exists: {add_url}")

        # Enable external sending
        enable = input("Enable external sending? (y/n, default=y): ").strip().lower()
        if enable != "n":
            service.enable_external_sending(True)
            print("✅ External sending enabled")

        # Set format
        format_choice = (
            input("Choose format (base64/multipart, default=base64): ").strip().lower()
        )
        if format_choice == "multipart":
            service.set_external_format("multipart")
            print("✅ Format set to multipart")
        else:
            service.set_external_format("base64")
            print("✅ Format set to base64")

        print("\n📋 Updated Configuration:")
        config = service.get_config()
        print(f"   📤 External sending: {config.get('send_to_external', False)}")
        print(f"   📨 Format: {config.get('external_format', 'base64')}")
        print(f"   🔗 Webhook URLs: {config.get('webhook_urls', [])}")

        return True

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def main():
    """Main function"""
    # Check current configuration
    check_screen_capture_config()

    # Offer interactive configuration
    configure = (
        input("\n🛠️  Would you like to configure webhooks interactively? (y/n): ")
        .strip()
        .lower()
    )
    if configure == "y":
        interactive_webhook_config()

    print("\n🔍 Next Steps:")
    print("1. Run 'python quick_webhook_test.py' to test your webhook URLs")
    print("2. Run 'python webhook_troubleshoot.py' for detailed diagnostics")
    print("3. Start screen capture with 'python main.py'")


if __name__ == "__main__":
    main()
