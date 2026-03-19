#!/usr/bin/env python3
"""
Test Screen Monitor with Image Upload
Allows you to upload your own screenshot to test the webhook system.
"""

import base64
from pathlib import Path

import requests

BASE_URL = "http://localhost:5000"


def upload_image_and_test_webhooks(image_path):
    """Upload an image and test the webhook system"""

    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return

    print(f"📤 Uploading image: {image_path}")

    try:
        # Method 1: Upload as base64 JSON
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        response = requests.post(
            f"{BASE_URL}/api/image/feed",
            json={
                "image": f"data:image/jpeg;base64,{image_data}",
                "timestamp": "2025-07-29T09:00:00Z",
                "source": "manual_upload",
            },
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Image uploaded successfully: {result.get('message', 'Success')}")

            # Now test webhook sending
            test_webhook_with_uploaded_image()
        else:
            print(f"❌ Upload failed: {response.text}")

    except Exception as e:
        print(f"❌ Error uploading image: {e}")


def test_webhook_with_uploaded_image():
    """Test sending the uploaded image to webhooks"""
    print("\n🔗 Testing webhook with uploaded image...")

    # First, make sure we have webhooks configured
    try:
        response = requests.get(f"{BASE_URL}/api/webhooks")
        result = response.json()

        if not result.get("webhooks"):
            print("⚠️ No webhooks configured. Adding test webhook...")

            # Add a test webhook
            add_response = requests.post(
                f"{BASE_URL}/api/webhooks", json={"add_url": "https://httpbin.org/post"}
            )

            if add_response.json().get("success"):
                print("✅ Test webhook added")

            # Enable external sending
            enable_response = requests.post(
                f"{BASE_URL}/api/webhooks", json={"enable_external": True}
            )

            if enable_response.json().get("success"):
                print("✅ External sending enabled")

        # Test the webhook
        test_response = requests.post(
            f"{BASE_URL}/api/test-webhook", json={"url": "https://httpbin.org/post"}
        )

        if test_response.json().get("success"):
            print("✅ Webhook test successful - your image was sent!")
        else:
            print(f"❌ Webhook test failed: {test_response.json().get('message')}")

    except Exception as e:
        print(f"❌ Error testing webhook: {e}")


def main():
    """Main function to run the test"""
    print("Screen Monitor - Image Upload Test")
    print("==================================")

    # You can replace this with the path to your screenshot
    example_paths = [
        "screenshot.png",
        "screenshot.jpg",
        "test_image.png",
        "Desktop/screenshot.png",
        "Downloads/screenshot.png",
    ]

    print("Looking for image files to upload...")

    found_image = None
    for path in example_paths:
        if Path(path).exists():
            found_image = path
            break

    if found_image:
        upload_image_and_test_webhooks(found_image)
    else:
        print("❌ No image files found in common locations.")
        print("\nTo test:")
        print("1. Take a screenshot and save it as 'screenshot.png' in this directory")
        print("2. Or modify this script to point to your image file")
        print("3. Run this script again")

        # Manual path input
        manual_path = input(
            "\nOr enter the full path to your image file (or press Enter to skip): "
        ).strip()
        if manual_path:
            upload_image_and_test_webhooks(manual_path)


if __name__ == "__main__":
    main()
