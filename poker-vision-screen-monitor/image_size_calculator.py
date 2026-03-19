#!/usr/bin/env python3
"""
Image Size Calculator
Calculate payload sizes for different image settings
"""

import base64
import json
from io import BytesIO

from PIL import Image


def create_test_image(width, height):
    """Create a test image with given dimensions"""
    return Image.new("RGB", (width, height), color=(100, 150, 200))


def calculate_payload_size(image, quality):
    """Calculate the payload size for a given image and quality"""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    img_str = base64.b64encode(buffer.getvalue()).decode()

    payload = {
        "image": img_str,
        "format": "jpeg",
        "metadata": {"source": "ScreenStream"},
    }

    payload_size = len(json.dumps(payload))
    base64_size = len(img_str)
    jpeg_size = len(buffer.getvalue())

    return {
        "payload_size": payload_size,
        "base64_size": base64_size,
        "jpeg_size": jpeg_size,
        "image_size": image.size,
    }


def format_size(size_bytes):
    """Format size in bytes to human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def main():
    """Main calculation function"""
    print("📐 IMAGE SIZE CALCULATOR FOR WEBHOOKS")
    print("=" * 50)

    # Common screen resolutions
    resolutions = [
        (1920, 1080, "Full HD"),
        (1280, 720, "HD"),
        (800, 600, "SVGA"),
        (640, 480, "VGA"),
    ]

    # Quality levels
    qualities = [85, 70, 60, 50, 30]

    print(
        f"{'Resolution':<15} {'Quality':<8} {'JPEG Size':<12} {'Base64 Size':<14} {'Payload Size':<14} {'Status':<10}"
    )
    print("-" * 80)

    for width, height, name in resolutions:
        test_image = create_test_image(width, height)

        for quality in qualities:
            result = calculate_payload_size(test_image, quality)

            # Determine status based on common limits
            payload_size = result["payload_size"]
            if payload_size < 1024 * 1024:  # < 1MB
                status = "✅ Good"
            elif payload_size < 5 * 1024 * 1024:  # < 5MB
                status = "⚠️  Large"
            else:
                status = "❌ Too Big"

            print(
                f"{name:<15} {quality:<8} {format_size(result['jpeg_size']):<12} "
                f"{format_size(result['base64_size']):<14} {format_size(payload_size):<14} {status:<10}"
            )

        print()  # Empty line between resolutions

    print("\n💡 RECOMMENDATIONS:")
    print("✅ Good: < 1MB payload (most servers accept)")
    print("⚠️  Large: 1-5MB payload (some servers may reject)")
    print("❌ Too Big: > 5MB payload (most servers will reject with 413)")

    print("\n🎯 OPTIMAL SETTINGS FOR WEBHOOKS:")
    print("• Resolution: 1280x720 or smaller")
    print("• Quality: 50-60%")
    print("• Expected payload: ~200-500KB")

    print("\n🔧 TO FIX 413 ERRORS:")
    print("1. Reduce image quality to 50-60%")
    print("2. Limit image size to 1280x720 or smaller")
    print("3. Consider using multipart format instead of base64")


if __name__ == "__main__":
    main()
