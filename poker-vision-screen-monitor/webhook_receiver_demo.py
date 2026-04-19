#!/usr/bin/env python3
"""
Webhook Receiver Demo
Demonstrates how external systems can receive images from the screen monitor.
This script creates a simple HTTP server that acts as a webhook endpoint.
"""

import base64
import os
from datetime import datetime

from flask import Flask, jsonify, request

app = Flask(__name__)

# Directory to save received images
SAVE_DIR = "received_images"
os.makedirs(SAVE_DIR, exist_ok=True)


@app.route("/webhook/base64", methods=["POST"])
def receive_base64_image():
    """Receive images in JSON format with base64 encoding"""
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"success": False, "message": "No image data provided"}), 400

        # Extract image data
        timestamp = data.get("timestamp", datetime.now().isoformat())
        image_data = data["image"]
        size = data.get("size", "unknown")
        source = data.get("source", "unknown")
        quality = data.get("quality", "unknown")

        print(f"📸 Received image from {source}")
        print(f"   Timestamp: {timestamp}")
        print(f"   Size: {size}")
        print(f"   Quality: {quality}%")

        # Remove data URL prefix if present
        if image_data.startswith("data:image"):
            image_data = image_data.split(",", 1)[1]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)

        # Save the image
        filename = f"base64_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(image_bytes)

        print(f"   Saved to: {filepath}")

        return jsonify(
            {
                "success": True,
                "message": "Image received successfully",
                "saved_as": filename,
            }
        )

    except Exception as e:
        print(f"❌ Error receiving base64 image: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/webhook/multipart", methods=["POST"])
def receive_multipart_image():
    """Receive images as multipart form data"""
    try:
        # Check if image file is present
        if "image" not in request.files:
            return jsonify({"success": False, "message": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "message": "No file selected"}), 400

        # Extract metadata
        timestamp = request.form.get("timestamp", datetime.now().isoformat())
        size = request.form.get("size", "unknown")
        source = request.form.get("source", "unknown")
        quality = request.form.get("quality", "unknown")

        print(f"📸 Received multipart image from {source}")
        print(f"   Timestamp: {timestamp}")
        print(f"   Size: {size}")
        print(f"   Quality: {quality}%")
        print(f"   Filename: {file.filename}")

        # Save the image
        filename = f"multipart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)

        file.save(filepath)

        print(f"   Saved to: {filepath}")

        return jsonify(
            {
                "success": True,
                "message": "Image received successfully",
                "saved_as": filename,
            }
        )

    except Exception as e:
        print(f"❌ Error receiving multipart image: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/webhook/status", methods=["GET"])
def webhook_status():
    """Health check endpoint for the webhook receiver"""
    image_count = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")])

    return jsonify(
        {
            "status": "active",
            "message": "Webhook receiver is running",
            "images_received": image_count,
            "save_directory": SAVE_DIR,
            "endpoints": {
                "base64": "/webhook/base64",
                "multipart": "/webhook/multipart",
                "status": "/webhook/status",
            },
        }
    )


@app.route("/", methods=["GET"])
def index():
    """Simple info page"""
    return """
    <html>
    <head><title>Screen Monitor Webhook Receiver</title></head>
    <body>
        <h1>Screen Monitor Webhook Receiver</h1>
        <p>This server is ready to receive screen captures from your screen monitor.</p>

        <h2>Available Endpoints:</h2>
        <ul>
            <li><strong>POST /webhook/base64</strong> - Receive JSON with base64 images</li>
            <li><strong>POST /webhook/multipart</strong> - Receive multipart form images</li>
            <li><strong>GET /webhook/status</strong> - Check receiver status</li>
        </ul>

        <h2>Usage:</h2>
        <p>Configure your screen monitor to send images to:</p>
        <ul>
            <li>Base64 format: <code>http://localhost:8000/webhook/base64</code></li>
            <li>Multipart format: <code>http://localhost:8000/webhook/multipart</code></li>
        </ul>
    </body>
    </html>
    """


if __name__ == "__main__":
    print("🚀 Starting Screen Monitor Webhook Receiver")
    print(f"📁 Images will be saved to: {os.path.abspath(SAVE_DIR)}")
    print("🌐 Available endpoints:")
    print("   • POST http://localhost:8000/webhook/base64 (JSON)")
    print("   • POST http://localhost:8000/webhook/multipart (Form)")
    print("   • GET  http://localhost:8000/webhook/status")
    print("\nAdd these URLs to your screen monitor webhook configuration!")

    app.run(host="0.0.0.0", port=8000, debug=True)
