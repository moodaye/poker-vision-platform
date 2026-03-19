import base64
import logging
import os
import time
from io import BytesIO

from flask import Flask, Response, jsonify, render_template, request
from PIL import Image
from screen_capture import ScreenCaptureService

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Initialize screen capture service
capture_service = ScreenCaptureService()


@app.route("/")
def index():
    """Main dashboard page"""
    return render_template(
        "index.html",
        is_capturing=capture_service.is_capturing(),
        config=capture_service.get_config(),
    )


@app.route("/api/start", methods=["POST"])
def start_capture():
    """Start screen capture"""
    try:
        data = request.get_json() or {}
        interval = data.get("interval", 1.0)
        quality = data.get("quality", 85)

        if data:
            config_ok = capture_service.update_config(data)
            if not config_ok:
                return jsonify(
                    {"success": False, "message": "Invalid configuration values"}
                ), 400

        success = capture_service.start_capture(interval=interval, quality=quality)

        if success:
            logger.info(
                f"Screen capture started with interval {interval}s, quality {quality}"
            )
            return jsonify(
                {
                    "success": True,
                    "message": "Screen capture started successfully",
                    "config": capture_service.get_config(),
                }
            )
        else:
            return jsonify(
                {"success": False, "message": "Failed to start screen capture"}
            ), 500

    except Exception as e:
        logger.error(f"Error starting capture: {str(e)}")
        return jsonify(
            {"success": False, "message": f"Error starting capture: {str(e)}"}
        ), 500


@app.route("/api/stop", methods=["POST"])
def stop_capture():
    """Stop screen capture"""
    try:
        capture_service.stop_capture()
        logger.info("Screen capture stopped")
        return jsonify(
            {"success": True, "message": "Screen capture stopped successfully"}
        )
    except Exception as e:
        logger.error(f"Error stopping capture: {str(e)}")
        return jsonify(
            {"success": False, "message": f"Error stopping capture: {str(e)}"}
        ), 500


@app.route("/api/status")
def get_status():
    """Get current capture status"""
    try:
        status = {
            "is_capturing": capture_service.is_capturing(),
            "config": capture_service.get_config(),
            "stats": capture_service.get_stats(),
            "last_error": capture_service.get_last_error(),
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({"error": f"Error getting status: {str(e)}"}), 500


@app.route("/api/image/latest")
def get_latest_image():
    """Get the latest captured image"""
    try:
        image_data = capture_service.get_latest_image()

        if image_data is None:
            return jsonify({"success": False, "message": "No image available"}), 404

        # Convert PIL Image to base64
        buffer = BytesIO()
        image_data.save(
            buffer,
            format="JPEG",
            quality=capture_service.get_config().get("quality", 85),
        )
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return jsonify(
            {
                "success": True,
                "image": f"data:image/jpeg;base64,{img_str}",
                "timestamp": capture_service.get_last_capture_time(),
                "size": image_data.size,
            }
        )

    except Exception as e:
        logger.error(f"Error getting latest image: {str(e)}")
        return jsonify(
            {"success": False, "message": f"Error getting image: {str(e)}"}
        ), 500


@app.route("/api/image/stream")
def image_stream():
    """Stream images for real-time viewing"""

    def generate():
        while capture_service.is_capturing():
            try:
                image_data = capture_service.get_latest_image()
                if image_data:
                    buffer = BytesIO()
                    image_data.save(
                        buffer,
                        format="JPEG",
                        quality=capture_service.get_config().get("quality", 85),
                    )

                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buffer.getvalue()
                        + b"\r\n"
                    )

                time.sleep(capture_service.get_config().get("interval", 1.0))

            except Exception as e:
                logger.error(f"Error in image stream: {str(e)}")
                break

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/image/feed", methods=["POST"])
def feed_image():
    """Accept external image data to feed into the system"""
    try:
        # Check if image is provided as base64 in JSON
        if request.is_json:
            data = request.get_json()
            if "image" in data:
                # Decode base64 image
                image_data = data["image"]
                if image_data.startswith("data:image"):
                    # Remove data URL prefix
                    image_data = image_data.split(",", 1)[1]

                # Decode base64
                import base64

                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))

                # Process and store the image
                processed_image = capture_service._process_image(image)
                capture_service._latest_image = processed_image
                capture_service._last_capture_time = time.time()

                logger.info(f"External image fed successfully: {processed_image.size}")
                return jsonify(
                    {
                        "success": True,
                        "message": "Image fed successfully",
                        "size": processed_image.size,
                    }
                )

        # Check if image is provided as file upload
        elif "image" in request.files:
            file = request.files["image"]
            if file and file.filename:
                # Open uploaded image
                image = Image.open(file.stream)

                # Process and store the image
                processed_image = capture_service._process_image(image)
                capture_service._latest_image = processed_image
                capture_service._last_capture_time = time.time()

                logger.info(
                    f"External image file fed successfully: {processed_image.size}"
                )
                return jsonify(
                    {
                        "success": True,
                        "message": "Image file fed successfully",
                        "size": processed_image.size,
                    }
                )

        return jsonify(
            {"success": False, "message": "No valid image data provided"}
        ), 400

    except Exception as e:
        logger.error(f"Error feeding image: {str(e)}")
        return jsonify(
            {"success": False, "message": f"Error feeding image: {str(e)}"}
        ), 500


@app.route("/api/image/raw")
def get_raw_image():
    """Get the latest captured image as raw JPEG bytes"""
    try:
        image_data = capture_service.get_latest_image()

        if image_data is None:
            return jsonify({"success": False, "message": "No image available"}), 404

        # Convert PIL Image to JPEG bytes
        buffer = BytesIO()
        image_data.save(
            buffer,
            format="JPEG",
            quality=capture_service.get_config().get("quality", 85),
        )
        buffer.seek(0)

        return Response(
            buffer.getvalue(),
            mimetype="image/jpeg",
            headers={
                "Content-Disposition": f"inline; filename=screenshot_{int(time.time())}.jpg",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    except Exception as e:
        logger.error(f"Error getting raw image: {str(e)}")
        return jsonify(
            {"success": False, "message": f"Error getting image: {str(e)}"}
        ), 500


@app.route("/api/config", methods=["GET", "POST"])
def handle_config():
    """Get or update configuration"""
    if request.method == "GET":
        return jsonify(capture_service.get_config())

    elif request.method == "POST":
        try:
            data = request.get_json()
            success = capture_service.update_config(data)

            if success:
                return jsonify(
                    {
                        "success": True,
                        "message": "Configuration updated successfully",
                        "config": capture_service.get_config(),
                    }
                )
            else:
                return jsonify(
                    {"success": False, "message": "Failed to update configuration"}
                ), 400

        except Exception as e:
            logger.error(f"Error updating config: {str(e)}")
            return jsonify(
                {"success": False, "message": f"Error updating configuration: {str(e)}"}
            ), 500


@app.route("/api/webhooks", methods=["GET", "POST", "DELETE"])
def manage_webhooks():
    """Manage webhook URLs for sending images to external systems"""
    if request.method == "GET":
        # Get all configured webhooks
        return jsonify(
            {
                "success": True,
                "webhooks": capture_service.get_webhook_urls(),
                "external_sending_enabled": capture_service.get_config().get(
                    "send_to_external", False
                ),
                "external_format": capture_service.get_config().get(
                    "external_format", "base64"
                ),
            }
        )

    elif request.method == "POST":
        # Add or configure webhooks
        try:
            data = request.get_json()

            if "add_url" in data:
                # Add a new webhook URL
                url = data["add_url"]
                if capture_service.add_webhook_url(url):
                    return jsonify(
                        {
                            "success": True,
                            "message": f"Webhook URL added: {url}",
                            "webhooks": capture_service.get_webhook_urls(),
                        }
                    )
                else:
                    return jsonify(
                        {"success": False, "message": "URL already exists or invalid"}
                    ), 400

            elif "enable_external" in data:
                # Enable/disable external sending
                enabled = bool(data["enable_external"])
                capture_service.enable_external_sending(enabled)
                return jsonify(
                    {
                        "success": True,
                        "message": f"External sending {'enabled' if enabled else 'disabled'}",
                        "external_sending_enabled": enabled,
                    }
                )

            elif "external_format" in data:
                # Set external format
                format_type = data["external_format"]
                if capture_service.set_external_format(format_type):
                    return jsonify(
                        {
                            "success": True,
                            "message": f"External format set to: {format_type}",
                            "external_format": format_type,
                        }
                    )
                else:
                    return jsonify(
                        {
                            "success": False,
                            "message": 'Invalid format. Use "base64" or "multipart"',
                        }
                    ), 400

            else:
                return jsonify(
                    {"success": False, "message": "Missing required parameter"}
                ), 400

        except Exception as e:
            logger.error(f"Error managing webhooks: {str(e)}")
            return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

    elif request.method == "DELETE":
        # Remove a webhook URL
        try:
            data = request.get_json()
            if "remove_url" in data:
                url = data["remove_url"]
                if capture_service.remove_webhook_url(url):
                    return jsonify(
                        {
                            "success": True,
                            "message": f"Webhook URL removed: {url}",
                            "webhooks": capture_service.get_webhook_urls(),
                        }
                    )
                else:
                    return jsonify({"success": False, "message": "URL not found"}), 404
            else:
                return jsonify(
                    {"success": False, "message": "Missing remove_url parameter"}
                ), 400

        except Exception as e:
            logger.error(f"Error removing webhook: {str(e)}")
            return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500


@app.route("/api/test-webhook", methods=["POST"])
def test_webhook():
    """Test sending to a webhook URL without adding it to configuration"""
    try:
        data = request.get_json()
        url = data.get("url")

        if not url:
            return jsonify({"success": False, "message": "URL parameter required"}), 400

        # Get latest image
        image_data = capture_service.get_latest_image()
        if image_data is None:
            return jsonify(
                {"success": False, "message": "No image available to test with"}
            ), 404

        # Test sending
        try:
            capture_service._send_image_to_url(image_data, url)
            return jsonify(
                {"success": True, "message": f"Successfully sent test image to {url}"}
            )
        except Exception as e:
            return jsonify(
                {"success": False, "message": f"Failed to send to {url}: {str(e)}"}
            ), 400

    except Exception as e:
        logger.error(f"Error testing webhook: {str(e)}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting Screen Monitor application...")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
