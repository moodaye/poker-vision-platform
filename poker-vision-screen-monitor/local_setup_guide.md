# Testing Screen Capture on Your Local Machine

Since Replit doesn't have access to your screen, here's how to test the system with your actual screen:

## Option 1: Download and Run Locally

1. **Download the project files** from Replit to your local machine
2. **Install dependencies:**
   ```bash
   pip install flask pillow mss pyautogui requests
   ```
3. **Run the application:**
   ```bash
   python main.py
   ```
4. **Open browser:** Go to `http://localhost:5000`

## Option 2: Use Image Upload Feature

You can test the webhook system by uploading your own images:

1. Take a screenshot on your computer and save it
2. Use the existing `/api/image/feed` endpoint to upload it
3. Configure webhooks to send the uploaded image to external systems

## Option 3: Remote Screen Capture Setup

If you want to capture your screen and send it to external systems:

1. **Set up the screen monitor on your local machine**
2. **Configure webhooks** to point to external services:
   - Webhook.site for testing: `https://webhook.site/your-unique-id`
   - Your own server endpoints
   - Cloud services that accept webhooks

## Testing the Webhook System Right Now

Even without real screen capture, you can test the webhook functionality:

1. **Start the webhook receiver:**
   ```bash
   python webhook_receiver_demo.py
   ```
   This runs on port 8000 and will receive images.

2. **Configure webhooks in the dashboard:**
   - Add URL: `http://localhost:8000/webhook/base64`
   - Enable external sending
   - Start capture

3. **The system will send test images** to your webhook receiver, demonstrating the full flow.

## Production Setup

For production use with real screen capture:

1. **Deploy on a machine with display access**
2. **Configure proper webhook URLs** pointing to your external systems
3. **Set appropriate capture intervals** and quality settings
4. **Monitor the logs** for successful image transmissions

The system is fully functional - it just needs to run on a machine that can access a display for real screen capture.