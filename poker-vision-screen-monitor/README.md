# Screen Monitor — Real-time Screen Capture & Webhook Integration

A real-time screen monitoring system that captures computer screen images and automatically sends them to external systems via configurable webhooks. Built with Flask and Python.

In the poker bot pipeline, the Screen Monitor captures the live game table and forwards each screenshot to the orchestrator. The orchestrator returns a decision JSON (`action`, `amount`, `reason`), which the Screen Monitor logs visibly and **speaks aloud** using Windows TTS.

![Screen Monitor Dashboard](https://img.shields.io/badge/Status-Production%20Ready-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-2.0+-blue)

## Features

### 🖥️ Screen Capture
- **Real-time screen monitoring** with configurable capture intervals
- **Multi-monitor support** for systems with multiple displays
- **Image processing** with timestamps, resizing, and quality adjustment
- **Fallback support** for headless environments (generates test images)

### 🔗 Webhook Integration
- **Automatic image sending** to external systems via HTTP webhooks
- **Multiple format support**: JSON (base64) and multipart form data
- **Webhook management**: Add, remove, and test webhook URLs
- **Error handling** with retry logic and comprehensive logging
- **Decision response handling**: logs and speaks the orchestrator's decision via Windows TTS

### 🌐 Web Dashboard
- **Real-time monitoring** with live image display and statistics
- **Configuration controls** for capture settings and webhook management
- **Bootstrap dark theme** with responsive design
- **API endpoints** for programmatic control and integration

## Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd screen-monitor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

4. **Open the dashboard:**
   Visit `http://localhost:5000` in your browser

### Basic Usage

1. **Configure capture settings** (interval, quality, monitor)
2. **Add webhook URLs** where you want to send captured images
3. **Enable external sending** and select format (base64 or multipart)
4. **Start capture** - images will be sent automatically to your webhooks

## Poker Bot Integration

The Screen Monitor is the entry point of the live-play bot pipeline.

### Setup

1. Start the pipeline services from the repo root:
   ```bash
   uv run python manage_services.py start
   ```

2. Open the web dashboard at `http://localhost:5000`

3. Under **Webhook Configuration**, add the orchestrator URL:
   ```
   http://127.0.0.1:5100/decide
   ```

4. Set **External Format** to `multipart` and enable **External Sending**

5. Start capture — each screenshot is POSTed to the orchestrator automatically

### Decision feedback

Each time the orchestrator returns a decision, the Screen Monitor:
- Logs it visibly at INFO level:
  ```
  *** DECISION: CALL 400 — Standard preflop call with suited connectors ***
  ```
- Speaks the action aloud using Windows built-in TTS (`System.Speech.Synthesis.SpeechSynthesizer` via PowerShell 5.1) — e.g. *"call 400"*, *"fold"*, *"raise 900"*
- `watch` and `wait` states are silent

No additional packages are required for TTS. Works on any standard Windows 10/11 machine.

---

## API Documentation

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web dashboard interface |
| `POST` | `/api/start` | Start screen capture |
| `POST` | `/api/stop` | Stop screen capture |
| `GET` | `/api/status` | Get system status and statistics |
| `POST` | `/api/config` | Update capture configuration |

### Image Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/image/latest` | Get latest image as JSON (base64) |
| `GET` | `/api/image/raw` | Get latest image as raw JPEG |
| `GET` | `/api/image/stream` | Real-time image stream |
| `POST` | `/api/image/feed` | Upload external images |

### Webhook Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/webhooks` | List configured webhooks |
| `POST` | `/api/webhooks` | Add webhook or configure settings |
| `DELETE` | `/api/webhooks` | Remove webhook URL |
| `POST` | `/api/test-webhook` | Test webhook URL |

## Configuration

### Environment Variables

- `SESSION_SECRET`: Flask session secret key
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Capture Settings

```json
{
  "interval": 1.0,           // Capture interval in seconds
  "quality": 85,             // JPEG quality (1-100)
  "resize_factor": 1.0,      // Image resize factor
  "add_timestamp": true,     // Add timestamp to images
  "monitor": 0               // Monitor index for multi-monitor
}
```

### Webhook Configuration

```json
{
  "webhook_urls": ["https://example.com/webhook"],
  "send_to_external": true,
  "external_format": "base64"  // "base64" or "multipart"
}
```

## Integration Examples

### Receiving Images (External System)

**Base64 JSON Format:**
```python
@app.route('/webhook', methods=['POST'])
def receive_image():
    data = request.get_json()
    image_data = data['image']  # data:image/jpeg;base64,<base64>
    timestamp = data['timestamp']
    # Process the image...
```

**Multipart Form Format:**
```python
@app.route('/webhook', methods=['POST'])
def receive_image():
    image_file = request.files['image']
    timestamp = request.form['timestamp']
    # Save or process the image...
```

### Programmatic Control

```python
import requests

# Start capture with custom settings
response = requests.post('http://localhost:5000/api/start', json={
    'interval': 5.0,
    'quality': 90
})

# Add webhook URL
response = requests.post('http://localhost:5000/api/webhooks', json={
    'add_url': 'https://your-server.com/webhook'
})

# Enable external sending
response = requests.post('http://localhost:5000/api/webhooks', json={
    'enable_external': True,
    'external_format': 'base64'
})
```

## Demo Scripts

The project includes several demonstration scripts:

- **`webhook_receiver_demo.py`**: Example webhook receiver server
- **`outbound_integration_demo.py`**: Demonstrates webhook configuration
- **`test_with_upload.py`**: Test system with uploaded images

Run these to see the system in action:

```bash
# Start webhook receiver (in separate terminal)
python webhook_receiver_demo.py

# Configure and test webhooks
python outbound_integration_demo.py

# Test with uploaded image
python test_with_upload.py
```

## System Requirements

### For Real Screen Capture
- **Operating System**: Windows, macOS, or Linux with GUI
- **Python**: 3.8 or higher
- **Screen Access**: System permissions for screen capture
- **Display**: Active display/desktop session

### For Headless/Server Deployment
- **No display required** - system generates test images
- **Full webhook functionality** available
- **API endpoints** work normally

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Dashboard │    │  Flask App       │    │ Screen Capture  │
│   (Bootstrap)   │◄──►│  (API Endpoints) │◄──►│ Service         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Webhook        │    │  Image          │
                       │   Integration    │    │  Processing     │
                       └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  External        │
                       │  Systems         │
                       └──────────────────┘
```

## Deployment

### Development
```bash
python main.py
```

### Production (with Gunicorn)
```bash
gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app"]
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Documentation**: Check the `/docs` folder for detailed guides
- **Examples**: See demo scripts for integration examples

## Changelog

### v1.0.0 (2025-07-29)
- Initial release with screen capture and webhook integration
- Web dashboard with real-time monitoring
- Support for base64 JSON and multipart formats
- Comprehensive API endpoints
- Demo scripts and documentation