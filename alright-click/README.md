# OK Button Auto-Clicker

A desktop application that automatically detects and clicks "OK" buttons on your screen using computer vision and mouse automation.

## Quick Start

### 1. Install Dependencies
```bash
python install.py
```
*This script automatically installs all required packages and verifies your setup.*

### 2. Run the Application
```bash
python main.py
```

### 3. Test It
```bash
python test_dialog.py
```

## Features

- 🎯 **Smart Detection**: Uses computer vision to find OK buttons
- 🖱️ **Safe Automation**: Built-in safety features and rate limiting
- ⚙️ **Adjustable Settings**: Customizable confidence threshold
- 📊 **Real-time Logging**: Monitor all detection and click activity
- 🛡️ **Fail-safe Controls**: Emergency stop mechanisms

## How It Works

1. **Start the app** and adjust the confidence threshold if needed
2. **Click "Start Detection"** to begin monitoring your screen
3. **The app automatically finds and clicks OK buttons** when they appear
4. **Click "Stop Detection"** when done

## Requirements

- Python 3.8+
- Windows, macOS, or Linux
- Dependencies: OpenCV, PyAutoGUI, Pillow, NumPy

## Files

| File | Description |
|------|-------------|
| `main.py` | Main application |
| `install.py` | Automated installer |
| `test_dialog.py` | Test dialog creator |
| `SETUP.md` | Detailed setup guide |
| `dependencies.txt` | Required packages |

**Need help?** Check `SETUP.md` for detailed installation and troubleshooting.