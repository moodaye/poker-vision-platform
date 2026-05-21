# OK Button Auto-Clicker - Setup Guide

This desktop application detects and automatically clicks OK buttons on your screen using computer vision.

## System Requirements

- Python 3.8 or higher
- Windows, macOS, or Linux
- Administrative privileges (for mouse control)

## Installation Steps

### 1. Download the Project
Download all the project files to a folder on your computer.

### 2. Install Python Dependencies

**Option A: Using pip with dependencies.txt**
```bash
pip install -r dependencies.txt
```

**Option B: Install packages individually**
```bash
pip install opencv-python>=4.8.0
pip install pyautogui>=0.9.54
pip install Pillow>=9.2.0
pip install numpy>=1.24.0
```

### 3. Platform-Specific Setup

**Windows:**
- No additional setup required
- Windows Defender might flag the app - add it to exceptions if needed

**macOS:**
- Grant accessibility permissions when prompted
- Go to System Preferences > Security & Privacy > Privacy > Accessibility
- Add Terminal or your Python interpreter to the list

**Linux:**
- Install additional packages:
```bash
sudo apt-get install scrot python3-tk python3-dev
# OR on Fedora/RHEL:
sudo dnf install scrot tkinter python3-devel
```

## Running the Application

1. Open a terminal/command prompt in the project folder
2. Run: `python main.py`
3. The GUI window will appear

## How to Test

### Test 1: Basic Detection Test
1. Click "Test Detection" button in the app
2. Check the log area - it should show detection results

### Test 2: Real Dialog Test
1. Open any application with an OK button (e.g., Windows Calculator > Help > About)
2. In the OK Button Auto-Clicker app, click "Start Detection"
3. The app should automatically detect and click the OK button
4. Click "Stop Detection" to stop

### Test 3: Custom Dialog
Run the included test dialog:
```bash
python test_dialog.py
```
This creates a test window with an OK button for testing purposes.

## Configuration

- **Confidence Threshold**: Adjust the slider to change detection sensitivity
  - Higher values = more strict detection (fewer false positives)
  - Lower values = more lenient detection (may click wrong buttons)

## Safety Features

- **Rate Limiting**: Maximum 60 clicks per minute
- **Cooldown**: 1-second minimum between clicks
- **Fail-safe**: Move mouse to top-left corner to stop
- **Logging**: All actions are logged for monitoring

## Troubleshooting

### Common Issues:

1. **"No module named 'cv2'"**
   - Run: `pip install opencv-python`

2. **"Permission denied" or mouse control not working**
   - Run as administrator (Windows) or with sudo (Linux)
   - Grant accessibility permissions (macOS)

3. **"X11 error" (Linux)**
   - Make sure you're running in a graphical environment
   - Install: `sudo apt-get install python3-tk`

4. **App doesn't detect buttons**
   - Adjust confidence threshold (try lowering it)
   - Check if buttons are visible and not obscured
   - Some button styles may not be recognized

### Getting Help:
- Check the log area in the app for error messages
- Ensure your screen is not scaled unusually (100% scaling works best)
- Test with simple dialogs first before complex applications

## File Structure

- `main.py` - Main application entry point
- `gui.py` - GUI interface
- `ok_button_detector.py` - Computer vision detection
- `mouse_controller.py` - Mouse automation
- `config.py` - Configuration settings
- `templates/` - Button template patterns
- `test_dialog.py` - Test dialog for verification