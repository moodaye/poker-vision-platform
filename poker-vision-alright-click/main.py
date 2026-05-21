#!/usr/bin/env python3
"""
OK Button Auto-Clicker Application
Main entry point for the desktop application that detects and clicks OK buttons on screen.
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os
import logging
import subprocess

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_display():
    """Setup X11 display for GUI applications."""
    try:
        # Set DISPLAY environment variable
        os.environ['DISPLAY'] = ':0'
        
        # Start Xvfb if not already running
        try:
            subprocess.check_output(['pgrep', 'Xvfb'], stderr=subprocess.DEVNULL)
            print("Xvfb already running")
        except subprocess.CalledProcessError:
            print("Starting Xvfb...")
            subprocess.Popen(['Xvfb', ':0', '-screen', '0', '1024x768x24'], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import time
            time.sleep(2)  # Give Xvfb time to start
            
        # Try to start a window manager for better window handling
        try:
            subprocess.Popen(['fluxbox'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            pass  # Fluxbox not available, continue without it
            
        return True
    except Exception as e:
        print(f"Warning: Could not setup display: {e}")
        return False

def setup_logging():
    """Setup logging configuration for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ok_detector.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main function to start the application."""
    try:
        # Setup display first
        print("Setting up display...")
        setup_display()
        
        # Setup logging
        setup_logging()
        logging.info("Starting OK Button Detector Application")
        
        # Import GUI after display setup to avoid X11 issues
        from gui import OKButtonDetectorGUI
        
        # Create and run the GUI
        root = tk.Tk()
        app = OKButtonDetectorGUI(root)
        
        # Handle window closing
        def on_closing():
            if app.is_running:
                app.stop_detection()
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        print("GUI starting...")
        root.mainloop()
        
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        print(f"Error: {e}")
        try:
            messagebox.showerror("Error", f"Failed to start application: {e}")
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
