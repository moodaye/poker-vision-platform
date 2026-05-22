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

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
