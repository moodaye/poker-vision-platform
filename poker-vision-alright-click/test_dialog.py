#!/usr/bin/env python3
"""
Test Dialog - Creates a simple dialog with OK button for testing the detector
"""

import tkinter as tk
from tkinter import messagebox


def create_test_dialog() -> str:
    """Create a test dialog with OK button"""
    root = tk.Tk()
    root.withdraw()  # Hide main window

    # Show a message box with OK button
    result = messagebox.showinfo(
        "Test Dialog", "This is a test dialog with an OK button.\nClick OK to close it."
    )

    root.destroy()
    return result


if __name__ == "__main__":
    print("Creating test dialog with OK button...")
    create_test_dialog()
    print("Dialog closed")
