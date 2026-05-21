#!/usr/bin/env python3
"""
Installation script for OK Button Auto-Clicker
Automatically installs required dependencies and verifies setup.
"""

import subprocess
import sys
import platform
import importlib

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_package(package_name):
    """Install a Python package using pip."""
    try:
        print(f"Installing {package_name}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package_name
        ], capture_output=True, text=True, check=True)
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e.stderr}")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is available")
        return True
    except ImportError:
        print(f"❌ {package_name} not found")
        return False

def install_dependencies():
    """Install all required dependencies."""
    dependencies = [
        ("opencv-python", "cv2"),
        ("pyautogui", "pyautogui"),
        ("Pillow", "PIL"),
        ("numpy", "numpy")
    ]
    
    all_installed = True
    for package, import_name in dependencies:
        if not check_package(package, import_name):
            if not install_package(package):
                all_installed = False
    
    return all_installed

def check_system_requirements():
    """Check system-specific requirements."""
    system = platform.system()
    print(f"Operating System: {system}")
    
    if system == "Linux":
        print("📋 Linux detected. You may need to install additional packages:")
        print("   sudo apt-get install scrot python3-tk python3-dev")
        print("   (or equivalent for your distribution)")
    elif system == "Darwin":  # macOS
        print("📋 macOS detected. You may need to grant accessibility permissions.")
        print("   Go to: System Preferences > Security & Privacy > Privacy > Accessibility")
    elif system == "Windows":
        print("📋 Windows detected. No additional setup required.")
    
    return True

def test_installation():
    """Test if all components can be imported."""
    print("\n🧪 Testing installation...")
    
    test_imports = [
        ("cv2", "OpenCV"),
        ("pyautogui", "PyAutoGUI"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("tkinter", "Tkinter (GUI)")
    ]
    
    all_working = True
    for module, name in test_imports:
        try:
            importlib.import_module(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            all_working = False
    
    return all_working

def main():
    """Main installation process."""
    print("🚀 OK Button Auto-Clicker Installation")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system requirements
    check_system_requirements()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("\n❌ Some packages failed to install. Check the errors above.")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("\n❌ Installation test failed. Some modules couldn't be imported.")
        sys.exit(1)
    
    print("\n🎉 Installation completed successfully!")
    print("\nTo run the application:")
    print("   python main.py")
    print("\nTo test with a sample dialog:")
    print("   python test_dialog.py")
    print("\nSee SETUP.md for detailed usage instructions.")

if __name__ == "__main__":
    main()