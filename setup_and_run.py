# Quick Setup and Run Script
# This script helps you quickly set up and run VisionDraw

import subprocess
import sys
import os

def install_dependencies():
    """Install required packages from requirements.txt"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies.")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import cv2
        import mediapipe
        import numpy
        return True
    except ImportError:
        return False

def main():
    """Main setup and run function"""
    print("=" * 50)
    print("🎨 VisionDraw (Air Canvas) Setup")
    print("=" * 50)
    
    # Check if dependencies are installed
    if not check_dependencies():
        print("\n⚠️  Dependencies not found!")
        response = input("Would you like to install them now? (y/n): ")
        if response.lower() == 'y':
            if not install_dependencies():
                print("\nPlease install dependencies manually:")
                print("pip install -r requirements.txt")
                return
        else:
            print("\nPlease install dependencies before running:")
            print("pip install -r requirements.txt")
            return
    else:
        print("\n✅ All dependencies are installed!")
    
    # Run the application
    print("\n🚀 Starting VisionDraw...")
    print("-" * 50)
    try:
        import air_canvas
        air_canvas.main()
    except Exception as e:
        print(f"\n❌ Error running application: {e}")
        print("\nTry running directly: python air_canvas.py")

if __name__ == "__main__":
    main()
