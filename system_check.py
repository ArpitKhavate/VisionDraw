"""
System Check and Test Script for VisionDraw
This script verifies that your system is ready to run Air Canvas
"""

import sys
import subprocess

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_python_version():
    """Check if Python version is adequate"""
    print_section("Python Version Check")
    
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 10:
        print("✅ Python version is compatible (3.10+)")
        return True
    elif version.major == 3 and version.minor >= 7:
        print("⚠️  Python version is acceptable but 3.10+ recommended")
        return True
    else:
        print("❌ Python version too old. Please upgrade to 3.10+")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print_section("Dependency Check")
    
    dependencies = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy'
    }
    
    all_installed = True
    
    for module_name, package_name in dependencies.items():
        try:
            if module_name == 'cv2':
                import cv2
                version = cv2.__version__
            elif module_name == 'mediapipe':
                import mediapipe
                version = mediapipe.__version__
            elif module_name == 'numpy':
                import numpy
                version = numpy.__version__
            
            print(f"✅ {package_name:20s} - Installed (v{version})")
        except ImportError:
            print(f"❌ {package_name:20s} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_camera():
    """Check if camera is accessible"""
    print_section("Camera Check")
    
    try:
        import cv2
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot access camera (index 0)")
            print("   Try checking:")
            print("   - Is camera connected?")
            print("   - Is camera being used by another app?")
            print("   - Try changing CAMERA_INDEX to 1 in the code")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Camera opened but cannot read frames")
            cap.release()
            return False
        
        height, width = frame.shape[:2]
        print(f"✅ Camera accessible")
        print(f"   Resolution: {width}x{height}")
        print(f"   Backend: {cap.getBackendName()}")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Error checking camera: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe hand detection (Tasks API)"""
    print_section("MediaPipe Hand Detection Test")
    
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        import numpy as np
        import os, urllib.request

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "hand_landmarker.task")
        model_url = ("https://storage.googleapis.com/mediapipe-models/"
                     "hand_landmarker/hand_landmarker/float16/latest/"
                     "hand_landmarker.task")

        if not os.path.exists(model_path):
            print("   Downloading hand_landmarker model …")
            urllib.request.urlretrieve(model_url, model_path)

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)
        
        print("✅ MediaPipe HandLandmarker initialised (Tasks API)")
        
        # Test with a blank image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=test_image)
        result = landmarker.detect(mp_image)
        
        print("✅ MediaPipe processing works")
        print("   Note: No hands detected in test image (expected)")
        
        landmarker.close()
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def test_drawing():
    """Test basic drawing functions"""
    print_section("Drawing Functions Test")
    
    try:
        import cv2
        import numpy as np
        
        # Create test canvas
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test drawing
        cv2.circle(canvas, (100, 100), 20, (0, 0, 255), -1)
        cv2.line(canvas, (50, 50), (150, 150), (255, 0, 0), 5)
        cv2.rectangle(canvas, (200, 200), (300, 300), (0, 255, 0), 2)
        
        print("✅ OpenCV drawing functions work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Drawing test failed: {e}")
        return False

def check_file_write_permission():
    """Check if we can write files to current directory"""
    print_section("File Write Permission Check")
    
    try:
        test_file = "test_write_permission.tmp"
        with open(test_file, 'w') as f:
            f.write("test")
        
        import os
        os.remove(test_file)
        
        print("✅ Can write files to current directory")
        print("   Saved drawings will be stored here")
        return True
        
    except Exception as e:
        print(f"❌ Cannot write files: {e}")
        print("   You may not be able to save your drawings")
        return False

def display_system_info():
    """Display system information"""
    print_section("System Information")
    
    import platform
    
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python Implementation: {platform.python_implementation()}")

def install_dependencies():
    """Attempt to install missing dependencies"""
    print_section("Installing Dependencies")
    
    print("Attempting to install required packages...")
    print("This may take a few minutes...\n")
    
    try:
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "-r", 
            "requirements.txt"
        ])
        print("\n✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to install dependencies: {e}")
        return False
    except FileNotFoundError:
        print("\n❌ requirements.txt not found!")
        print("Please ensure you're in the correct directory")
        return False

def main():
    """Main test function"""
    print("\n" + "🎨"*30)
    print("   VisionDraw (Air Canvas) - System Check")
    print("🎨"*30)
    
    all_checks = []
    
    # Run all checks
    all_checks.append(("Python Version", check_python_version()))
    all_checks.append(("Dependencies", check_dependencies()))
    
    # If dependencies missing, offer to install
    if not all_checks[-1][1]:
        print("\n" + "-"*60)
        response = input("Would you like to install missing dependencies? (y/n): ")
        if response.lower() == 'y':
            if install_dependencies():
                all_checks[-1] = ("Dependencies", True)
                print("\nRe-checking dependencies...")
                all_checks[-1] = ("Dependencies", check_dependencies())
    
    # Continue with other checks only if dependencies are installed
    if all_checks[-1][1]:
        all_checks.append(("Camera", check_camera()))
        all_checks.append(("MediaPipe", test_mediapipe()))
        all_checks.append(("Drawing", test_drawing()))
        all_checks.append(("File Write", check_file_write_permission()))
    
    # Display system info
    display_system_info()
    
    # Summary
    print_section("Summary")
    
    passed = sum(1 for _, result in all_checks if result)
    total = len(all_checks)
    
    print(f"\nTests Passed: {passed}/{total}\n")
    
    for check_name, result in all_checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} - {check_name}")
    
    print("\n" + "-"*60)
    
    if passed == total:
        print("\n🎉 All checks passed! Your system is ready to run VisionDraw!")
        print("\nTo start the application, run:")
        print("   python air_canvas.py")
    else:
        print("\n⚠️  Some checks failed. Please resolve the issues above.")
        print("\nCommon solutions:")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Check camera permissions in system settings")
        print("   - Close other apps using the camera")
        print("   - Try a different camera by changing CAMERA_INDEX")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
