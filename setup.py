import os
import sys
import subprocess
from pathlib import Path


def create_directories():
    # Create required directories for the project
    directories = [
        'logs',
        'alerts',
        'detection',
        'dashboard',
        'utils',
        'tests'
    ]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")


def install_requirements():
    # Install Python requirements from requirements.txt
    print("Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True


def download_models():
    # Download and initialize AI models
    print("Downloading AI models...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("YOLOv8 model downloaded successfully")
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        print("MediaPipe initialized successfully")
    except ImportError as e:
        print(f"Error importing required libraries: {e}")
        return False
    except Exception as e:
        print(f"Error downloading models: {e}")
        return False
    return True


def test_camera():
    # Test camera access
    print("Testing camera access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Warning: Default camera (index 0) not accessible")
            print("This might be normal if no camera is connected")
        else:
            print("Camera access test passed")
            cap.release()
    except Exception as e:
        print(f"Error testing camera: {e}")
        return False
    return True


def create_sample_config():
    # Create a sample configuration file
    print("Creating sample configuration...")
    config_content = """# Smart Campus Surveillance Configuration

# Detection Settings
AGGRESSION_THRESHOLD = 0.7
CROWD_THRESHOLD = 5
WEAPON_CONFIDENCE = 0.5

# Camera Settings
DEFAULT_CAMERA_ID = "cam_001"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Logging Settings
LOG_RETENTION_DAYS = 30
SCREENSHOT_QUALITY = 95

# Dashboard Settings
AUTO_REFRESH = True
REFRESH_INTERVAL = 5
"""
    with open('config.py', 'w') as f:
        f.write(config_content)
    print("Configuration file created: config.py")


def run_initial_test():
    # Run initial system test to check components
    print("Running initial system test...")
    try:
        from detection.yolo_pose_detect import SmartSurveillanceDetector
        from utils.log_event import EventLogger
        detector = SmartSurveillanceDetector()
        logger = EventLogger()
        print("All components initialized successfully")
        print("System test passed!")
    except Exception as e:
        print(f"System test failed: {e}")
        return False
    return True


def main():
    # Main setup function
    print("=" * 60)
    print("SMART CAMPUS SURVEILLANCE - SETUP")
    print("=" * 60)
    success = True
    print("\nStep 1: Creating directories...")
    create_directories()
    print("\nStep 2: Installing requirements...")
    if not install_requirements():
        success = False
    print("\nStep 3: Downloading AI models...")
    if not download_models():
        success = False
    print("\nStep 4: Testing camera access...")
    test_camera()
    print("\nStep 5: Creating configuration...")
    create_sample_config()
    print("\nStep 6: Running system test...")
    if not run_initial_test():
        success = False
    print("\n" + "=" * 60)
    if success:
        print("SETUP COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Run webcam surveillance: python main.py --mode webcam")
        print("2. Launch dashboard: python main.py --mode dashboard")
        print("3. Process video file: python main.py --mode video --source your_video.mp4")
    else:
        print("SETUP COMPLETED WITH ERRORS")
        print("Please check the error messages above and resolve issues")
    print("=" * 60)


if __name__ == "__main__":
    main()