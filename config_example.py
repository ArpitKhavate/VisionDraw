"""
VisionDraw Configuration File
Customize your Air Canvas experience by modifying these settings.
"""

# ============================================
# CAMERA SETTINGS
# ============================================

# Camera index (0 = default camera, 1 = external camera, etc.)
CAMERA_INDEX = 0

# Camera resolution (higher = better quality, lower = better performance)
FRAME_WIDTH = 1280  # Options: 640, 1280, 1920
FRAME_HEIGHT = 720  # Options: 480, 720, 1080

# ============================================
# HAND DETECTION SETTINGS
# ============================================

# Detection confidence (0.5-0.9): How confident MediaPipe should be to detect a hand
# Lower = more sensitive (may detect false hands)
# Higher = less sensitive (may miss your hand)
MIN_DETECTION_CONFIDENCE = 0.7

# Tracking confidence (0.5-0.9): How confident to track an already detected hand
# Lower = may lose tracking
# Higher = more stable tracking
MIN_TRACKING_CONFIDENCE = 0.7

# Maximum number of hands to detect (1 recommended for best performance)
MAX_NUM_HANDS = 1

# ============================================
# DRAWING SETTINGS
# ============================================

# Brush thickness in pixels
BRUSH_THICKNESS = 5  # Range: 1-15

# Default starting color (BGR format)
DEFAULT_COLOR = (0, 0, 255)  # Red
DEFAULT_COLOR_NAME = 'Red'

# Available colors (BGR format)
COLORS = {
    'Red': (0, 0, 255),
    'Blue': (255, 0, 0),
    'Green': (0, 255, 0),
    'Yellow': (0, 255, 255),
    'White': (255, 255, 255),
    'Clear': (0, 0, 0)
}

# ============================================
# UI SETTINGS
# ============================================

# Button dimensions
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 60
BUTTON_MARGIN = 20

# Hover duration to select a button (in seconds)
HOVER_THRESHOLD = 0.5  # Range: 0.3-1.0

# Show FPS counter
SHOW_FPS = True

# Show instructions overlay
SHOW_INSTRUCTIONS = True

# Show hand landmarks (skeleton)
SHOW_HAND_LANDMARKS = True

# ============================================
# PERFORMANCE SETTINGS
# ============================================

# Mirror the camera feed (True recommended for natural interaction)
MIRROR_CAMERA = True

# Save format for drawings
SAVE_FORMAT = 'png'  # Options: 'png', 'jpg'

# Save quality for JPG (1-100, only used if SAVE_FORMAT = 'jpg')
SAVE_QUALITY = 95

# ============================================
# ADVANCED SETTINGS
# ============================================

# Gesture detection sensitivity
# These values determine how strict the finger detection is
# Adjust if gestures are not being recognized properly

# Minimum distance ratio between fingertip and PIP joint (as percentage of hand size)
FINGER_UP_THRESHOLD = 0.05  # Default: 0.05 (5% of hand height)

# Cooldown time between button selections (prevents accidental double-clicks)
BUTTON_COOLDOWN = 0.2  # seconds

# Canvas background color (when saved)
CANVAS_BACKGROUND_COLOR = (0, 0, 0)  # Black

# Window name
WINDOW_NAME = 'VisionDraw - Air Canvas'

# ============================================
# KEYBOARD SHORTCUTS
# ============================================

KEY_SAVE = ord('s')
KEY_QUIT = ord('q')
KEY_CLEAR = ord('c')  # Alternative to using the button
KEY_INCREASE_BRUSH = ord('+')  # Increase brush size
KEY_DECREASE_BRUSH = ord('-')  # Decrease brush size

# ============================================
# DEBUG SETTINGS
# ============================================

# Print debug information to console
DEBUG_MODE = False

# Show finger status in console
DEBUG_FINGERS = False

# ============================================
# HOW TO USE THIS FILE
# ============================================

"""
To use these settings:

1. Copy this file as: config.py
2. In air_canvas.py, add at the top:
   import config
3. Replace hardcoded values with config.SETTING_NAME

Example modifications in air_canvas.py:

Instead of:
    self.cap = cv2.VideoCapture(0)
    
Use:
    self.cap = cv2.VideoCapture(config.CAMERA_INDEX)

Instead of:
    self.brush_thickness = 5
    
Use:
    self.brush_thickness = config.BRUSH_THICKNESS
"""

# ============================================
# QUICK PRESETS
# ============================================

# Uncomment one of these presets to quickly configure your setup:

# PRESET: High Performance (for slower computers)
"""
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIN_DETECTION_CONFIDENCE = 0.6
SHOW_HAND_LANDMARKS = False
"""

# PRESET: High Quality (for powerful computers)
"""
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8
"""

# PRESET: High Sensitivity (if hand not detected easily)
"""
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
"""

# PRESET: Artist Mode (thick brush, more colors)
"""
BRUSH_THICKNESS = 10
BUTTON_WIDTH = 120
"""
