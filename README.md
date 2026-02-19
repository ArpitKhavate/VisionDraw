# VisionDraw (Air Canvas) 

A Python-based desktop application that allows you to draw in digital space using hand gestures captured via webcam. Create amazing artwork using just your index finger!

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.9.0-green.svg)
![MediaPipe](https://img.shields.io/badge/mediapipe-0.10.9-orange.svg)

## 🎯 Features

### Core Functionality
- **Real-time Hand Tracking**: Tracks 21 hand landmarks using Google's MediaPipe for precise gesture recognition
- **Intuitive Gesture Controls**:
  - **Drawing Mode**: Raise only your index finger to draw
  - **Hover/Selection Mode**: Raise both index and middle fingers to navigate without drawing
- **Color Palette**: Six vibrant colors to choose from:
  - Red
  - Blue
  - Green
  - Yellow
  - White
  - Clear (eraser mode)
- **Canvas Management**:
  - Clear All button to reset your canvas
  - Save drawings as PNG files with timestamps
- **Mirrored Display**: Natural mirror-like webcam feed for intuitive control
- **Visual Feedback**:
  - Dynamic cursor that changes based on mode
  - Progress bars for button selection
  - FPS counter for performance monitoring
  - Real-time mode indicators

## 🛠️ Technical Stack

- **Language**: Python 3.10+
- **Computer Vision**: OpenCV (opencv-python)
- **Hand Tracking**: MediaPipe
- **Mathematics**: NumPy

## 📋 Prerequisites

- Python 3.10 or higher
- Webcam (built-in or external)
- Windows/Linux/macOS

## 🚀 Installation

1. **Clone or download this repository**

2. **Navigate to the project directory**:
```bash
cd AirCanvas
```

3. **Check system compatibility** (recommended):
```bash
python system_check.py
```

4. **Install required dependencies**:
```bash
pip install -r requirements.txt
```

   Or use the automated setup script:
```bash
python setup_and_run.py
```

## 🎮 Usage

### Starting the Application

Run the main script:
```bash
python air_canvas.py
```

### Controls & Gestures

#### Hand Gestures:
- **Drawing Mode** 🎨
  - Raise **only your index finger**
  - Move your finger to draw on the canvas
  - The cursor will be filled and colored

- **Hover/Selection Mode** 👆
  - Raise **both index and middle fingers**
  - Navigate without drawing
  - Select colors and buttons by hovering for 0.5 seconds
  - The cursor will be a hollow yellow circle

#### Color Selection:
1. Switch to Hover Mode (index + middle finger up)
2. Move your hand over the desired color button at the top
3. Hold for 0.5 seconds until the progress bar fills
4. The color will be selected automatically

#### Keyboard Shortcuts:
- **S**: Save your current drawing as PNG
- **Q**: Quit the application

### Tips for Best Experience

1. **Lighting**: Ensure good lighting for better hand detection
2. **Background**: A plain background helps with tracking accuracy
3. **Distance**: Position yourself 1-2 feet from the webcam
4. **Steady Hand**: Keep your hand steady for cleaner lines
5. **Single Hand**: Use only one hand for best results

## 🎨 Interface Overview

```
┌──────────────────────────────────────────────────────────┐
│ [Red] [Blue] [Green] [Yellow] [White] [Clear]   MODE    │
│                                                           │
│                                                           │
│                    Drawing Area                          │
│                                                           │
│                                                           │
│ Instructions:                                    FPS: 30 │
│ - Index finger: Draw                                     │
│ - Index + Middle: Hover/Select                          │
│ - Hover over buttons to select                          │
│ - Press 'S' to save | 'Q' to quit                       │
└──────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
AirCanvas/
│
├── air_canvas.py          # Main application file
├── requirements.txt       # Python dependencies
├── setup_and_run.py       # Easy setup and launch script
├── system_check.py        # System compatibility checker
├── config_example.py      # Configuration template
├── README.md             # Comprehensive documentation
├── QUICKSTART.md         # Quick reference guide
├── PROJECT_SUMMARY.md    # Technical overview
├── .gitignore            # Git ignore rules
└── air_canvas_*.png      # Saved drawings (generated)
```

## 🔧 Configuration

You can customize various parameters in `air_canvas.py`:

### Hand Detection Sensitivity:
```python
self.hands = self.mp_hands.Hands(
    min_detection_confidence=0.7,  # Adjust 0.5-0.9
    min_tracking_confidence=0.7    # Adjust 0.5-0.9
)
```

### Brush Settings:
```python
self.brush_thickness = 5  # Change brush size (1-15)
```

### Webcam Resolution:
```python
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

### Button Hover Duration:
```python
hover_threshold = 0.5  # Seconds to hover (0.3-1.0)
```

## 🐛 Troubleshooting

### Camera Not Opening
- Check if another application is using the webcam
- Try changing camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`

### Hand Not Detected
- Ensure adequate lighting
- Move closer to the camera
- Try adjusting `min_detection_confidence` to a lower value (e.g., 0.5)

### Laggy Performance
- Reduce webcam resolution
- Close other resource-intensive applications
- Lower MediaPipe confidence thresholds

### Drawing Lines Appear Choppy
- Ensure good lighting
- Keep hand movements smooth and steady
- Increase `min_tracking_confidence` for more stable tracking

## 🔄 How It Works

1. **Capture**: OpenCV captures frames from your webcam
2. **Detection**: MediaPipe detects and tracks 21 hand landmarks in real-time
3. **Gesture Recognition**: Algorithm analyzes finger positions to determine mode:
   - Compares Y-coordinates of fingertips vs PIP joints
   - Only index up = Drawing mode
   - Index + Middle up = Hover mode
4. **Drawing**: Index finger position tracked and drawn on transparent canvas
5. **Rendering**: Canvas overlay merged with webcam feed for final display

## 🎓 Learning Resources

- [MediaPipe Hands Documentation](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [NumPy Documentation](https://numpy.org/doc/)

## 🤝 Contributing

Suggestions and improvements are welcome! Some ideas for enhancement:
- Multiple brush sizes
- Undo/Redo functionality
- Shape drawing tools (circles, rectangles)
- Recording drawing sessions as videos
- Multi-hand support
- Custom background images

## 📝 License

This project is open-source and available for educational purposes.

## 👨‍💻 Author

Created with ❤️ using Python, OpenCV, and MediaPipe

## 🙏 Acknowledgments

- Google MediaPipe team for the amazing hand tracking solution
- OpenCV community for computer vision tools
- All contributors and testers

---

**Enjoy creating amazing air drawings! 🎨✨**

For issues or questions, please check the troubleshooting section or refer to the official documentation of the libraries used.
