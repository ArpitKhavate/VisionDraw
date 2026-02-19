# VisionDraw - Quick Start Guide

## Installation (One-Time Setup)

1. Open terminal/command prompt in this folder
2. Run: `pip install -r requirements.txt`

## Running the Application

**Option 1: Direct Run**
```bash
python air_canvas.py
```

**Option 2: Using Setup Script**
```bash
python setup_and_run.py
```

## Quick Controls

### Drawing
- **Raise ONLY index finger** → Start drawing
- Move your hand to create strokes

### Navigation (No Drawing)
- **Raise index + middle finger** → Hover mode
- Hover over color buttons for 0.5 seconds to select

### Colors Available
- Red, Blue, Green, Yellow, White
- Clear button to erase canvas

### Keyboard
- **S** → Save drawing
- **Q** → Quit app

## Troubleshooting

**Camera not working?**
- Close other apps using camera
- In `air_canvas.py` line 38, change `VideoCapture(0)` to `VideoCapture(1)`

**Hand not detected?**
- Check lighting (needs good light)
- Move closer to camera
- Use plain background

**App slow?**
- Close other programs
- In `air_canvas.py` lines 39-40, reduce resolution to:
  ```python
  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  ```

## Tips
1. Position 1-2 feet from camera
2. Good lighting is essential
3. Use one hand only
4. Keep fingers clearly separated
5. Start with slow movements

**Happy Drawing! 🎨**
