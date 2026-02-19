"""
VisionDraw (Air Canvas)
A Python-based desktop application for drawing in digital space using hand gestures.
Uses the MediaPipe Tasks API (compatible with mediapipe 0.10.14+).
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import urllib.request

# ── Landmark indices (same as the legacy HandLandmark enum) ──────────────
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Hand skeleton connections for drawing
HAND_CONNECTIONS = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (MIDDLE_MCP, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (RING_MCP, PINKY_MCP), (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
]

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def _ensure_model():
    """Download the hand-landmarker model if it is not already present."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading hand_landmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


class AirCanvas:
    def __init__(self):
        """Initialize the Air Canvas application."""
        # Make sure the model file exists
        _ensure_model()

        # ── MediaPipe Tasks hand-landmarker ──────────────────────────────
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # ── Webcam setup ─────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        ret, frame = self.cap.read()
        if ret:
            self.frame_height, self.frame_width = frame.shape[:2]
        else:
            self.frame_height, self.frame_width = 720, 1280

        # ── Canvas setup ─────────────────────────────────────────────────
        self.canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        # ── Drawing state ────────────────────────────────────────────────
        self.drawing_mode = False
        self.prev_x, self.prev_y = None, None

        # ── Colour palette ───────────────────────────────────────────────
        self.colors = {
            "Red": (0, 0, 255),
            "Blue": (255, 0, 0),
            "Green": (0, 255, 0),
            "Yellow": (0, 255, 255),
            "White": (255, 255, 255),
            "Clear": (0, 0, 0),
        }
        self.current_color = (0, 0, 255)       # Default: Red
        self.current_color_name = "Red"

        # ── UI button setup ──────────────────────────────────────────────
        self.button_width = 150
        self.button_height = 60
        self.button_margin = 20
        self.buttons = {}
        self._setup_buttons()

        # ── Brush settings ───────────────────────────────────────────────
        self.brush_thickness = 5

        # ── Performance tracking ─────────────────────────────────────────
        self.prev_time = 0
        self.frame_counter = 0      # monotonic frame id for MediaPipe

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────
    def _setup_buttons(self):
        """Create the top-row colour / clear buttons."""
        x_offset = self.button_margin
        y_offset = self.button_margin

        for color_name, color_rgb in self.colors.items():
            self.buttons[color_name] = {
                "pos": (x_offset, y_offset,
                        x_offset + self.button_width,
                        y_offset + self.button_height),
                "color": color_rgb,
                "name": color_name,
            }
            x_offset += self.button_width + self.button_margin

    # ── Finger detection ─────────────────────────────────────────────────
    @staticmethod
    def _detect_fingers_up(landmarks):
        """
        Given a list of 21 NormalizedLandmark objects return a tuple
        (index_up, middle_up, ring_up, pinky_up, thumb_up).
        """
        # Fingers: tip vs PIP joint – finger is UP when tip.y < pip.y
        index_up  = landmarks[INDEX_TIP].y  < landmarks[INDEX_PIP].y
        middle_up = landmarks[MIDDLE_TIP].y < landmarks[MIDDLE_PIP].y
        ring_up   = landmarks[RING_TIP].y   < landmarks[RING_PIP].y
        pinky_up  = landmarks[PINKY_TIP].y  < landmarks[PINKY_PIP].y

        # Thumb: horizontal check (tip further from palm centre than IP)
        thumb_up = landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x

        return index_up, middle_up, ring_up, pinky_up, thumb_up

    def _get_index_finger_pos(self, landmarks):
        """Return pixel (x, y) of index-finger tip."""
        tip = landmarks[INDEX_TIP]
        return int(tip.x * self.frame_width), int(tip.y * self.frame_height)

    def _check_button_click(self, x, y):
        """Return button name if (x, y) is inside any button, else None."""
        for name, info in self.buttons.items():
            x1, y1, x2, y2 = info["pos"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return name
        return None

    # ── Drawing helpers ──────────────────────────────────────────────────
    @staticmethod
    def _draw_hand_landmarks(frame, landmarks, w, h):
        """Draw hand skeleton on the frame (replaces mp.solutions.drawing_utils)."""
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # Draw connections
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 128), 2)

        # Draw landmarks
        for px, py in pts:
            cv2.circle(frame, (px, py), 4, (255, 0, 128), -1)

    def _draw_ui(self, frame):
        """Render the colour-palette buttons on the frame."""
        for name, info in self.buttons.items():
            x1, y1, x2, y2 = info["pos"]
            color = info["color"]

            # Button fill
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

            # Border – highlight the selected colour
            border_color = (255, 255, 255) if name == self.current_color_name else (100, 100, 100)
            border_thick = 4 if name == self.current_color_name else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thick)

            # Label
            text_color = (0, 0, 0) if name in ("Yellow", "White") else (255, 255, 255)
            cv2.putText(frame, name, (x1 + 10, y1 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    def _draw_cursor(self, frame, x, y, mode="hover"):
        """Show cursor at the fingertip."""
        if mode == "drawing":
            cv2.circle(frame, (x, y), self.brush_thickness + 5, self.current_color, -1)
            cv2.circle(frame, (x, y), self.brush_thickness + 7, (255, 255, 255), 2)
        else:
            cv2.circle(frame, (x, y), 15, (255, 255, 0), 2)
            cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)

    def _draw_info(self, frame):
        """Draw HUD: mode indicator, instructions, FPS."""
        mode_text = "DRAWING MODE" if self.drawing_mode else "HOVER MODE"
        mode_color = (0, 255, 0) if self.drawing_mode else (0, 255, 255)
        cv2.putText(frame, mode_text, (self.frame_width - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        instructions = [
            "Index finger: Draw",
            "Index + Middle: Hover/Select",
            "Hover over buttons to select",
            "Press 'S' to save | 'Q' to quit",
        ]
        y_off = self.frame_height - 120
        for line in instructions:
            cv2.putText(frame, line, (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_off += 25

        # FPS
        now = time.time()
        fps = 1 / (now - self.prev_time + 1e-6)
        self.prev_time = now
        cv2.putText(frame, f"FPS: {int(fps)}",
                    (self.frame_width - 120, self.frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ── Save ─────────────────────────────────────────────────────────────
    def save_canvas(self):
        """Export the drawing canvas as a .png file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"air_canvas_{timestamp}.png"
        cv2.imwrite(filename, self.canvas)
        print(f"Canvas saved as: {filename}")
        return filename

    # ─────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────
    def run(self):
        """Start the application."""
        print("VisionDraw (Air Canvas) – Starting …")
        print("Controls:")
        print("  Index finger only  → Draw")
        print("  Index + Middle     → Hover / Select")
        print("  Hover buttons      → Pick colour")
        print("  S → Save   Q → Quit\n")

        button_hover_timer: dict[str, float] = {}
        hover_threshold = 0.5  # seconds

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break

            # Mirror the image for natural interaction
            frame = cv2.flip(frame, 1)

            # Convert BGR → RGB and wrap for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Detect hand landmarks (VIDEO mode needs a monotonic timestamp)
            self.frame_counter += 1
            timestamp_ms = int(self.frame_counter * (1000 / 30))  # ~30 fps
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]   # first hand

                # Draw skeleton overlay
                self._draw_hand_landmarks(frame, landmarks,
                                          self.frame_width, self.frame_height)

                # Finger analysis
                index_up, middle_up, ring_up, pinky_up, thumb_up = \
                    self._detect_fingers_up(landmarks)

                x, y = self._get_index_finger_pos(landmarks)

                # ── DRAWING MODE: only index finger raised ───────────
                if index_up and not middle_up and not ring_up and not pinky_up:
                    self.drawing_mode = True

                    if self._check_button_click(x, y):
                        # Inside a button – don't draw
                        self._draw_cursor(frame, x, y, "hover")
                        self.prev_x, self.prev_y = None, None
                    else:
                        if self.prev_x is not None and self.prev_y is not None:
                            cv2.line(self.canvas,
                                     (self.prev_x, self.prev_y), (x, y),
                                     self.current_color, self.brush_thickness)
                        self.prev_x, self.prev_y = x, y
                        self._draw_cursor(frame, x, y, "drawing")

                # ── HOVER / SELECT MODE: index + middle raised ───────
                elif index_up and middle_up:
                    self.drawing_mode = False
                    self.prev_x, self.prev_y = None, None
                    self._draw_cursor(frame, x, y, "hover")

                    btn = self._check_button_click(x, y)
                    if btn:
                        if btn not in button_hover_timer:
                            button_hover_timer[btn] = time.time()

                        elapsed = time.time() - button_hover_timer[btn]

                        # Progress bar
                        x1, y1, x2, y2 = self.buttons[btn]["pos"]
                        progress_w = int((x2 - x1) * min(elapsed / hover_threshold, 1.0))
                        cv2.rectangle(frame, (x1, y2 + 5),
                                      (x1 + progress_w, y2 + 15),
                                      (0, 255, 0), -1)

                        if elapsed >= hover_threshold:
                            if btn == "Clear":
                                self.canvas[:] = 0
                                print("Canvas cleared!")
                            else:
                                self.current_color = self.colors[btn]
                                self.current_color_name = btn
                                print(f"Colour → {btn}")
                            button_hover_timer.clear()
                    else:
                        button_hover_timer.clear()

                # ── No recognised gesture ─────────────────────────────
                else:
                    self.drawing_mode = False
                    self.prev_x, self.prev_y = None, None
                    button_hover_timer.clear()
            else:
                # No hand visible
                self.drawing_mode = False
                self.prev_x, self.prev_y = None, None
                button_hover_timer.clear()

            # ── Merge canvas onto the camera feed ─────────────────────
            mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
            frame = cv2.add(frame_bg, canvas_fg)

            # ── UI overlay ────────────────────────────────────────────
            self._draw_ui(frame)
            self._draw_info(frame)

            cv2.imshow("VisionDraw - Air Canvas", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("s"), ord("S")):
                self.save_canvas()

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()
        print("Application closed.")


def main():
    app = AirCanvas()
    app.run()


if __name__ == "__main__":
    main()
