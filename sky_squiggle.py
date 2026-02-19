"""
SkySquiggle – Draw in the air, let AI guess what it is.
Upgraded from VisionDraw (Air Canvas).
Uses MediaPipe Tasks API, Gemini 2.0 Flash (vision), and ElevenLabs TTS.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import math
import random
import threading
import tempfile
import urllib.request

import pygame
from PIL import Image
from google import genai
from elevenlabs import ElevenLabs
from dotenv import load_dotenv

try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False

# ── Load environment variables ───────────────────────────────────────────
# Use explicit path so it works regardless of the working directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(_SCRIPT_DIR, ".env")

# Strip UTF-8 BOM if present (PowerShell's -Encoding UTF8 adds one)
if os.path.exists(_ENV_PATH):
    with open(_ENV_PATH, "rb") as _f:
        _raw = _f.read()
    if _raw.startswith(b"\xef\xbb\xbf"):
        with open(_ENV_PATH, "wb") as _f:
            _f.write(_raw[3:])

load_dotenv(_ENV_PATH)

_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
_ELEVENLABS_KEY = os.getenv("ELEVENLABS_API_KEY", "")

if not _GEMINI_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. "
        "Create a .env file with GEMINI_API_KEY=your_key  (see .env.example)"
    )
if not _ELEVENLABS_KEY:
    raise RuntimeError(
        "ELEVENLABS_API_KEY not found. "
        "Create a .env file with ELEVENLABS_API_KEY=your_key  (see .env.example)"
    )

# ── Landmark indices ─────────────────────────────────────────────────────
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

HAND_CONNECTIONS = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (MIDDLE_MCP, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (RING_MCP, PINKY_MCP), (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
]

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# ── AI constants ─────────────────────────────────────────────────────────
# ElevenLabs voice (Rachel - free pre-made voice)
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # "Rachel" – free pre-made voice
ELEVENLABS_MODEL_ID = "eleven_flash_v2_5"

# Gemini model (using stable v1 API)
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_PROMPT = (
    "You are a witty, slightly sarcastic art critic. "
    "Look at this hand-drawn sketch and guess what it is. "
    "Be playful and keep your answer under 10 words."
)

FILLER_PHRASES = [
    "Hmm, let me see...",
    "Interesting lines you've got there...",
    "Ooh, this is a tricky one...",
    "Let me put on my art-critic glasses...",
    "A masterpiece in the making, perhaps?",
    "Hold on, I'm channelling my inner Picasso...",
    "Squinting at this very carefully...",
    "Give me a sec, my brain is buffering...",
]

GUESS_DISPLAY_SECONDS = 5  # how long the guess banner stays on screen

# ── Cartoon UI constants ──────────────────────────────────────────────────
SHADOW_OFFSET = 6       # hard-shadow pixel offset (right & down)
BORDER_THICK  = 3       # thick black border for cartoon look
CORNER_RADIUS = 15      # rounded-corner radius
BTN_W         = 135     # button width
BTN_H         = 55      # button height
BTN_GAP       = 14      # gap between buttons
BTN_TOP       = 18      # top margin
BTN_LEFT      = 18      # left margin


def _ensure_model():
    """Download the hand-landmarker model if it is not already present."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading hand_landmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


# ─────────────────────────────────────────────────────────────────────────
# Main application class
# ─────────────────────────────────────────────────────────────────────────
class SkySquiggle:
    def __init__(self):
        """Initialize the SkySquiggle application."""
        _ensure_model()

        # ── MediaPipe hand-landmarker ─────────────────────────────────
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

        # ── Webcam ────────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        ret, frame = self.cap.read()
        if ret:
            self.frame_height, self.frame_width = frame.shape[:2]
        else:
            self.frame_height, self.frame_width = 720, 1280

        # ── Canvas ────────────────────────────────────────────────────
        self.canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        # ── Drawing state ─────────────────────────────────────────────
        self.drawing_mode = False
        self.prev_x, self.prev_y = None, None

        # ── Colour palette ────────────────────────────────────────────
        self.colors = {
            "Red": (0, 0, 255),
            "Blue": (255, 0, 0),
            "Green": (0, 255, 0),
            "Yellow": (0, 255, 255),
            "White": (255, 255, 255),
            "Clear": (0, 0, 0),
        }
        self.current_color = (0, 0, 255)
        self.current_color_name = "Red"

        # ── UI button setup ───────────────────────────────────────────
        self.buttons = {}
        self._setup_buttons()

        # ── Brush ─────────────────────────────────────────────────────
        self.brush_thickness = 5

        # ── Performance tracking ──────────────────────────────────────
        self.prev_time = 0
        self.frame_counter = 0  # monotonic frame id for MediaPipe

        # ── AI state ──────────────────────────────────────────────────
        self.ai_thinking = False
        self.is_thinking = False  # Rate limit prevention flag
        self.ai_guess_text = ""
        self.ai_guess_time = 0.0  # timestamp when guess arrived
        self.rate_limit_message = ""  # For displaying 429 error messages
        self.rate_limit_time = 0.0  # When to clear the rate limit message

        # ── Gemini ────────────────────────────────────────────────────
        # Explicitly use v1 API (stable, not v1beta)
        self.gemini_client = genai.Client(api_key=_GEMINI_KEY)
        # List available models to find what actually works
        self.gemini_model = None
        try:
            models = list(self.gemini_client.models.list())
            print(f"[Gemini] Found {len(models)} available models")
            # Print all model names for debugging
            all_model_names = [m.name for m in models]
            print(f"[Gemini] Available models: {all_model_names}")
            
            # Find gemini-1.5-flash model (try different patterns)
            flash_models = []
            for m in models:
                name_lower = m.name.lower()
                if "flash" in name_lower and ("1.5" in name_lower or "1_5" in name_lower):
                    flash_models.append(m)
            
            if flash_models:
                # Use the FULL model name (keep "models/" prefix if present)
                model_obj = flash_models[0]
                self.gemini_model = model_obj.name  # Use full name as-is
                print(f"[Gemini] Using model: {self.gemini_model}")
            else:
                # Try to find any model that supports generateContent
                # Look for models with "generateContent" in supported methods
                working_models = []
                for m in models:
                    # Check if model supports generateContent
                    if hasattr(m, 'supported_generation_methods'):
                        if 'generateContent' in m.supported_generation_methods:
                            working_models.append(m)
                    else:
                        # If we can't check, try common names
                        name_lower = m.name.lower()
                        if any(x in name_lower for x in ["flash", "pro", "1.5"]):
                            working_models.append(m)
                
                if working_models:
                    # Use first working model with full name
                    self.gemini_model = working_models[0].name
                    print(f"[Gemini] Using available model: {self.gemini_model}")
                elif models:
                    # Use first available model with full name
                    self.gemini_model = models[0].name
                    print(f"[Gemini] Using first model: {self.gemini_model}")
                else:
                    # Fallback: construct full model path
                    self.gemini_model = "models/gemini-1.5-flash"
                    print(f"[Gemini] No models found, using: {self.gemini_model}")
        except Exception as e:
            print(f"[Gemini] Error listing models: {e}")
            # Fallback: use full model path
            self.gemini_model = "models/gemini-1.5-flash"
            print(f"[Gemini] Using fallback model: {self.gemini_model}")

        # ── ElevenLabs ────────────────────────────────────────────────
        self.elevenlabs_client = ElevenLabs(api_key=_ELEVENLABS_KEY)
        self.elevenlabs_voice_id = ELEVENLABS_VOICE_ID

        # ── Offline TTS fallback (pyttsx3) ────────────────────────────
        self._tts_engine = None
        if _PYTTSX3_AVAILABLE:
            try:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty("rate", 175)
            except Exception:
                pass

        # ── Pygame mixer for audio playback ───────────────────────────
        pygame.mixer.init()


    # ─────────────────────────────────────────────────────────────────
    # Button setup
    # ─────────────────────────────────────────────────────────────────
    def _setup_buttons(self):
        """Create the top-row colour / clear buttons with cartoon layout."""
        x = BTN_LEFT
        y = BTN_TOP
        for color_name, color_rgb in self.colors.items():
            self.buttons[color_name] = {
                "pos": (x, y, x + BTN_W, y + BTN_H),
                "color": color_rgb,
                "name": color_name,
            }
            x += BTN_W + BTN_GAP

    # ─────────────────────────────────────────────────────────────────
    # Finger detection
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _detect_fingers_up(landmarks):
        """Return (index_up, middle_up, ring_up, pinky_up, thumb_up)."""
        index_up  = landmarks[INDEX_TIP].y  < landmarks[INDEX_PIP].y
        middle_up = landmarks[MIDDLE_TIP].y < landmarks[MIDDLE_PIP].y
        ring_up   = landmarks[RING_TIP].y   < landmarks[RING_PIP].y
        pinky_up  = landmarks[PINKY_TIP].y  < landmarks[PINKY_PIP].y
        thumb_up  = landmarks[THUMB_TIP].x  < landmarks[THUMB_IP].x
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

    # ─────────────────────────────────────────────────────────────────
    # Cartoon drawing primitives
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _rounded_rect(frame, x1, y1, x2, y2, r, color, thickness=-1):
        """Draw a rounded rectangle (filled if thickness == -1)."""
        r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
        if r == 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            return
        if thickness == -1:
            # filled: two rects + four corner circles
            cv2.rectangle(frame, (x1 + r, y1), (x2 - r, y2), color, -1)
            cv2.rectangle(frame, (x1, y1 + r), (x2, y2 - r), color, -1)
            cv2.circle(frame, (x1 + r, y1 + r), r, color, -1)
            cv2.circle(frame, (x2 - r, y1 + r), r, color, -1)
            cv2.circle(frame, (x1 + r, y2 - r), r, color, -1)
            cv2.circle(frame, (x2 - r, y2 - r), r, color, -1)
        else:
            # outline: four lines + four arc corners
            cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness)
            cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness)
            cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness)
            cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness)
            cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
            cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
            cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
            cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    # ─────────────────────────────────────────────────────────────────
    # Cartoon button renderer
    # ─────────────────────────────────────────────────────────────────
    def _draw_cartoon_button(self, frame, x1, y1, x2, y2,
                             fill_color, label, is_active=False):
        """Render one cartoon button with shadow, fill, border, label."""
        r = CORNER_RADIUS
        so = SHADOW_OFFSET
        bt = BORDER_THICK

        if is_active:
            # ── "Popped" active state: white glow halo ───────────────
            glow = 6
            self._rounded_rect(frame,
                                x1 - glow, y1 - glow, x2 + glow, y2 + glow,
                                r + 4, (255, 255, 255), -1)
            # Slight upward shadow to feel "raised"
            self._rounded_rect(frame,
                                x1 - 2, y1 - 3, x2 - 2, y2 - 3,
                                r, (180, 180, 180), -1)
        else:
            # ── Hard offset shadow (solid black, bottom-right) ───────
            self._rounded_rect(frame,
                                x1 + so, y1 + so, x2 + so, y2 + so,
                                r, (0, 0, 0), -1)

        # ── Fill ─────────────────────────────────────────────────────
        self._rounded_rect(frame, x1, y1, x2, y2, r, fill_color, -1)

        # ── Inner highlight stripe (top) for "3D" pop ────────────────
        hi_color = tuple(min(255, c + 60) for c in fill_color)
        self._rounded_rect(frame, x1 + 4, y1 + 4, x2 - 4, y1 + 14,
                            r // 2, hi_color, -1)

        # ── Thick black border ───────────────────────────────────────
        self._rounded_rect(frame, x1, y1, x2, y2, r, (0, 0, 0), bt)

        # ── Label centred ────────────────────────────────────────────
        text_color = (0, 0, 0) if label in ("Yellow", "White", "Green") else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thick = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 + (y2 - y1 + th) // 2 + 2
        # text shadow
        cv2.putText(frame, label, (tx + 1, ty + 1), font, scale, (0, 0, 0), thick + 1)
        cv2.putText(frame, label, (tx, ty), font, scale, text_color, thick)

    # ─────────────────────────────────────────────────────────────────
    # Drawing helpers
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _draw_hand_landmarks(frame, landmarks, w, h):
        """Draw hand skeleton on the frame."""
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 128), 2)
        for px, py in pts:
            cv2.circle(frame, (px, py), 4, (255, 0, 128), -1)

    def _draw_ui(self, frame):
        """Render the cartoon-style colour-palette buttons."""
        for name, info in self.buttons.items():
            x1, y1, x2, y2 = info["pos"]
            fill = info["color"]
            # Darken the "Clear" button fill to dark-grey
            if name == "Clear":
                fill = (50, 50, 50)
            active = (name == self.current_color_name and name != "Clear")
            self._draw_cartoon_button(frame, x1, y1, x2, y2,
                                       fill, name, is_active=active)

    def _draw_cursor(self, frame, x, y, mode="hover"):
        """Cartoon-style cursor at the fingertip."""
        if mode == "drawing":
            # Filled brush dot + thick cartoon outline
            cv2.circle(frame, (x, y), self.brush_thickness + 6,
                       (0, 0, 0), 3)                           # black outline
            cv2.circle(frame, (x, y), self.brush_thickness + 4,
                       self.current_color, -1)                  # colour fill
            # Small white highlight
            cv2.circle(frame, (x - 2, y - 2), 3, (255, 255, 255), -1)
        else:
            # Cross-hair ring
            cv2.circle(frame, (x, y), 16, (0, 0, 0), 3)
            cv2.circle(frame, (x, y), 14, (255, 255, 0), 2)
            cv2.line(frame, (x - 20, y), (x + 20, y), (0, 0, 0), 1)
            cv2.line(frame, (x, y - 20), (x, y + 20), (0, 0, 0), 1)
            cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)

    # ─────────────────────────────────────────────────────────────────
    # Cartoon thought bubble (replaces fullscreen purple overlay)
    # ─────────────────────────────────────────────────────────────────
    def _draw_thought_bubble(self, frame):
        """Render a cartoon thought bubble with animated progress bar."""
        now = time.time()
        bw, bh = 300, 80
        bx = self.frame_width - bw - 24
        by = self.frame_height - bh - 90

        # ── Shadow ───────────────────────────────────────────────────
        self._rounded_rect(frame,
                            bx + SHADOW_OFFSET, by + SHADOW_OFFSET,
                            bx + bw + SHADOW_OFFSET, by + bh + SHADOW_OFFSET,
                            22, (0, 0, 0), -1)
        # ── White bubble ─────────────────────────────────────────────
        self._rounded_rect(frame, bx, by, bx + bw, by + bh,
                            22, (255, 255, 255), -1)
        self._rounded_rect(frame, bx, by, bx + bw, by + bh,
                            22, (0, 0, 0), BORDER_THICK)

        # ── Small trailing thought circles ───────────────────────────
        cx, cy = bx - 6, by + bh + 4
        for radius in (12, 7, 4):
            # shadow
            cv2.circle(frame, (cx + 3, cy + 3), radius, (0, 0, 0), -1)
            cv2.circle(frame, (cx, cy), radius, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy), radius, (0, 0, 0), 2)
            cx -= radius + 6
            cy += radius + 2

        # ── Animated "Thinking…" text ────────────────────────────────
        dots = "." * (int(now * 2) % 4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Thinking" + dots, (bx + 22, by + 34),
                    font, 0.75, (0, 0, 0), 2)

        # ── Filling progress bar ─────────────────────────────────────
        bar_x = bx + 18
        bar_y = by + 50
        bar_w = bw - 36
        bar_h = 16
        # bar background
        self._rounded_rect(frame, bar_x, bar_y,
                            bar_x + bar_w, bar_y + bar_h,
                            bar_h // 2, (210, 210, 210), -1)
        # animated fill (bounces left to right)
        fill_pct = 0.5 + 0.5 * math.sin(now * 3)
        fill_w = max(bar_h, int(bar_w * fill_pct))
        self._rounded_rect(frame, bar_x, bar_y,
                            bar_x + fill_w, bar_y + bar_h,
                            bar_h // 2, (180, 120, 255), -1)
        # bar border
        self._rounded_rect(frame, bar_x, bar_y,
                            bar_x + bar_w, bar_y + bar_h,
                            bar_h // 2, (0, 0, 0), 2)

    # ─────────────────────────────────────────────────────────────────
    # HUD: cartoon-style status, thought bubble, guess banner
    # ─────────────────────────────────────────────────────────────────
    def _draw_info(self, frame):
        """Draw cartoon-style HUD elements."""
        now = time.time()

        # ── Bottom-left cartoon pill: mode + shortcuts ────────────────
        mode_text = "DRAWING" if self.drawing_mode else "HOVER"
        instructions = "G: Guess  S: Save  Q: Quit"

        pill_w, pill_h = 370, 55
        pill_x, pill_y = 14, self.frame_height - pill_h - 14
        # shadow
        self._rounded_rect(frame,
                            pill_x + 4, pill_y + 4,
                            pill_x + pill_w + 4, pill_y + pill_h + 4,
                            pill_h // 2, (0, 0, 0), -1)
        # fill
        self._rounded_rect(frame, pill_x, pill_y,
                            pill_x + pill_w, pill_y + pill_h,
                            pill_h // 2, (40, 40, 40), -1)
        self._rounded_rect(frame, pill_x, pill_y,
                            pill_x + pill_w, pill_y + pill_h,
                            pill_h // 2, (0, 0, 0), 2)

        mode_color = (0, 255, 0) if self.drawing_mode else (0, 255, 255)
        cv2.putText(frame, mode_text, (pill_x + 20, pill_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)
        cv2.putText(frame, instructions, (pill_x + 20, pill_y + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

        # ── Bottom-right cartoon pill: FPS ────────────────────────────
        fps = 1 / (now - self.prev_time + 1e-6)
        self.prev_time = now
        fps_text = f"FPS: {int(fps)}"
        fp_w, fp_h = 115, 36
        fp_x = self.frame_width - fp_w - 14
        fp_y = self.frame_height - fp_h - 14
        self._rounded_rect(frame, fp_x + 3, fp_y + 3,
                            fp_x + fp_w + 3, fp_y + fp_h + 3,
                            fp_h // 2, (0, 0, 0), -1)
        self._rounded_rect(frame, fp_x, fp_y,
                            fp_x + fp_w, fp_y + fp_h,
                            fp_h // 2, (40, 40, 40), -1)
        self._rounded_rect(frame, fp_x, fp_y,
                            fp_x + fp_w, fp_y + fp_h,
                            fp_h // 2, (0, 0, 0), 2)
        cv2.putText(frame, fps_text, (fp_x + 16, fp_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # ── Thought bubble while AI is thinking ───────────────────────
        if self.ai_thinking:
            self._draw_thought_bubble(frame)

        # ── Rate-limit warning banner ─────────────────────────────────
        if self.rate_limit_message and (now - self.rate_limit_time) < 5.0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, thick = 0.8, 2
            (tw, th), _ = cv2.getTextSize(self.rate_limit_message, font, scale, thick)
            bw = tw + 50
            bh = th + 30
            bx = (self.frame_width - bw) // 2
            by = BTN_TOP + BTN_H + 20
            self._rounded_rect(frame, bx + 5, by + 5, bx + bw + 5, by + bh + 5,
                                12, (0, 0, 0), -1)
            self._rounded_rect(frame, bx, by, bx + bw, by + bh,
                                12, (60, 40, 140), -1)
            self._rounded_rect(frame, bx, by, bx + bw, by + bh,
                                12, (0, 0, 0), 3)
            cv2.putText(frame, self.rate_limit_message,
                        (bx + 25, by + th + 14),
                        font, scale, (100, 180, 255), thick)

        # ── Guess banner (cartoon speech bubble, fades) ───────────────
        if self.ai_guess_text and (now - self.ai_guess_time) < GUESS_DISPLAY_SECONDS:
            elapsed = now - self.ai_guess_time
            fade_start = GUESS_DISPLAY_SECONDS - 2
            if elapsed > fade_start:
                banner_alpha = max(0.0, 1.0 - (elapsed - fade_start) / 2.0)
            else:
                banner_alpha = 1.0

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, thick = 0.85, 2
            (tw, th), _ = cv2.getTextSize(self.ai_guess_text, font, scale, thick)
            bw = tw + 50
            bh = th + 36
            bx = (self.frame_width - bw) // 2
            by = BTN_TOP + BTN_H + 20

            overlay = frame.copy()
            # shadow
            self._rounded_rect(overlay, bx + 5, by + 5, bx + bw + 5, by + bh + 5,
                                14, (0, 0, 0), -1)
            # bubble fill
            self._rounded_rect(overlay, bx, by, bx + bw, by + bh,
                                14, (255, 255, 255), -1)
            # border
            self._rounded_rect(overlay, bx, by, bx + bw, by + bh,
                                14, (0, 0, 0), 3)
            # small triangle pointer
            tri_cx = bx + bw // 2
            tri_pts = np.array([
                [tri_cx - 12, by + bh],
                [tri_cx + 12, by + bh],
                [tri_cx, by + bh + 18],
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [tri_pts], (255, 255, 255))
            cv2.polylines(overlay, [tri_pts], True, (0, 0, 0), 3)
            # text
            cv2.putText(overlay, self.ai_guess_text,
                        (bx + 25, by + th + 16),
                        font, scale, (50, 30, 80), thick)

            cv2.addWeighted(overlay, banner_alpha, frame,
                            1 - banner_alpha, 0, frame)

    # ─────────────────────────────────────────────────────────────────
    # AI pipeline (runs on a background thread)
    # ─────────────────────────────────────────────────────────────────
    def _play_audio(self, audio_bytes: bytes):
        """Write audio bytes to temp mp3 and play via pygame."""
        if not audio_bytes:
            return
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        try:
            tmp.write(audio_bytes)
            tmp.close()
            pygame.mixer.music.load(tmp.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            pygame.mixer.music.unload()
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _speak_offline(self, text: str):
        """Speak text using pyttsx3 (offline fallback)."""
        if self._tts_engine:
            try:
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
            except Exception as e:
                print(f"[pyttsx3 error] {e}")
        else:
            print(f"[TTS unavailable] Would say: {text}")

    def _speak(self, text: str):
        """Speak text via ElevenLabs, falling back to offline TTS."""
        if self.elevenlabs_voice_id:
            try:
                audio_iter = self.elevenlabs_client.text_to_speech.convert(
                    text=text,
                    voice_id=self.elevenlabs_voice_id,
                    model_id=ELEVENLABS_MODEL_ID,
                    output_format="mp3_44100_128",
                )
                audio_bytes = b"".join(audio_iter)
                self._play_audio(audio_bytes)
                return
            except Exception as e:
                print(f"[ElevenLabs error] {e}")
        # Fallback to offline TTS
        self._speak_offline(text)

    def _speak_filler(self):
        """Pick and speak a random filler phrase."""
        filler = random.choice(FILLER_PHRASES)
        print(f"[AI filler] {filler}")
        self._speak(filler)

    def _analyze_drawing(self) -> str:
        """Save canvas, send to Gemini, return the witty guess text."""
        # Save canvas to a temporary PNG
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        try:
            cv2.imwrite(tmp.name, self.canvas)
            tmp.close()
            img = Image.open(tmp.name)

            # Add delay before API call to prevent rate limiting
            time.sleep(2)

            try:
                print(f"[Gemini] Calling {self.gemini_model}...")
                # Use v1 API explicitly (stable, not v1beta)
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=[GEMINI_PROMPT, img],
                )
                guess = response.text.strip()
                print(f"[Gemini guess] {guess}")
                return guess
            except Exception as e:
                err_str = str(e)
                print(f"[Gemini error] {err_str[:200]}")
                
                # Handle 429 rate limit error
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    self.rate_limit_message = "Too fast! Wait a moment..."
                    self.rate_limit_time = time.time()
                    print("[Gemini] Rate limit hit – showing message to user")
                    return "Hmm, I'm speechless on this one!"
                else:
                    return "Hmm, I'm speechless on this one!"
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _ai_guess_pipeline(self):
        """Background thread: filler -> Gemini analysis -> speak guess."""
        try:
            # Step 1: speak a filler phrase while we wait
            self._speak_filler()

            # Step 2: send canvas to Gemini
            guess = self._analyze_drawing()

            # Step 3: display + speak the Gemini guess
            self.ai_guess_text = guess
            self.ai_guess_time = time.time()
            self._speak(guess)
        finally:
            self.ai_thinking = False
            self.is_thinking = False  # Clear rate limit prevention flag

    # ─────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────
    def save_canvas(self):
        """Export the drawing canvas as a .png file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"sky_squiggle_{timestamp}.png"
        cv2.imwrite(filename, self.canvas)
        print(f"Canvas saved as: {filename}")
        return filename

    # ─────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────
    def run(self):
        """Start the application."""
        print("SkySquiggle – Starting ...")
        print("Controls:")
        print("  Index finger only  -> Draw")
        print("  Index + Middle     -> Hover / Select")
        print("  Hover buttons      -> Pick colour")
        print("  G -> AI Guess   S -> Save   Q -> Quit\n")

        button_hover_timer: dict[str, float] = {}
        hover_threshold = 0.5

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            self.frame_counter += 1
            timestamp_ms = int(self.frame_counter * (1000 / 30))
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                self._draw_hand_landmarks(frame, landmarks,
                                          self.frame_width, self.frame_height)

                index_up, middle_up, ring_up, pinky_up, thumb_up = \
                    self._detect_fingers_up(landmarks)

                x, y = self._get_index_finger_pos(landmarks)

                # ── DRAWING MODE ──────────────────────────────────────
                if index_up and not middle_up and not ring_up and not pinky_up:
                    self.drawing_mode = True
                    if self._check_button_click(x, y):
                        self._draw_cursor(frame, x, y, "hover")
                        self.prev_x, self.prev_y = None, None
                    else:
                        if self.prev_x is not None and self.prev_y is not None:
                            cv2.line(self.canvas,
                                     (self.prev_x, self.prev_y), (x, y),
                                     self.current_color, self.brush_thickness)
                        self.prev_x, self.prev_y = x, y
                        self._draw_cursor(frame, x, y, "drawing")

                # ── HOVER / SELECT MODE ───────────────────────────────
                elif index_up and middle_up:
                    self.drawing_mode = False
                    self.prev_x, self.prev_y = None, None
                    self._draw_cursor(frame, x, y, "hover")

                    btn = self._check_button_click(x, y)
                    if btn:
                        if btn not in button_hover_timer:
                            button_hover_timer[btn] = time.time()
                        elapsed = time.time() - button_hover_timer[btn]

                        x1, y1, x2, y2 = self.buttons[btn]["pos"]
                        pct = min(elapsed / hover_threshold, 1.0)
                        bar_w = int((x2 - x1) * pct)
                        bar_y = y2 + SHADOW_OFFSET + 6
                        bar_h = 10
                        # bar background
                        self._rounded_rect(frame, x1, bar_y,
                                            x2, bar_y + bar_h,
                                            bar_h // 2, (80, 80, 80), -1)
                        # bar fill
                        if bar_w > bar_h:
                            self._rounded_rect(frame, x1, bar_y,
                                                x1 + bar_w, bar_y + bar_h,
                                                bar_h // 2, (0, 255, 0), -1)
                        # bar border
                        self._rounded_rect(frame, x1, bar_y,
                                            x2, bar_y + bar_h,
                                            bar_h // 2, (0, 0, 0), 2)

                        if elapsed >= hover_threshold:
                            if btn == "Clear":
                                self.canvas[:] = 0
                                print("Canvas cleared!")
                            else:
                                self.current_color = self.colors[btn]
                                self.current_color_name = btn
                                print(f"Colour -> {btn}")
                            button_hover_timer.clear()
                    else:
                        button_hover_timer.clear()

                else:
                    self.drawing_mode = False
                    self.prev_x, self.prev_y = None, None
                    button_hover_timer.clear()
            else:
                self.drawing_mode = False
                self.prev_x, self.prev_y = None, None
                button_hover_timer.clear()

            # ── Merge canvas onto camera feed ─────────────────────────
            mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
            frame = cv2.add(frame_bg, canvas_fg)

            # ── UI overlay ────────────────────────────────────────────
            self._draw_ui(frame)
            self._draw_info(frame)

            cv2.imshow("SkySquiggle - Air Canvas", frame)

            # ── Keyboard ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("s"), ord("S")):
                self.save_canvas()
            elif key in (ord("g"), ord("G")):
                # Rate limit prevention: ignore if already processing
                if not self.ai_thinking and not self.is_thinking:
                    self.ai_thinking = True
                    self.is_thinking = True
                    threading.Thread(target=self._ai_guess_pipeline, daemon=True).start()

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()
        pygame.mixer.quit()
        print("Application closed.")


def main():
    app = SkySquiggle()
    app.run()


if __name__ == "__main__":
    main()
