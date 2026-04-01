"""
bubo/face/led_matrix_face.py — Bubo v10000

LED Matrix Face Controller: Egg-Shaped Emotional Expression Display

════════════════════════════════════════════════════════════════════
HARDWARE DESIGN
════════════════════════════════════════════════════════════════════

Physical layout: 5× Adafruit 8×8 LED Matrix with HT16K33 I2C backpack
arranged in an egg shape (approximately 200mm tall × 140mm wide):

         [M0]          ← top cap (8×8, address 0x70)
       [M1][M2]        ← upper cheeks (8×8, 0x71 left / 0x72 right)
       [M3][M4]        ← lower face (8×8, 0x73 left / 0x74 right)

Total: 40×16 effective pixels remapped onto an egg-shaped canvas.

Virtual canvas: 24 wide × 32 tall pixels.
The egg mask clips to the rounded shape.

Wiring (I2C to Jetson Nano / Social Node):
  SDA → Pin 3 (GPIO2)
  SCL → Pin 5 (GPIO3)
  VCC → 3.3V or 5V
  GND → GND
  Each matrix has a unique I2C address (set via solder jumpers A0/A1/A2)

Cost: 5× Adafruit 878 @ ~$9.95 each = ~$50 total.

════════════════════════════════════════════════════════════════════
EMOTIONAL EXPRESSION MAP
════════════════════════════════════════════════════════════════════

All 14 EmotionChip emotions + neutral mapped to pixel patterns.
Expressions use three channels:
  BROWS: upper eyelid / eyebrow simulation (rows 0-7)
  EYES:  main eye area (rows 8-18)
  MOUTH: lower face expression (rows 20-28)

Johnny 5 inspiration: the entire character of Short Circuit was
conveyed primarily through the angle and openness of the eye
covers. Small changes = enormous emotional shift.
The same principle applies here: brow angle is everything.

Designed to work WITH the Johnny 5 servo eye covers in eye_covers.py.
The LED face provides secondary expression detail; the servo covers
provide the primary emotional geometry.
"""

import time
import threading
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple
from enum import Enum

logger = logging.getLogger("LEDFace")

try:
    import board
    import busio
    from adafruit_ht16k33 import matrix as ht16k33_matrix
    HAS_HARDWARE = True
except ImportError:
    HAS_HARDWARE = False
    logger.info("LED face: simulation mode (adafruit_ht16k33 not installed)")


# ── Canvas dimensions ─────────────────────────────────────────────────────────

CANVAS_W = 24
CANVAS_H = 32

# Egg-shaped mask — True = pixel visible
def _make_egg_mask() -> np.ndarray:
    mask = np.zeros((CANVAS_H, CANVAS_W), dtype=bool)
    cx, cy = CANVAS_W // 2, CANVAS_H // 2
    for y in range(CANVAS_H):
        for x in range(CANVAS_W):
            # Egg formula: wider at top (eyes), narrower at chin
            dy = (y - cy) / cy
            dx = (x - cx) / (cx * (1.0 - 0.25 * dy))  # slight taper toward chin
            if dx*dx + dy*dy <= 1.0:
                mask[y, x] = True
    return mask

EGG_MASK = _make_egg_mask()


# ── Expression definitions ────────────────────────────────────────────────────
# Each expression is a dict with pixel patterns for:
#   brows, left_eye, right_eye, mouth
# Represented as lists of (x, y) pixel coordinates to illuminate
# on the 24×32 canvas.

class Expressions:
    """
    Pixel art expressions for the LED face.
    Coordinate origin: top-left of 24×32 canvas.
    
    BROW region:  y = 2-8    (raised/lowered for emotion)
    EYE region:   y = 9-18   (openness, shape)
    MOUTH region: y = 22-29  (smile/frown/neutral)
    
    Johnny 5 principle: angle of brows does 70% of the work.
    """

    NEUTRAL = {
        "name": "neutral",
        "brows": [(8,4),(9,4),(10,4),(11,4),  (13,4),(14,4),(15,4),(16,4)],
        "eyes":  [
            (8,11),(9,11),(10,11),(11,11),   (13,11),(14,11),(15,11),(16,11),
            (8,12),(11,12),                   (13,12),(16,12),
            (8,13),(9,13),(10,13),(11,13),   (13,13),(14,13),(15,13),(16,13),
        ],
        "mouth": [(10,25),(11,25),(12,25),(13,25),(14,25)],
        "blink_speed": 0.0,
    }

    JOY = {
        "name": "joy",
        "brows": [(8,3),(9,3),(10,3),(11,3),  (13,3),(14,3),(15,3),(16,3)],  # raised
        "eyes":  [
            (8,11),(9,10),(10,10),(11,11),   (13,11),(14,10),(15,10),(16,11),
            (8,12),                  (11,12), (13,12),               (16,12),
            (8,13),(9,13),(10,13),(11,13),   (13,13),(14,13),(15,13),(16,13),
            (9,14),(10,14),                  (14,14),(15,14),  # sparkle dots
        ],
        "mouth": [
            (9,24),(10,23),(11,23),(12,23),(13,23),(14,23),(15,24),  # smile arc
            (10,25),(11,25),(12,25),(13,25),(14,25),
        ],
        "blink_speed": 0.0,
    }

    SADNESS = {
        "name": "sadness",
        "brows": [
            (8,5),(9,4),(10,4),(11,5),   # drooping inward
            (13,5),(14,4),(15,4),(16,5),
        ],
        "eyes":  [
            (8,12),(9,12),(10,12),(11,12),  (13,12),(14,12),(15,12),(16,12),
            (8,13),(11,13),                  (13,13),(16,13),
            (9,14),(10,14),                  (14,14),(15,14),
            # tear drops
            (9,16),  (14,16),
            (9,17),  (14,17),
        ],
        "mouth": [
            (9,26),(10,27),(11,27),(12,27),(13,27),(14,27),(15,26),  # frown
        ],
        "blink_speed": 0.0,
    }

    FEAR = {
        "name": "fear",
        "brows": [
            (8,6),(9,5),(10,4),(11,4),  # inner raised sharply
            (13,4),(14,4),(15,5),(16,6),
        ],
        "eyes":  [
            (7,10),(8,10),(9,10),(10,10),(11,10),(12,10),  # wide open left
            (12,10),(13,10),(14,10),(15,10),(16,10),(17,10),  # wide open right
            (7,11),(12,11), (12,11),(17,11),
            (7,12),(12,12), (12,12),(17,12),
            (7,13),(8,13),(9,13),(10,13),(11,13),(12,13),
            (12,13),(13,13),(14,13),(15,13),(16,13),(17,13),
        ],
        "mouth": [
            (9,25),(10,24),(11,24),(12,24),(13,24),(14,24),(15,25),
            (10,25),(11,25),(12,25),(13,25),(14,25),
        ],
        "blink_speed": 0.0,
    }

    ANGER = {
        "name": "anger",
        "brows": [
            (8,6),(9,6),(10,5),(11,4),  # V-shape anger — inner corners down
            (13,4),(14,5),(15,6),(16,6),
        ],
        "eyes":  [
            (8,12),(9,12),(10,12),(11,12),   (13,12),(14,12),(15,12),(16,12),
            (9,13),(10,13),                   (14,13),(15,13),  # narrowed
        ],
        "mouth": [
            (9,26),(10,26),(11,25),(12,25),(13,25),(14,26),(15,26),
        ],
        "blink_speed": 0.0,
    }

    SURPRISE = {
        "name": "surprise",
        "brows": [
            (8,2),(9,2),(10,2),(11,2),  # maximally raised
            (13,2),(14,2),(15,2),(16,2),
        ],
        "eyes":  [
            (8,10),(9,10),(10,10),(11,10),   (13,10),(14,10),(15,10),(16,10),
            (8,11),(11,11),                   (13,11),(16,11),
            (8,12),(11,12),                   (13,12),(16,12),
            (8,13),(9,13),(10,13),(11,13),   (13,13),(14,13),(15,13),(16,13),
        ],
        "mouth": [
            # O shape
            (11,24),(12,24),(13,24),
            (10,25),(14,25),
            (10,26),(14,26),
            (11,27),(12,27),(13,27),
        ],
        "blink_speed": 0.0,
    }

    CURIOSITY = {
        "name": "curiosity",
        "brows": [
            (8,5),(9,4),(10,3),(11,4),  # left brow raised higher — quizzical tilt
            (13,4),(14,4),(15,4),(16,4),
        ],
        "eyes":  [
            # left eye slightly larger (curious lean)
            (7,10),(8,10),(9,10),(10,10),(11,10),(12,10),
            (7,11),(12,11),
            (7,12),(12,12),
            (7,13),(8,13),(9,13),(10,13),(11,13),(12,13),
            # right eye normal
            (13,11),(14,11),(15,11),(16,11),
            (13,12),(16,12),
            (13,13),(14,13),(15,13),(16,13),
        ],
        "mouth": [
            (10,25),(11,25),(12,24),(13,25),(14,25),  # slight asymmetric smile
        ],
        "blink_speed": 0.0,
    }

    EMPATHY = {
        "name": "empathy",
        "brows": [
            (8,4),(9,4),(10,4),(11,5),   # gentle inner softening
            (13,5),(14,4),(15,4),(16,4),
        ],
        "eyes":  [
            (8,11),(9,11),(10,11),(11,11),  (13,11),(14,11),(15,11),(16,11),
            (8,12),(9,12),(10,12),(11,12),  (13,12),(14,12),(15,12),(16,12),
            (9,13),(10,13),                  (14,13),(15,13),
        ],
        "mouth": [
            (9,24),(10,23),(11,23),(12,23),(13,23),(14,23),(15,24),
            (10,24),(14,24),
        ],
        "blink_speed": 2.5,  # slow gentle blinks
    }

    CONTENTMENT = {
        "name": "contentment",
        "brows": [(8,4),(9,4),(10,4),(11,4),  (13,4),(14,4),(15,4),(16,4)],
        "eyes":  [
            # half-closed, soft
            (8,12),(9,12),(10,12),(11,12),  (13,12),(14,12),(15,12),(16,12),
            (9,13),(10,13),                  (14,13),(15,13),
        ],
        "mouth": [
            (9,24),(10,23),(11,23),(12,23),(13,23),(14,23),(15,24),
        ],
        "blink_speed": 4.0,  # very slow drowsy blinks
    }

    EXCITEMENT = {
        "name": "excitement",
        "brows": [
            (8,2),(9,2),(10,3),(11,2),  # wildly raised
            (13,2),(14,3),(15,2),(16,2),
        ],
        "eyes":  [
            (7,9),(8,9),(9,9),(10,9),(11,9),(12,9),
            (13,9),(14,9),(15,9),(16,9),(17,9),
            (7,10),(12,10), (13,10),(17,10),
            (7,11),(12,11), (13,11),(17,11),
            (7,12),(8,12),(9,12),(10,12),(11,12),(12,12),
            (13,12),(14,12),(15,12),(16,12),(17,12),
            # sparkle pixels
            (5,8),(18,8),(5,14),(18,14),
        ],
        "mouth": [
            (8,23),(9,22),(10,22),(11,22),(12,22),(13,22),(14,22),(15,22),(16,23),
            (9,23),(15,23),
        ],
        "blink_speed": 0.3,  # rapid excited blinks
    }

    CONFUSION = {
        "name": "confusion",
        "brows": [
            (8,5),(9,4),(10,4),(11,4),   # left normal
            (13,3),(14,4),(15,5),(16,5),  # right raised asymmetric
        ],
        "eyes":  [
            (8,11),(9,11),(10,11),(11,11),  (13,11),(14,11),(15,11),(16,11),
            (8,12),(11,12),                  (13,12),(16,12),
            (8,13),(9,13),(10,13),(11,13),  (13,13),(14,13),(15,13),(16,13),
            # question mark suggestion right side
            (17,10),(17,12),(17,14),
        ],
        "mouth": [
            (10,25),(11,25),(12,24),(13,25),(14,25),(15,26),  # zigzag uncertain
        ],
        "blink_speed": 0.0,
    }

    LONELINESS = {
        "name": "loneliness",
        "brows": [
            (8,5),(9,5),(10,5),(11,5),  # heavy, low
            (13,5),(14,5),(15,5),(16,5),
        ],
        "eyes":  [
            # downcast — lower portion only
            (9,14),(10,14),(11,14),  (13,14),(14,14),(15,14),
            (9,15),(10,15),           (14,15),(15,15),
        ],
        "mouth": [
            (9,27),(10,28),(11,28),(12,28),(13,28),(14,28),(15,27),
        ],
        "blink_speed": 5.0,
    }

    PRIDE = {
        "name": "pride",
        "brows": [
            (8,4),(9,4),(10,4),(11,4),
            (13,4),(14,4),(15,4),(16,4),
        ],
        "eyes":  [
            (8,12),(9,12),(10,12),(11,12),  (13,12),(14,12),(15,12),(16,12),
            (9,13),(10,13),                  (14,13),(15,13),
        ],
        "mouth": [
            (9,24),(10,23),(11,23),(12,23),(13,23),(14,23),(15,24),
            (15,25),  # slight asymmetric proud smirk
        ],
        "blink_speed": 0.0,
    }

    SHAME = {
        "name": "shame",
        "brows": [
            (8,6),(9,5),(10,5),(11,5),
            (13,5),(14,5),(15,5),(16,6),
        ],
        "eyes":  [
            # looking down
            (9,14),(10,14),(11,14),  (13,14),(14,14),(15,14),
        ],
        "mouth": [(10,26),(11,27),(12,27),(13,27),(14,26)],
        "blink_speed": 0.0,
    }

    DISGUST = {
        "name": "disgust",
        "brows": [
            (8,5),(9,5),(10,4),(11,4),
            (13,4),(14,5),(15,6),(16,6),  # asymmetric sneer
        ],
        "eyes":  [
            (8,12),(9,12),(10,12),(11,12),  (13,12),(14,12),(15,12),(16,12),
            (9,13),(10,13),
        ],
        "mouth": [
            (9,25),(10,24),(11,24),(12,24),  # half sneer left
            (13,25),(14,26),(15,26),
        ],
        "blink_speed": 0.0,
    }


# Emotion name → expression mapping
EMOTION_MAP = {
    "neutral":    Expressions.NEUTRAL,
    "joy":        Expressions.JOY,
    "sadness":    Expressions.SADNESS,
    "fear":       Expressions.FEAR,
    "anger":      Expressions.ANGER,
    "surprise":   Expressions.SURPRISE,
    "curiosity":  Expressions.CURIOSITY,
    "empathy":    Expressions.EMPATHY,
    "contentment":Expressions.CONTENTMENT,
    "excitement": Expressions.EXCITEMENT,
    "confusion":  Expressions.CONFUSION,
    "loneliness": Expressions.LONELINESS,
    "pride":      Expressions.PRIDE,
    "shame":      Expressions.SHAME,
    "disgust":    Expressions.DISGUST,
}


# ── LED Matrix Driver ─────────────────────────────────────────────────────────

class SimulatedMatrix:
    """Simulated matrix for development/testing without hardware."""
    def __init__(self, addr): self._addr = addr; self._pixels = {}
    def __setitem__(self, xy, v): self._pixels[xy] = v
    def fill(self, v): self._pixels.clear()
    def show(self): pass


class LEDMatrixFace:
    """
    Egg-shaped LED matrix face controller.
    Renders EmotionChip emotional states as pixel art expressions.
    
    Works identically in simulation (no hardware) and on real Jetson
    hardware with Adafruit HT16K33 I2C matrices.
    """

    I2C_ADDRESSES = [0x70, 0x71, 0x72, 0x73, 0x74]
    TRANSITION_FPS = 30
    BLINK_DURATION_MS = 150

    def __init__(self, brightness: float = 0.7):
        self._brightness  = brightness
        self._canvas      = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
        self._target      = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
        self._current_expr= "neutral"
        self._blink_task  = None
        self._running     = False
        self._lock        = threading.Lock()
        self._matrices    = self._init_matrices()

    def _init_matrices(self):
        if not HAS_HARDWARE:
            return [SimulatedMatrix(a) for a in self.I2C_ADDRESSES]
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            mats = []
            for addr in self.I2C_ADDRESSES:
                try:
                    m = ht16k33_matrix.MatrixBackpack8x8(i2c, address=addr)
                    m.brightness = int(self._brightness * 15)
                    mats.append(m)
                except Exception as e:
                    logger.warning(f"Matrix {hex(addr)} not found: {e}")
                    mats.append(SimulatedMatrix(addr))
            return mats
        except Exception as e:
            logger.warning(f"I2C init failed: {e} — simulation mode")
            return [SimulatedMatrix(a) for a in self.I2C_ADDRESSES]

    def express(self, emotion: str, intensity: float = 1.0,
                transition_ms: int = 300):
        """
        Display an emotional expression.
        Transitions smoothly from current expression.
        """
        expr = EMOTION_MAP.get(emotion.lower(), Expressions.NEUTRAL)
        self._current_expr = emotion

        # Build target canvas
        target = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
        brightness = min(255, int(intensity * 255))

        for region in ("brows", "eyes", "mouth"):
            for x, y in expr.get(region, []):
                if 0 <= y < CANVAS_H and 0 <= x < CANVAS_W:
                    if EGG_MASK[y, x]:
                        target[y, x] = brightness

        with self._lock:
            self._target = target

        # Start transition
        threading.Thread(target=self._transition,
                         args=(target, transition_ms), daemon=True).start()

        # Schedule blink if emotion has a blink rate
        blink_speed = expr.get("blink_speed", 0.0)
        if blink_speed > 0:
            threading.Thread(target=self._auto_blink,
                             args=(blink_speed,), daemon=True).start()

        logger.debug(f"Face: expressing {emotion} (intensity={intensity:.2f})")

    def blink(self):
        """Execute a single natural blink."""
        self._set_eye_row_brightness(0)
        time.sleep(self.BLINK_DURATION_MS / 1000)
        self._set_eye_row_brightness(255)

    def _set_eye_row_brightness(self, brightness: int):
        eye_rows = range(9, 18)
        with self._lock:
            canvas = self._canvas.copy()
        for y in eye_rows:
            for x in range(CANVAS_W):
                if canvas[y, x] > 0:
                    canvas[y, x] = brightness
        self._render(canvas)

    def _transition(self, target: np.ndarray, duration_ms: int):
        steps  = max(1, duration_ms * self.TRANSITION_FPS // 1000)
        with self._lock:
            start = self._canvas.copy()
        for i in range(steps + 1):
            alpha  = i / steps
            frame  = (start * (1 - alpha) + target * alpha).astype(np.uint8)
            frame  = (frame * EGG_MASK).astype(np.uint8)
            with self._lock:
                self._canvas = frame
            self._render(frame)
            time.sleep(1 / self.TRANSITION_FPS)

    def _auto_blink(self, interval_s: float):
        """Autonomous periodic blinking."""
        while self._current_expr in EMOTION_MAP:
            time.sleep(interval_s + np.random.normal(0, 0.3))
            self.blink()

    def _render(self, canvas: np.ndarray):
        """Map 24×32 canvas onto physical matrices and push."""
        # Matrix layout mapping (approximate):
        # M0 (0x70): top 8 rows, centre 8 cols (8:16)
        # M1 (0x71): rows 8-15, left cols 0:8
        # M2 (0x72): rows 8-15, right cols 16:24
        # M3 (0x73): rows 16-23, left cols 0:8
        # M4 (0x74): rows 16-23, right cols 8:16

        regions = [
            (0, 8,  8, 16),   # M0: forehead
            (8, 16, 0, 8),    # M1: upper left
            (8, 16, 16, 24),  # M2: upper right
            (16, 24, 0, 8),   # M3: lower left
            (16, 24, 8, 16),  # M4: lower right / chin
        ]

        for mat_idx, (y0, y1, x0, x1) in enumerate(regions):
            if mat_idx >= len(self._matrices): continue
            mat = self._matrices[mat_idx]
            mat.fill(0)
            sub = canvas[y0:y1, x0:x1]
            for local_y in range(sub.shape[0]):
                for local_x in range(sub.shape[1]):
                    if sub[local_y, local_x] > 0:
                        try:
                            mat[local_x, local_y] = 1
                        except Exception:
                            pass
            try: mat.show()
            except Exception: pass

    def startup_animation(self):
        """Play a startup sequence on power-on."""
        logger.info("LED face: startup animation")
        for emotion in ["neutral", "curiosity", "joy", "neutral"]:
            self.express(emotion, transition_ms=200)
            time.sleep(0.4)

    def clear(self):
        """Turn off all LEDs."""
        canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
        with self._lock:
            self._canvas = canvas
        self._render(canvas)

    def start(self):
        self._running = True
        self.startup_animation()
        logger.info("LEDMatrixFace started")

    def stop(self):
        self._running = False
        self.clear()

    @property
    def current_emotion(self) -> str:
        return self._current_expr
