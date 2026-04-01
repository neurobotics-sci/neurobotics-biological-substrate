"""
bubo/face/eye_covers.py — Bubo v10000

Johnny 5 Style Servo Eye Covers
================================

INSPIRATION: Johnny 5, Short Circuit (1986)
  Designed by Syd Mead. The entire emotional life of Johnny 5 was
  conveyed through a deceptively simple mechanism: two metal flap
  "eyelids" over camera-like eyes. Small angular changes produced
  enormous shifts in apparent mood.

  From the prop archive: "black metal flaps that serve as upper and
  lower eyelids — small changes there totally change his mood."

BUBO IMPLEMENTATION:
  Two SG90 or MG90S micro-servos, one per eye cover.
  Each servo drives a lightweight anodised aluminium flap that
  covers the upper portion of the camera eye assembly.

  The flap angle encodes:
    0°   = fully raised (wide open — surprised, happy, alert)
    30°  = slightly raised (neutral-positive)
    45°  = neutral/resting
    70°  = slightly lowered (thoughtful, listening)
    90°  = half-closed (tired, sad, drowsy)
    120° = mostly closed with inward angle (suspicious, determined)
    150° = nearly closed (sleeping, shutdown)

  BILATERAL CONTROL:
    Both servos can move independently, enabling:
    - Symmetric expressions (both same angle)
    - Quizzical (one raised, one neutral)  ← Johnny 5's signature look
    - Winking (one closed, one open)
    - Anger (both angled inward from above — V-shape)

HARDWARE:
  Servo type:   SG90 9g micro-servo (or MG90S metal gear)
  Torque:       1.8 kg·cm (SG90) — sufficient for light aluminium flap
  Speed:        0.1s/60° at 4.8V
  Control:      PWM from STM32H7 co-processor or Jetson GPIO
                50Hz, pulse 500-2400μs
  Mounting:     Above each camera eye, angled flap pivots on M3 shaft
  Flap size:    ~40mm × 25mm × 1mm aluminium sheet
  Weight:       ~3g per flap — easily within servo torque budget

WIRING (to STM32H7 or Jetson GPIO):
  Left eye servo:  PWM → GPIO12 (or STM32 TIM1_CH1)
  Right eye servo: PWM → GPIO13 (or STM32 TIM1_CH2)
  VCC: 5V (servo rail, Galvanic Barrier protected)
  GND: Common ground

COORDINATE CONVENTION:
  Angles defined as eyelid depression from fully-open position.
  0° = fully open. 150° = nearly closed.
  Inward = inner edge lower than outer edge (anger geometry).
"""

import time
import threading
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger("EyeCovers")

try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    try:
        import Jetson.GPIO as GPIO
        HAS_GPIO = True
    except ImportError:
        HAS_GPIO = False
        logger.info("EyeCovers: GPIO not available — simulation mode")


# ── Servo configuration ───────────────────────────────────────────────────────

LEFT_EYE_PIN  = 12   # PWM GPIO pin
RIGHT_EYE_PIN = 13
PWM_FREQ      = 50   # Hz
MIN_PULSE     = 500  # μs (0°)
MAX_PULSE     = 2400 # μs (180°)


@dataclass
class EyeState:
    """Current state of both eye covers."""
    left_angle:   float = 45.0   # degrees (0=open, 150=closed)
    right_angle:  float = 45.0
    left_target:  float = 45.0
    right_target: float = 45.0


# ── Predefined eye cover positions per emotion ────────────────────────────────
# (left_angle, right_angle)
# Johnny 5 principle: asymmetry is character. Symmetry is mood.

EYE_POSITIONS = {
    # ── Wide open / positive ──────────────────────────────────────────────────
    "neutral":    (45, 45),    # resting, alert
    "joy":        (15, 15),    # wide, bright — Johnny 5 "happy" look
    "contentment":(55, 55),    # soft half-open — relaxed
    "pride":      (40, 40),    # slightly raised, confident

    # ── Surprise / fear / excitement ─────────────────────────────────────────
    "surprise":   (0, 0),      # maximally open — the Johnny 5 lightning moment
    "fear":       (5, 5),      # wide + slight upward tension
    "excitement": (0, 0),      # wide open + will flutter

    # ── Quizzical / curious ───────────────────────────────────────────────────
    "curiosity":  (10, 40),    # LEFT raised high, RIGHT neutral — SIGNATURE LOOK
    "confusion":  (35, 10),    # RIGHT raised, LEFT normal — other direction
    "empathy":    (30, 30),    # gently open, warm

    # ── Sad / tired ───────────────────────────────────────────────────────────
    "sadness":    (80, 80),    # drooping — the sad robot look
    "loneliness": (90, 90),    # heavy lids
    "shame":      (100, 95),   # downcast, slightly asymmetric

    # ── Anger / negative ─────────────────────────────────────────────────────
    "anger":      (30, 30),    # PLUS inward_angle=True for V-geometry
    "disgust":    (50, 70),    # asymmetric sneer geometry
    
    # ── Special states ───────────────────────────────────────────────────────
    "sleep":      (145, 145),  # nearly closed
    "shutdown":   (150, 150),  # fully closed
    "wink_left":  (15, 140),   # wink right eye
    "wink_right": (140, 15),   # wink left eye
    "thinking":   (35, 60),    # slight asymmetry, focused
}


class ServoController:
    """PWM servo controller — hardware or simulation."""

    def __init__(self, pin: int):
        self._pin = pin
        self._pwm = None
        self._angle = 90.0
        self._simulated = not HAS_GPIO
        if not self._simulated:
            self._init_hw()

    def _init_hw(self):
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._pin, GPIO.OUT)
            self._pwm = GPIO.PWM(self._pin, PWM_FREQ)
            self._pwm.start(self._angle_to_duty(90))
            logger.info(f"Servo on GPIO{self._pin} initialised")
        except Exception as e:
            logger.warning(f"Servo init failed: {e} — simulation")
            self._simulated = True

    def _angle_to_duty(self, angle: float) -> float:
        """Convert angle (0-180°) to PWM duty cycle."""
        pulse_us = MIN_PULSE + (angle / 180.0) * (MAX_PULSE - MIN_PULSE)
        return pulse_us / (1_000_000 / PWM_FREQ) * 100

    def set_angle(self, angle: float):
        angle = float(np.clip(angle, 0, 150))
        self._angle = angle
        if not self._simulated and self._pwm:
            try:
                self._pwm.ChangeDutyCycle(self._angle_to_duty(angle))
            except Exception as e:
                logger.debug(f"Servo set_angle: {e}")
        else:
            logger.debug(f"SIM servo GPIO{self._pin}: {angle:.1f}°")

    def stop(self):
        if self._pwm:
            try: self._pwm.stop()
            except Exception: pass

    @property
    def angle(self) -> float:
        return self._angle


class EyeCovers:
    """
    Johnny 5 style servo-driven eye cover controller.
    Integrates with EmotionChip via ZMQ subscriptions.
    """

    MOVE_SPEED = 60.0    # degrees per second
    FLUTTER_HZ = 8.0     # excitement flutter rate

    def __init__(self):
        self._left   = ServoController(LEFT_EYE_PIN)
        self._right  = ServoController(RIGHT_EYE_PIN)
        self._state  = EyeState()
        self._running= False
        self._lock   = threading.Lock()

    def express(self, emotion: str, intensity: float = 1.0,
                transition_s: float = 0.3):
        """
        Move eye covers to the position for this emotion.
        Intensity modulates how extreme the position is.
        """
        if emotion not in EYE_POSITIONS:
            emotion = "neutral"

        base_l, base_r = EYE_POSITIONS[emotion]

        # Scale toward neutral based on intensity
        neutral_l, neutral_r = EYE_POSITIONS["neutral"]
        target_l = neutral_l + (base_l - neutral_l) * intensity
        target_r = neutral_r + (base_r - neutral_r) * intensity

        with self._lock:
            self._state.left_target  = target_l
            self._state.right_target = target_r

        # Special handling for excitement — flutter
        if emotion == "excitement":
            threading.Thread(target=self._flutter, daemon=True).start()
        else:
            threading.Thread(target=self._smooth_move,
                             args=(target_l, target_r, transition_s),
                             daemon=True).start()

        logger.debug(f"EyeCovers: {emotion} → L:{target_l:.0f}° R:{target_r:.0f}°")

    def wink(self, side: str = "right", duration_s: float = 0.4):
        """Execute a wink on the specified side."""
        pos_key = f"wink_{side}"
        l, r = EYE_POSITIONS.get(pos_key, EYE_POSITIONS["neutral"])
        threading.Thread(
            target=self._wink_sequence,
            args=(l, r, duration_s),
            daemon=True
        ).start()

    def _wink_sequence(self, target_l, target_r, duration_s):
        curr_l = self._left.angle
        curr_r = self._right.angle
        self._smooth_move(target_l, target_r, 0.1)
        time.sleep(duration_s)
        self._smooth_move(curr_l, curr_r, 0.15)

    def _smooth_move(self, target_l: float, target_r: float,
                     duration_s: float):
        """Smooth interpolation to target positions."""
        steps = max(1, int(duration_s * 60))
        start_l = self._left.angle
        start_r = self._right.angle
        for i in range(steps + 1):
            alpha = i / steps
            # Ease in-out
            t = alpha * alpha * (3 - 2 * alpha)
            self._left.set_angle(start_l  + (target_l - start_l)  * t)
            self._right.set_angle(start_r + (target_r - start_r)  * t)
            time.sleep(duration_s / steps)

    def _flutter(self):
        """Excitement flutter — rapid open/close."""
        flutter_range = 20
        base_l, base_r = EYE_POSITIONS["excitement"]
        for _ in range(6):
            self._left.set_angle(base_l + flutter_range)
            self._right.set_angle(base_r + flutter_range)
            time.sleep(1 / (self.FLUTTER_HZ * 2))
            self._left.set_angle(base_l)
            self._right.set_angle(base_r)
            time.sleep(1 / (self.FLUTTER_HZ * 2))

    def set_thinking_mode(self):
        """Animated 'thinking' micro-movement — natural eye restlessness."""
        l, r = EYE_POSITIONS["thinking"]
        self._smooth_move(l, r, 0.5)
        time.sleep(0.8)
        self._smooth_move(l + 5, r - 5, 0.3)
        time.sleep(0.5)
        self._smooth_move(l, r, 0.4)

    def start(self):
        self._running = True
        # Startup: open eyes slowly
        self._smooth_move(150, 150, 0)  # start closed
        time.sleep(0.3)
        self._smooth_move(*EYE_POSITIONS["neutral"], 1.5)  # open slowly
        logger.info("EyeCovers started — Johnny 5 style initialised")

    def stop(self):
        self._running = False
        self._smooth_move(*EYE_POSITIONS["shutdown"], 0.5)
        time.sleep(0.6)
        self._left.stop()
        self._right.stop()

    @property
    def state(self) -> EyeState:
        with self._lock:
            return EyeState(
                left_angle=self._left.angle,
                right_angle=self._right.angle,
                left_target=self._state.left_target,
                right_target=self._state.right_target,
            )
