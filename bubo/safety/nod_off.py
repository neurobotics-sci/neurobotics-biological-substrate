"""
bubo/safety/nod_off.py — v11.14
Nodding-off / Microsleep Safety Protocol

BIOLOGY:
  Sleep pressure (adenosine) accumulates during sustained waking.
  When it crosses a threshold the brain enters NREM1 transitional sleep:
  - Alpha spindles appear in EEG
  - Muscle tone drops (head nods)
  - Blink rate increases, eye movements slow
  - Reaction time increases 3-10×
  - Brief microsleeps (0.5–30s) intrude into consciousness

  Crucially: the brain does NOT simply switch off. It performs graduated
  vigilance reduction while maintaining arousal circuits on standby.
  Any threat (pain, loud noise, imbalance) triggers immediate re-arousal
  within 50–200ms — faster than a voluntary response.

PRIORITY ORDER (hard-coded, cannot be overridden by cortex):
  1. Physical safety — ZMP stable, no looming, no active grip
  2. Self-preservation — battery, thermal, posture
  3. Function — conversation, task completion

STATE MACHINE:
  AWAKE
    → adenosine > 0.65 AND fear < 0.30 AND DA < 0.35
      → PRE_SLEEP (safety checks, balance lock)
        → all checks pass → MICROSLEEP
          → timeout OR wake trigger → WAKING → AWAKE
        → any check fails → AWAKE (abort)

WAKE TRIGGERS (immediate, any one sufficient):
  - Nociceptive input (pain > 0.15)
  - Loud audio event (rms > threshold)
  - ZMP perturbation (jerk_mag > 0.2 g)
  - Fear signal spike (CEA > 0.30)
  - Battery critical (< 5%)
  - Looming object detected by SC
  - External interrupt (SFY_FREEZE broadcast)
"""

import time, logging, threading
import numpy as np
from enum import Enum
from typing import Optional

logger = logging.getLogger("NodOff")

ADENOSINE_THRESH  = 0.65
FEAR_MAX          = 0.30
DA_MAX_NOD        = 0.35
MICROSLEEP_MIN_S  = 5.0
MICROSLEEP_MAX_S  = 30.0
WAKE_PAIN_THRESH  = 0.15
WAKE_JERK_THRESH  = 0.20   # g
WAKE_FEAR_THRESH  = 0.30
SAFETY_BALANCE_POSTURE = {"hip_flex": 0.02, "knee_flex": 0.03, "ankle_df": -0.02}


class NodState(Enum):
    AWAKE      = "awake"
    PRE_SLEEP  = "pre_sleep"    # safety checks running
    MICROSLEEP = "microsleep"   # eyes closed, motor drive 20%
    WAKING     = "waking"       # arousal ramp, 500ms


class NodOffController:
    """
    Safety-first microsleep controller.
    Instantiated inside InsulaNode (interoception hub).
    """

    def __init__(self, bus):
        self._bus          = bus
        self._state        = NodState.AWAKE
        self._sleep_t      = 0.0
        self._duration     = 0.0
        self._wake_reason  = ""
        self._lock         = threading.Lock()

        # Live safety signals (updated by insula handlers)
        self._zmp_ok       = True
        self._looming      = False
        self._gripping     = False
        self._fear         = 0.0
        self._jerk_g       = 0.0
        self._battery_frac = 1.0
        self._pain         = 0.0

    # ── Public update methods (called from insula handlers) ─────────────

    def update_safety(self, zmp_ok: bool, looming: bool, gripping: bool,
                      fear: float, jerk_g: float, battery_frac: float, pain: float):
        with self._lock:
            self._zmp_ok       = zmp_ok
            self._looming      = looming
            self._gripping     = gripping
            self._fear         = fear
            self._jerk_g       = jerk_g
            self._battery_frac = battery_frac
            self._pain         = pain

        # Immediate wake triggers — do not wait for step()
        if self._state == NodState.MICROSLEEP:
            if pain > WAKE_PAIN_THRESH:
                self._wake("pain_noci")
            elif jerk_g > WAKE_JERK_THRESH:
                self._wake("balance_perturbation")
            elif fear > WAKE_FEAR_THRESH:
                self._wake("fear_spike")
            elif battery_frac < 0.05:
                self._wake("battery_critical")
            elif looming:
                self._wake("looming_object")

    def external_interrupt(self):
        """Force wake from any external safety broadcast."""
        self._wake("external_interrupt")

    # ── Main evaluation (called from insula control loop ~10Hz) ─────────

    def step(self, adenosine: float, da: float) -> dict:
        """Advance state machine. Returns current nod-off state dict."""
        with self._lock:
            state = self._state
            zmp   = self._zmp_ok
            loom  = self._looming
            grip  = self._gripping
            fear  = self._fear
            batt  = self._battery_frac
            pain  = self._pain

        if state == NodState.AWAKE:
            if (adenosine > ADENOSINE_THRESH and
                    fear < FEAR_MAX and da < DA_MAX_NOD):
                self._enter_pre_sleep()

        elif state == NodState.PRE_SLEEP:
            # SAFETY CHECK (priority 1): physical safety
            if not zmp:
                self._abort("zmp_unstable")
            elif loom:
                self._abort("looming")
            elif grip:
                self._abort("active_grip")
            # SELF-PRESERVATION (priority 2)
            elif batt < 0.10:
                self._abort("battery_low")
            elif pain > WAKE_PAIN_THRESH:
                self._abort("pain")
            else:
                self._enter_microsleep()

        elif state == NodState.MICROSLEEP:
            elapsed = time.time() - self._sleep_t
            if elapsed >= self._duration:
                self._wake("duration_complete")

        elif state == NodState.WAKING:
            # 500ms arousal ramp, then back to AWAKE
            if time.time() - self._sleep_t > 0.5:
                with self._lock:
                    self._state = NodState.AWAKE
                self._bus.publish(b"SFY_NODOFF", {
                    "event": "fully_awake", "reason": self._wake_reason})

        return {
            "state":          self._state.value,
            "is_sleeping":    self._state == NodState.MICROSLEEP,
            "motor_scale":    0.20 if self._state == NodState.MICROSLEEP else 1.0,
            "sc_mode":        "peripheral_wide" if self._state == NodState.MICROSLEEP else "normal",
            "adenosine":      adenosine,
            "da":             da,
        }

    # ── State transitions ────────────────────────────────────────────────

    def _enter_pre_sleep(self):
        with self._lock:
            self._state = NodState.PRE_SLEEP
        logger.info("NodOff: PRE_SLEEP — running safety checks")
        # SAFETY FIRST: lock balance posture before reducing vigilance
        self._bus.publish(b"SFY_FREEZE", {
            "type":    "nod_off_prep",
            "action":  "balance_lock",
            "posture": SAFETY_BALANCE_POSTURE,
            "message": "Pre-sleep balance lock engaged",
        })

    def _enter_microsleep(self):
        duration = float(np.random.uniform(MICROSLEEP_MIN_S, MICROSLEEP_MAX_S))
        with self._lock:
            self._state    = NodState.MICROSLEEP
            self._sleep_t  = time.time()
            self._duration = duration
        self._bus.publish(b"SFY_NODOFF", {
            "event":       "microsleep_start",
            "duration_s":  duration,
            "motor_scale": 0.20,
            "sc_mode":     "peripheral_wide",   # SC monitors wide field, not foveal
            "s1_active":   True,                # tactile monitoring stays full
        })
        logger.info(f"NodOff: MICROSLEEP for {duration:.1f}s")

    def _abort(self, reason: str):
        """Safety check failed — do not enter microsleep."""
        with self._lock:
            self._state = NodState.AWAKE
        logger.info(f"NodOff: PRE_SLEEP aborted ({reason}) — staying awake")
        self._bus.publish(b"SFY_NODOFF", {
            "event": "sleep_aborted", "reason": reason})

    def _wake(self, reason: str):
        with self._lock:
            if self._state not in (NodState.MICROSLEEP, NodState.PRE_SLEEP):
                return
            self._state     = NodState.WAKING
            self._sleep_t   = time.time()
            self._wake_reason = reason
        self._bus.publish(b"SFY_NODOFF", {
            "event":  "waking",
            "reason": reason,
            "ramp_ms": 500,
        })
        logger.info(f"NodOff: WAKING ({reason})")

    @property
    def is_sleeping(self) -> bool:
        return self._state == NodState.MICROSLEEP

    @property
    def state(self) -> NodState:
        return self._state
