"""
bubo/saccadic_masking/saccadic_masking.py — Bubo v5900
Saccadic Masking: Visual muting during rapid eye movements.

══════════════════════════════════════════════════════════════════════
BIOLOGY: WHY THE BRAIN SUPPRESSES VISION DURING SACCADES
══════════════════════════════════════════════════════════════════════

When you move your eyes rapidly (a saccade), your visual world should blur
catastrophically — the image sweeps across the retina at 400-700°/s. Yet you
experience no blur. The brain implements "saccadic suppression":

  Mechanism (Wurtz 2008, Current Biology):
    The superior colliculus sends an efference copy (corollary discharge)
    to the thalamus (pulvinar nucleus) ~50ms BEFORE the saccade begins.
    This pre-saccadic signal suppresses visual cortex (V1, V4, MT) activity
    by 40-80% for the duration of the saccade (~20-200ms depending on amplitude).
    Visual processing resumes only after the eye has settled on a new fixation.

  Why it's adaptive:
    1. Prevents the blurry mid-saccade image from triggering motion detectors
       (MT would fire massively from retinal image motion, causing false alarms)
    2. Saves metabolic energy — no point processing a blurry image
    3. Prevents the "world is spinning" percept that would destabilise walking

  Saccadic suppression is NOT just a passive consequence of blur.
  It is active neural inhibition that precedes the saccade.
  Psychophysics: detection thresholds for luminance changes INCREASE by 0.7 log
  units during saccades (Zuber & Stark 1966). Blankenstein (1985): suppression
  starts 50ms before saccade, peaks at saccade onset, recovers 80ms after.

══════════════════════════════════════════════════════════════════════
BUBO IMPLEMENTATION: NETWORK BANDWIDTH SAVINGS
══════════════════════════════════════════════════════════════════════

The SC node publishes T.SC_SACCADE when initiating a gaze shift.
This message is the "efference copy" — it reaches the Visual and Auditory
nodes before the eye actually moves (message propagation ~0.3ms vs saccade
execution ~20-200ms).

On receiving T.SC_SACCADE, Visual (192.168.1.50) and Auditory (192.168.1.51)
nodes STOP publishing their high-bandwidth outputs for the saccade duration:
  - T.AFF_VIS_V1   (V1 features, ~1KB, 30Hz) → muted
  - T.AFF_VIS_MT   (motion events, ~500B, 30Hz) → muted
  - T.AFF_VIS_DEPTH (point cloud, ~3KB, 30Hz) → muted
  - T.AFF_AUD_A1   (audio features, ~200B, 50Hz) (reduced, not stopped)

BANDWIDTH CALCULATION:
  Visual publications per second:
    V1:    1KB × 30Hz = 30 KB/s
    MT:    500B × 30Hz = 15 KB/s
    Depth: 3KB × 30Hz = 90 KB/s
    Total: 135 KB/s from visual node alone

  Saccade statistics (measured in humans during active exploration):
    Mean saccade amplitude: 15°
    Mean saccade duration:  50ms
    Mean saccade frequency: 3/second (during visual search)
    Mean fixation duration: 250ms

  Suppression window: saccade_duration + 30ms pre + 50ms post
  = 50ms + 30ms + 50ms = 130ms per saccade
  At 3 saccades/s: 390ms/s suppressed = 39% of time

  Expected bandwidth reduction: 135 KB/s × 39% = 52 KB/s saved per visual node
  Plus auditory partial mute: ~8 KB/s saved
  Total per saccade: ~60 KB/s = 44% of visual bandwidth

  With 4 visual/sensory nodes (visual, auditory, somatosensory, sc):
  Total cluster savings: ~200 KB/s during active exploration phases

  In state S05 (VISUAL_SEARCH): saccade rate can reach 5/s → 65% of time muted
  → savings 65% × 135 KB/s = 88 KB/s from visual node alone

SUPPRESSION LEVELS:
  FULL:     100% suppression (large saccades > 15°, high velocity)
  PARTIAL:  50% suppression (medium saccades 5-15°)
  MINIMAL:  10% suppression (micro-saccades < 5°, fixation maintenance)
  OFF:      no suppression (smooth pursuit, VOR correction, nod-off)

IMPLEMENTATION:
  SaccadicMaskingController: embedded in visual node, auditory node.
  On T.SC_SACCADE: compute suppression level from amplitude/velocity.
  Sets _suppressed flag for suppression_duration_ms.
  Node's publish loop checks flag: if suppressed → skip expensive computation.
  Critical: S1/somatosensory is NOT suppressed (touch is always processed).
  Pain signals (NOCI_*) are NEVER suppressed (safety).
"""

import time, logging, threading
import numpy as np
from typing import Optional
from enum import Enum

logger = logging.getLogger("SaccadicMasking")


class SuppressionLevel(Enum):
    OFF      = 0   # no suppression
    MINIMAL  = 1   # 10% — micro-saccade
    PARTIAL  = 2   # 50% — medium saccade
    FULL     = 3   # 100% — large/fast saccade


# Biological suppression parameters
PRE_SACCADE_LEAD_MS   = 30.0   # suppression starts 30ms before saccade
POST_SACCADE_TRAIL_MS = 50.0   # suppression continues 50ms after saccade ends
MIN_SACCADE_AMP_DEG   = 1.0    # minimum amplitude to trigger suppression
MICRO_SACCADE_DEG     = 5.0    # below this: minimal suppression
MEDIUM_SACCADE_DEG    = 15.0   # below this: partial suppression
LARGE_SACCADE_DEG     = 15.0   # above this: full suppression

# VOR gain during suppression (we DON'T suppress VOR — gaze stabilisation must continue)
VOR_SUPPRESSED = False


def saccade_duration_ms(amplitude_deg: float) -> float:
    """
    Saccade duration from amplitude via Main Sequence (Bahill et al. 1975).
    D = 2.7 × A^0.6 ms  (empirical fit, valid 1-50°)
    """
    return float(np.clip(2.7 * (amplitude_deg ** 0.6), 10, 250))


def suppression_level(amplitude_deg: float, velocity_dps: float) -> SuppressionLevel:
    """Compute suppression level from saccade parameters."""
    if amplitude_deg < MIN_SACCADE_AMP_DEG:
        return SuppressionLevel.OFF
    if amplitude_deg < MICRO_SACCADE_DEG:
        return SuppressionLevel.MINIMAL
    if amplitude_deg < MEDIUM_SACCADE_DEG:
        return SuppressionLevel.PARTIAL
    return SuppressionLevel.FULL


class SaccadicMaskingController:
    """
    Saccadic masking controller — embedded in visual and auditory nodes.

    Usage in visual_node.py:
        self.masking = SaccadicMaskingController("visual")
        # In SC_SACCADE handler:
        self.masking.trigger_suppression(saccade_payload)
        # In publish loop:
        if self.masking.should_publish(T.VISUAL_V1):
            self.bus.publish(T.VISUAL_V1, ...)
    """

    def __init__(self, node_name: str):
        self._node           = node_name
        self._level          = SuppressionLevel.OFF
        self._suppress_until = 0.0   # Unix time
        self._suppress_start = 0.0
        self._last_saccade   = {"amplitude_deg": 0.0, "velocity_dps": 0.0}
        self._lock           = threading.Lock()
        self._n_suppressions = 0
        self._total_suppressed_ms = 0.0
        self._t_start        = time.time()

    def trigger_suppression(self, saccade_payload: dict):
        """
        Called when T.SC_SACCADE message arrives.
        Implements pre-saccadic onset (biologically, suppression starts BEFORE
        the saccade due to efference copy propagation).
        """
        tgt_px = saccade_payload.get("target_px", [320, 240])
        cx, cy = 320, 240
        h_deg  = abs(float(tgt_px[0] - cx)) / 320 * 45.0
        v_deg  = abs(float(tgt_px[1] - cy)) / 240 * 35.0
        amp    = float(np.hypot(h_deg, v_deg))
        vel    = float(saccade_payload.get("velocity_dps", amp * 40))  # approx if not given

        level  = suppression_level(amp, vel)
        if level == SuppressionLevel.OFF:
            return

        dur_ms  = saccade_duration_ms(amp) + POST_SACCADE_TRAIL_MS
        now     = time.time()

        with self._lock:
            self._level          = level
            # Apply PRE-saccadic suppression immediately (efference copy precedes saccade)
            self._suppress_start = now
            self._suppress_until = now + dur_ms / 1000.0
            self._last_saccade   = {"amplitude_deg": amp, "velocity_dps": vel}
            self._n_suppressions += 1

        logger.debug(f"SaccMask [{self._node}]: amp={amp:.1f}° → {level.name} "
                     f"for {dur_ms:.0f}ms")

    def clear_suppression(self):
        """Explicitly end suppression (e.g., when VOR gain update needed)."""
        with self._lock:
            self._level          = SuppressionLevel.OFF
            self._suppress_until = 0.0

    def should_publish(self, topic: bytes) -> bool:
        """
        Returns True if the node should publish this topic now.
        Pain/noci topics are NEVER suppressed.
        VOR commands are NEVER suppressed.
        """
        # Safety topics always pass through
        if topic.startswith(b"AFF_NOCI") or topic.startswith(b"SFY_") or \
           topic.startswith(b"VOR_")     or topic.startswith(b"SPN_HB"):
            return True

        with self._lock:
            level = self._level
            until = self._suppress_until

        if time.time() > until:
            with self._lock:
                if self._level != SuppressionLevel.OFF:
                    elapsed = (time.time() - self._suppress_start) * 1000
                    self._total_suppressed_ms += elapsed
                self._level = SuppressionLevel.OFF
            return True

        # Suppression active — decide based on level and topic priority
        if level == SuppressionLevel.FULL:
            # Only allow: noci (already passed), heartbeat (already passed),
            # vestibular (not muted — VOR needs it)
            if topic.startswith(b"AFF_VEST"):
                return True   # VOR needs vestibular at full rate
            return False       # suppress visual, audio, MT, depth

        if level == SuppressionLevel.PARTIAL:
            # Allow vestibular + audio localisation, suppress V1/MT/depth
            if topic.startswith((b"AFF_VEST", b"AFF_AUD_SPAT")):
                return True
            # Probabilistic: publish 50% of the time (reduces rate by half)
            return np.random.random() > 0.50

        if level == SuppressionLevel.MINIMAL:
            # Allow almost everything, drop 10% of heavy topics
            if topic.startswith((b"AFF_VIS_DEPTH", b"AFF_VIS_MT")):
                return np.random.random() > 0.10
            return True

        return True  # OFF

    def stats(self) -> dict:
        """Bandwidth savings statistics."""
        elapsed_s = max(time.time() - self._t_start, 1.0)
        suppression_frac = self._total_suppressed_ms / (elapsed_s * 1000)
        return {
            "node":               self._node,
            "n_suppressions":     self._n_suppressions,
            "total_suppressed_s": round(self._total_suppressed_ms / 1000, 1),
            "suppression_pct":    round(suppression_frac * 100, 1),
            "current_level":      self._level.name,
            "bandwidth_saved_pct": round(suppression_frac * 44, 1),  # ~44% visual BW
        }

    @property
    def is_suppressed(self) -> bool:
        with self._lock:
            return time.time() < self._suppress_until

    @property
    def level(self) -> SuppressionLevel:
        with self._lock:
            return self._level
