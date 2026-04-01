"""
bubo/safety/limp_mode.py — v11.14
Cerebellum Limp Mode — takes over partial motor logic when Spinal node unresponsive.

BIOLOGY:
  When the descending corticospinal tract is disrupted (stroke, cord lesion),
  the cerebellum and brainstem reticulospinal tract provide emergency motor
  output: stereotyped postural reflexes, weight-shifting, basic anti-gravity.
  This is why decerebrate animals still have motor tone — brainstem/cerebellar
  output bypasses cortex entirely.

BUBO IMPLEMENTATION:
  Trigger:  SPINAL_HB heartbeat gap > 150ms  (3 × 50ms nominal period)
  Response: Cerebellum publishes EFF_M1_ARM_L/R and EFF_M1_LEG_L/R directly,
            using its own DCN (deep cerebellar nuclei) output mapped via
            a pre-learned Jacobian approximation.
  Velocity: Capped at 30% of normal — "safe crawl" mode.
  Recovery: 5s stable SPINAL_HB → 3-step velocity ramp → normal handoff.

STATE MACHINE:
  NORMAL → (HB gap > 150ms) → LIMP_ENGAGING → (3 steps) → LIMP_ACTIVE
  LIMP_ACTIVE → (HB resumes, 5s stable) → RECOVERING → (ramp done) → NORMAL
"""

import time, logging, threading
import numpy as np
from collections import deque
from enum import Enum
from typing import Optional

logger = logging.getLogger("LimpMode")

# Heartbeat timing constants
HB_NOMINAL_MS  = 50.0    # SPINAL_HB published at 100Hz → 50ms period
HB_TIMEOUT_MS  = 150.0   # 3 missed heartbeats → engage limp mode
HB_STABLE_S    = 5.0     # seconds of stable HB before recovery
VELOCITY_CAP   = 0.30    # fraction of normal velocity in limp mode
RAMP_STEPS     = 3       # velocity ramp steps on recovery: 10%→20%→30%→full


class LimpState(Enum):
    NORMAL        = "normal"
    LIMP_ENGAGING = "limp_engaging"
    LIMP_ACTIVE   = "limp_active"
    RECOVERING    = "recovering"


class JacobianStore:
    """
    Stores a linearised Jacobian mapping DCN output → joint-space corrections.
    Updated online during normal operation; used during limp mode as fallback.

    Biological analogy: the cerebellum's internal body model —
    the "forward model" that predicts motor consequences without
    needing peripheral feedback.
    """
    DIM_DCN  = 32
    DIM_ARM  = 14   # 2×7 arm joints
    DIM_LEG  = 12   # 2×6 leg joints

    def __init__(self):
        rng = np.random.default_rng(7)
        # Initialise with small random Jacobians (will be refined online)
        self._J_arm = rng.standard_normal((self.DIM_ARM, self.DIM_DCN)) * 0.05
        self._J_leg = rng.standard_normal((self.DIM_LEG, self.DIM_DCN)) * 0.05
        self._update_count = 0
        self._alpha = 0.02   # online update learning rate

    def update(self, dcn_out: np.ndarray,
               arm_joints: np.ndarray, leg_joints: np.ndarray):
        """Update Jacobian estimate online (recursive least squares approx)."""
        dcn = np.resize(dcn_out, self.DIM_DCN)
        arm = np.resize(arm_joints, self.DIM_ARM)
        leg = np.resize(leg_joints, self.DIM_LEG)
        # Hebbian-style: Δ = α · (target - J·input) · input^T
        arm_pred = self._J_arm @ dcn
        leg_pred = self._J_leg @ dcn
        self._J_arm += self._alpha * np.outer(arm - arm_pred, dcn)
        self._J_leg += self._alpha * np.outer(leg - leg_pred, dcn)
        self._update_count += 1

    def map(self, dcn_out: np.ndarray, velocity_cap: float = 1.0):
        """Map DCN output → joint targets via learned Jacobian."""
        dcn = np.resize(dcn_out, self.DIM_DCN)
        arm = np.clip(self._J_arm @ dcn, -1.0, 1.0) * velocity_cap
        leg = np.clip(self._J_leg @ dcn, -1.0, 1.0) * velocity_cap
        return arm, leg

    @property
    def is_trained(self) -> bool:
        return self._update_count > 200   # ~2s at 100Hz


class LimpModeController:
    """
    Monitors spinal heartbeat and manages limp mode state machine.
    Instantiated inside CerebellumNode and called each control cycle.
    """

    def __init__(self, bus, jacobian_store: JacobianStore):
        self._bus      = bus
        self._jac      = jacobian_store
        self._state    = LimpState.NORMAL
        self._last_hb_ns = time.time_ns()
        self._stable_since: Optional[float] = None
        self._ramp_step  = 0
        self._velocity_cap = 1.0
        self._affected   = "none"
        self._lock       = threading.Lock()

        # HB arrival history (rolling 2s)
        self._hb_times: deque = deque(maxlen=200)

    def heartbeat_received(self, limb: str = "both"):
        """Called when SPINAL_HB message arrives."""
        with self._lock:
            self._last_hb_ns  = time.time_ns()
            self._hb_times.append(self._last_hb_ns)
            self._affected     = limb

    def step(self, dcn_out: np.ndarray,
             arm_joints: Optional[np.ndarray] = None,
             leg_joints: Optional[np.ndarray] = None) -> dict:
        """
        Called every cerebellum control cycle (100 Hz).
        Returns limp mode status and any direct motor commands.
        """
        with self._lock:
            now_ns    = time.time_ns()
            gap_ms    = (now_ns - self._last_hb_ns) / 1e6
            state     = self._state

        # Online Jacobian update during NORMAL operation
        if state == LimpState.NORMAL and arm_joints is not None:
            self._jac.update(dcn_out, arm_joints, leg_joints or np.zeros(12))

        # ── STATE TRANSITIONS ────────────────────────────────────────────
        if state == LimpState.NORMAL:
            if gap_ms > HB_TIMEOUT_MS:
                self._transition(LimpState.LIMP_ENGAGING)

        elif state == LimpState.LIMP_ENGAGING:
            if gap_ms <= HB_NOMINAL_MS * 2:
                self._transition(LimpState.NORMAL)   # recovered quickly
            else:
                self._transition(LimpState.LIMP_ACTIVE)

        elif state == LimpState.LIMP_ACTIVE:
            self._velocity_cap = VELOCITY_CAP
            if gap_ms <= HB_NOMINAL_MS * 2:
                # HB restored — start stable countdown
                if self._stable_since is None:
                    self._stable_since = time.time()
                elif time.time() - self._stable_since >= HB_STABLE_S:
                    self._transition(LimpState.RECOVERING)
            else:
                self._stable_since = None   # reset if HB drops again

        elif state == LimpState.RECOVERING:
            ramp_caps = [0.10, 0.20, 0.30, 1.0]
            if self._ramp_step < len(ramp_caps):
                self._velocity_cap = ramp_caps[self._ramp_step]
                self._ramp_step   += 1
            else:
                self._transition(LimpState.NORMAL)

        # ── GENERATE LIMP MOTOR COMMANDS ─────────────────────────────────
        limp_arm = limp_leg = None
        if self._state in (LimpState.LIMP_ACTIVE, LimpState.RECOVERING):
            if self._jac.is_trained:
                limp_arm, limp_leg = self._jac.map(dcn_out, self._velocity_cap)
            else:
                # Untrained: safe hold-current posture (zero delta)
                limp_arm = np.zeros(14)
                limp_leg = np.zeros(12)

        return {
            "state":         self._state.value,
            "active":        self._state != LimpState.NORMAL,
            "velocity_cap":  self._velocity_cap,
            "gap_ms":        gap_ms,
            "jacobian_ok":   self._jac.is_trained,
            "limp_arm_cmd":  limp_arm.tolist() if limp_arm is not None else None,
            "limp_leg_cmd":  limp_leg.tolist() if limp_leg is not None else None,
            "affected":      self._affected,
            "ramp_step":     self._ramp_step,
        }

    def _transition(self, new_state: LimpState):
        old = self._state
        self._state = new_state
        logger.warning(f"LimpMode: {old.value} → {new_state.value}")

        if new_state == LimpState.LIMP_ENGAGING:
            self._bus.publish(b"SFY_LIMP", {
                "state": "engaging", "gap_ms": (time.time_ns() - self._last_hb_ns) / 1e6,
                "affected": self._affected, "velocity_cap": VELOCITY_CAP,
                "message": "Spinal node unresponsive — cerebellum taking over motor",
            })
            logger.warning("⚠ LIMP MODE ENGAGING — spinal heartbeat lost")

        elif new_state == LimpState.LIMP_ACTIVE:
            self._ramp_step  = 0
            self._velocity_cap = VELOCITY_CAP
            self._bus.publish(b"SFY_LIMP", {
                "state": "active", "velocity_cap": VELOCITY_CAP,
                "jacobian_trained": self._jac.is_trained,
                "message": "LIMP MODE ACTIVE — cerebellar direct drive at 30%",
            })
            logger.warning("⚠ LIMP MODE ACTIVE — direct cerebellar motor drive")

        elif new_state == LimpState.RECOVERING:
            self._stable_since = None
            self._bus.publish(b"SFY_LIMP", {
                "state": "recovering", "ramp_steps": RAMP_STEPS,
                "message": "Spinal recovered — ramping velocity",
            })
            logger.info("LimpMode: recovering — velocity ramp started")

        elif new_state == LimpState.NORMAL:
            self._velocity_cap = 1.0
            self._ramp_step    = 0
            self._stable_since = None
            self._bus.publish(b"SFY_LIMP_CLR", {
                "state": "normal", "message": "Limp mode cleared — full spinal control restored",
            })
            logger.info("LimpMode: NORMAL restored")

    @property
    def is_active(self) -> bool:
        return self._state in (LimpState.LIMP_ACTIVE, LimpState.RECOVERING)

    @property
    def state(self) -> LimpState:
        return self._state
