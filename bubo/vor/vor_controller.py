"""
bubo/vor/vor_controller.py — Bubo v50.0
Vestibulo-Ocular Reflex (VOR) controller.

BIOLOGY:
  The VOR is the fastest reflex in the vertebrate body (7-10ms latency).
  Pathway: Semicircular canals → CN VIII (vestibular nerve) →
           Medial vestibular nucleus (MVN) → Abducens nucleus (CN VI) →
           Lateral rectus muscle (horizontal) / CN III → medial rectus

  The compensatory eye movement: θ_eye = -G_VOR · θ_head
  where G_VOR ≈ 0.95 (biological) — eyes move 95% of head displacement

  Three canals per side (orthogonal):
    Horizontal canal (HC): detects yaw — drives horizontal VOR
    Anterior canal (AC):   detects pitch/roll — drives vertical/torsional VOR
    Posterior canal (PC):  detects roll/pitch — drives torsional/vertical VOR

  VOR suppression:
    During voluntary saccades: VOR is suppressed (~100ms window)
    During near-target fixation: VOR gain reduced (vergence-VOR interaction)
    Cerebellar adaptation: VOR gain can be modified over days

IMPLEMENTATION:
  Runs as a tight loop at 200Hz (5ms period) — faster than biological VOR.
  Input:  IMU gyroscope readings (rad/s) — from BeagleBoard MPU-6050
  Output: Eye servo PWM commands (T.EFF_EYE_L, T.EFF_EYE_R)

  Servo mapping:
    2 servos per eye (horizontal + vertical)
    Horizontal servo range: ±45° → PWM 1000-2000μs
    Vertical servo range:   ±35° → PWM 1100-1900μs

  VOR + OKR integration:
    θ_eye_total = -G_VOR·θ_head + G_OKR·optic_flow
    At low freq (< 0.5Hz): OKR dominates (visual stabilisation)
    At high freq (> 2Hz):  VOR dominates (inertial stabilisation)

CEREBELLAR ADAPTATION:
  VOR gain is not fixed. The cerebellum learns to adjust G_VOR to minimise
  retinal slip (motion on the retina despite VOR).
  Error signal: visual motion during head movement → climbing fibre error
  Adaptation: Purkinje cells adjust flocculus output → VOR gain change
  Implemented: simple integrating gain controller (not full Purkinje model)
"""

import time, threading, logging
import numpy as np
from collections import deque
from bubo.shared.bus.neural_bus import NeuralBus, T

logger = logging.getLogger("VOR")

VOR_GAIN_DEFAULT   = 0.95    # biological VOR gain
VOR_ADAPT_RATE     = 0.001   # cerebellar adaptation learning rate
VOR_SUPPRESS_MS    = 120.0   # suppress VOR for 120ms during saccades
EYE_H_MAX_DEG      = 45.0    # horizontal eye range
EYE_V_MAX_DEG      = 35.0    # vertical eye range
PWM_CENTER         = 1500    # neutral servo pulse (μs)
PWM_H_SCALE        = float(500 / EYE_H_MAX_DEG)   # μs per degree horizontal
PWM_V_SCALE        = float(400 / EYE_V_MAX_DEG)   # μs per degree vertical


class EyePositionIntegrator:
    """
    Integrates VOR commands and enforces eye position limits.
    Models the neural integrator in the brainstem that converts
    velocity commands to position commands for the eye muscles.
    Leak time constant τ ≈ 25s (eyes drift back to centre slowly).
    """
    TAU_S = 25.0

    def __init__(self):
        self._h_deg = 0.0   # current horizontal position (degrees)
        self._v_deg = 0.0   # current vertical position
        self._t = time.time()

    def step(self, h_vel_dps: float, v_vel_dps: float) -> tuple:
        """
        h_vel_dps: horizontal eye velocity command (°/s)
        v_vel_dps: vertical eye velocity command (°/s)
        Returns: (h_deg, v_deg) current eye position
        """
        t = time.time(); dt = max(t - self._t, 0.001); self._t = t
        # Leaky integration (neural integrator leak)
        self._h_deg = self._h_deg * np.exp(-dt / self.TAU_S) + h_vel_dps * dt
        self._v_deg = self._v_deg * np.exp(-dt / self.TAU_S) + v_vel_dps * dt
        # Eye position limits
        self._h_deg = float(np.clip(self._h_deg, -EYE_H_MAX_DEG, EYE_H_MAX_DEG))
        self._v_deg = float(np.clip(self._v_deg, -EYE_V_MAX_DEG, EYE_V_MAX_DEG))
        return self._h_deg, self._v_deg

    def reset_to_centre(self):
        self._h_deg = 0.0; self._v_deg = 0.0


def deg_to_pwm(deg_h: float, deg_v: float) -> tuple:
    """Convert eye position (degrees) to servo PWM (microseconds)."""
    pwm_h = int(np.clip(PWM_CENTER + deg_h * PWM_H_SCALE, 1000, 2000))
    pwm_v = int(np.clip(PWM_CENTER - deg_v * PWM_V_SCALE, 1100, 1900))  # inverted vertical
    return pwm_h, pwm_v


class VORController:
    """
    Main VOR controller.
    Receives vestibular data (gyroscope) and generates compensatory
    eye movement commands at 200Hz.
    """
    HZ = 200

    def __init__(self, bus: NeuralBus):
        self._bus     = bus
        self._gain    = VOR_GAIN_DEFAULT
        self._eye_l   = EyePositionIntegrator()
        self._eye_r   = EyePositionIntegrator()

        # VOR suppression (during saccades)
        self._suppressed = False
        self._suppress_until_ns = 0

        # Gyroscope state (from vestibular node)
        self._gyro_rps = np.zeros(3)   # [pitch, roll, yaw] rad/s
        self._lock = threading.Lock()

        # OKR gain (optic flow contribution)
        self._okr_h = 0.0; self._okr_v = 0.0

        # Retinal slip history (for cerebellar gain adaptation)
        self._slip_history = deque(maxlen=100)

        # Running flag
        self._running = False

    def update_vestibular(self, gyro_rps: list):
        """Called when vestibular data arrives (from auditory/IMU node)."""
        with self._lock:
            self._gyro_rps = np.array(gyro_rps, dtype=float)

    def update_optic_flow(self, h_flow: float, v_flow: float):
        """OKR contribution from visual cortex MT motion field."""
        self._okr_h = float(np.clip(h_flow * 0.3, -10, 10))  # deg/s
        self._okr_v = float(np.clip(v_flow * 0.3, -10, 10))

    def suppress(self, duration_ms: float = VOR_SUPPRESS_MS):
        """Suppress VOR during intentional saccade."""
        self._suppressed = True
        self._suppress_until_ns = time.time_ns() + int(duration_ms * 1e6)

    def adapt_gain(self, retinal_slip_dps: float):
        """
        Cerebellar adaptation: adjust VOR gain based on retinal slip.
        If slip is in same direction as head movement → gain too low → increase
        If slip is opposite to head movement → gain too high → decrease
        """
        self._slip_history.append(retinal_slip_dps)
        if len(self._slip_history) > 20:
            mean_slip = float(np.mean(self._slip_history))
            # Simple P-controller on gain
            self._gain = float(np.clip(
                self._gain - VOR_ADAPT_RATE * mean_slip,
                0.5, 1.3))

    def _vor_loop(self):
        interval = 1.0 / self.HZ
        while self._running:
            t0 = time.time()

            # Check suppression
            if self._suppressed:
                if time.time_ns() > self._suppress_until_ns:
                    self._suppressed = False
                else:
                    time.sleep(interval); continue

            with self._lock:
                gyro = self._gyro_rps.copy()

            # VOR velocity commands (opposite to head motion, scaled by gain)
            # gyro_rps: [gx=pitch, gy=roll, gz=yaw]
            # Horizontal VOR: driven by yaw (gz)
            # Vertical VOR:   driven by pitch (gx)
            gz_dps = float(np.degrees(gyro[2]))   # yaw → horizontal
            gx_dps = float(np.degrees(gyro[0]))   # pitch → vertical

            # Compensatory velocity (negative: eye moves opposite to head)
            vor_h_vel = -self._gain * gz_dps + self._okr_h
            vor_v_vel = -self._gain * gx_dps + self._okr_v

            # Integrate to position
            h_l, v_l = self._eye_l.step(vor_h_vel, vor_v_vel)
            h_r, v_r = self._eye_r.step(vor_h_vel, vor_v_vel)
            # Conjugate: both eyes move together (version movement)
            # Small vergence correction could be added here

            # Convert to PWM
            pwm_h_l, pwm_v_l = deg_to_pwm(h_l, v_l)
            pwm_h_r, pwm_v_r = deg_to_pwm(h_r, v_r)

            ns = time.time_ns()
            self._bus.publish(T.EFF_EYE_L, {
                "h_deg": h_l, "v_deg": v_l,
                "pwm_h": pwm_h_l, "pwm_v": pwm_v_l,
                "source": "VOR", "gain": self._gain,
                "timestamp_ns": ns,
            })
            self._bus.publish(T.EFF_EYE_R, {
                "h_deg": h_r, "v_deg": v_r,
                "pwm_h": pwm_h_r, "pwm_v": pwm_v_r,
                "source": "VOR", "gain": self._gain,
                "timestamp_ns": ns,
            })
            self._bus.publish(T.VOR_CMD, {
                "vor_h_vel_dps": vor_h_vel, "vor_v_vel_dps": vor_v_vel,
                "gain": self._gain, "suppressed": False,
                "gyro_yaw_dps": gz_dps, "gyro_pitch_dps": gx_dps,
                "okr_h": self._okr_h, "okr_v": self._okr_v,
                "timestamp_ns": ns,
            })

            time.sleep(max(0, interval - (time.time() - t0)))

    def start(self):
        self._running = True
        threading.Thread(target=self._vor_loop, daemon=True).start()
        logger.info(f"VOR v50.0 | gain={self._gain} | {self.HZ}Hz | OKR-coupled")

    def stop(self): self._running = False
