"""
bubo/reflexes/reflex_coordinator.py — v50.0
Unified reflex arc coordinator: ASR, TLR, ATNR, Grasp, Moro, OKR, PLR.

Each reflex is a self-contained class with:
  - evaluate(): returns reflex command or None if threshold not met
  - latency_ms: biological latency budget
  - refractory_ms: minimum interval between triggers

The ReflexCoordinator runs as a sub-system inside the spinal nodes
and the oculomotor node, checked at the appropriate frequency.
"""
import time, logging
import numpy as np
from typing import Optional
from bubo.shared.bus.neural_bus import NeuralBus, T

logger = logging.getLogger("ReflexCoordinator")


class AcousticStartleReflex:
    """
    ASR: Loud sound → whole-body flinch < 30ms.
    Prepulse inhibition (PPI): quiet tone 30-500ms before → reduced amplitude.

    Biology: Cochlear nucleus → nucleus reticularis pontis caudalis (PnC) →
             reticulospinal tract → motor neurons.
    Clinical significance: PPI deficit models schizophrenia attentional gating.
    """
    THRESHOLD_DB   = -25.0   # dB RMS above which ASR triggers
    REFRACTORY_MS  = 1000.0  # 1s refractory period
    LATENCY_MS     = 30.0
    PPI_WINDOW_MS  = 500.0   # prepulse inhibition window
    PPI_REDUCTION  = 0.60    # 60% amplitude reduction with prepulse

    def __init__(self):
        self._last_trigger_ms = 0.0
        self._prepulse_t_ms   = 0.0
        self._prepulse_active = False

    def prepulse(self, quiet_tone: bool):
        """Register a prepulse (quiet tone that would inhibit startle)."""
        if quiet_tone:
            self._prepulse_t_ms   = time.time() * 1000
            self._prepulse_active = True

    def evaluate(self, audio_rms_db: float) -> Optional[dict]:
        now_ms = time.time() * 1000
        if now_ms - self._last_trigger_ms < self.REFRACTORY_MS: return None
        if audio_rms_db < self.THRESHOLD_DB: return None

        amplitude = 1.0
        # Check prepulse inhibition
        if self._prepulse_active:
            dt = now_ms - self._prepulse_t_ms
            if 30 < dt < self.PPI_WINDOW_MS:
                amplitude *= (1.0 - self.PPI_REDUCTION)
                logger.debug(f"ASR PPI: amplitude reduced to {amplitude:.2f}")

        self._last_trigger_ms = now_ms; self._prepulse_active = False
        return {
            "reflex": "ASR", "amplitude": amplitude,
            "pattern": {"arm_l": [-0.3*amplitude, 0.2*amplitude, 0, 0.5*amplitude, 0, 0, 0],
                        "arm_r": [-0.3*amplitude, -0.2*amplitude, 0, 0.5*amplitude, 0, 0, 0],
                        "neck_flex": 0.2*amplitude},
            "duration_ms": 150.0, "latency_ms": self.LATENCY_MS,
        }


class TonicLabyrinthineReflex:
    """
    TLR: Head position → modulates limb extensor tone.
    Head pitched forward (flexion) → limb flexion tone increases.
    Head pitched backward (extension) → limb extension tone increases.

    Biology: Otolith organs → vestibular nuclei → reticulospinal → motor neurons.
    Used in Bubo for automatic postural adaptation on inclines.
    """
    PITCH_THRESH_DEG = 20.0
    GAIN = 0.30

    def evaluate(self, pitch_deg: float) -> Optional[dict]:
        if abs(pitch_deg) < self.PITCH_THRESH_DEG: return None
        # Normalised pitch beyond threshold
        excess = (abs(pitch_deg) - self.PITCH_THRESH_DEG) / 60.0
        sign   = np.sign(pitch_deg)
        gain   = float(np.clip(self.GAIN * excess, 0, 0.5))
        return {
            "reflex": "TLR",
            "pitch_deg": pitch_deg,
            "extensor_bias": float(-sign * gain),   # + = extension, - = flexion
            "cpg_mod": {"extensor_gain": float(1.0 + sign * gain * 0.5)},
        }


class AsymmetricTonicNeckReflex:
    """
    ATNR: Head rotation → ipsilateral arm extension, contralateral flexion.
    Fencer posture: turn right → right arm extends (ready to reach right).

    Biology: Neck proprioceptors → C1-C4 interneurons → motor neurons.
    Useful in Bubo: pre-positions reaching arm in direction of gaze.
    """
    YAW_THRESH_DEG = 15.0
    GAIN = 0.25

    def evaluate(self, yaw_deg: float) -> Optional[dict]:
        if abs(yaw_deg) < self.YAW_THRESH_DEG: return None
        excess = (abs(yaw_deg) - self.YAW_THRESH_DEG) / 60.0
        g = float(np.clip(self.GAIN * excess, 0, 0.4))
        sign = np.sign(yaw_deg)  # positive = turn right
        return {
            "reflex": "ATNR",
            "yaw_deg": yaw_deg,
            # Ipsilateral (same side as turn): extend
            # Contralateral (opposite): flex
            "arm_r_delta": [float(sign * g * 0.4), 0, 0, float(-sign * g * 0.3), 0, 0, 0],
            "arm_l_delta": [float(-sign * g * 0.4), 0, 0, float(sign * g * 0.3), 0, 0, 0],
        }


class PalmarGraspReflex:
    """
    Grasp reflex: Pressure on palm → finger flexion (palmar grasp).
    Useful for catching objects and maintaining grip under unexpected load.

    Biology: SA1/RA1 afferents from palm → C6-T1 spinal cord → finger flexors.
    """
    PRESSURE_THRESH_N = 3.0   # N
    REFRACTORY_MS     = 500.0

    def __init__(self):
        self._last_l = self._last_r = 0.0

    def evaluate(self, pressure_l: float, pressure_r: float) -> Optional[dict]:
        now_ms = time.time() * 1000; cmds = {}
        if pressure_l > self.PRESSURE_THRESH_N and now_ms - self._last_l > self.REFRACTORY_MS:
            cmds["hand_l"] = {"wrist_l_flex": -0.3, "finger_close": 0.8}
            self._last_l = now_ms
        if pressure_r > self.PRESSURE_THRESH_N and now_ms - self._last_r > self.REFRACTORY_MS:
            cmds["hand_r"] = {"wrist_r_flex": -0.3, "finger_close": 0.8}
            self._last_r = now_ms
        return {"reflex": "GRASP", **cmds} if cmds else None


class MoroReflex:
    """
    Moro (vestibular fall detection): sudden downward acceleration → arm abduction.
    Adapted for Bubo as emergency fall-catch response.

    Biology: Otolith saccule → vestibular nuclei → motor neurons (C5, C6).
    Trigger: sudden loss of support (Z accel < -1.5g net) or rapid pitch > 30°/s
    """
    ACCEL_THRESH_G  = 1.5   # net downward acceleration threshold
    PITCH_RATE_DPS  = 30.0  # °/s pitch rate threshold
    REFRACTORY_MS   = 2000.0

    def __init__(self):
        self._last_trigger_ms = 0.0

    def evaluate(self, z_accel_g: float, pitch_rate_dps: float) -> Optional[dict]:
        now_ms = time.time() * 1000
        if now_ms - self._last_trigger_ms < self.REFRACTORY_MS: return None
        if z_accel_g > -self.ACCEL_THRESH_G and abs(pitch_rate_dps) < self.PITCH_RATE_DPS: return None
        self._last_trigger_ms = now_ms
        amp = float(np.clip((abs(z_accel_g) - 0.5) / 1.0, 0.3, 1.0))
        return {
            "reflex": "MORO",
            "amplitude": amp,
            "phase1": {"arm_l_abduct": 0.6*amp, "arm_r_abduct": 0.6*amp,
                       "elbow_l_flex": 0.3*amp, "elbow_r_flex": 0.3*amp},
            "phase2_ms": 200,  # adduction follows after 200ms
            "phase2": {"arm_l_abduct": -0.2*amp, "arm_r_abduct": -0.2*amp},
        }


class OptoKineticReflex:
    """
    OKR: Large-field visual motion → slow eye movement to stabilise image.
    Complements VOR for low-frequency stabilisation (< 1Hz).
    Driven by MT optic flow in direction of whole-field motion.

    Biology: MT → accessory optic system (AOS) → vestibular nuclei → oculomotor.
    """
    FLOW_THRESH = 0.1  # minimum optic flow to trigger OKR

    def evaluate(self, h_flow: float, v_flow: float) -> Optional[dict]:
        if abs(h_flow) < self.FLOW_THRESH and abs(v_flow) < self.FLOW_THRESH: return None
        # Slow-phase eye movement in direction of flow
        return {
            "reflex": "OKR",
            "eye_h_vel_dps": float(-h_flow * 8.0),  # negative: follow motion
            "eye_v_vel_dps": float(-v_flow * 8.0),
        }


class ReflexCoordinator:
    """
    Manages all spinal/brainstem reflexes.
    Instantiated in SpinalArms, SpinalLegs, and Oculomotor nodes.
    Each node subscribes to its relevant reflex outputs.
    """

    def __init__(self, bus: NeuralBus):
        self._bus  = bus
        self.asr   = AcousticStartleReflex()
        self.tlr   = TonicLabyrinthineReflex()
        self.atnr  = AsymmetricTonicNeckReflex()
        self.grasp = PalmarGraspReflex()
        self.moro  = MoroReflex()
        self.okr   = OptoKineticReflex()

    def check_vestibular(self, pitch_deg: float, yaw_deg: float,
                          z_accel_g: float, pitch_rate_dps: float):
        now_ns = time.time_ns()
        tlr = self.tlr.evaluate(pitch_deg)
        if tlr: self._bus.publish(T.REFLEX_TLR, {**tlr, "timestamp_ns": now_ns})

        atnr = self.atnr.evaluate(yaw_deg)
        if atnr: self._bus.publish(T.REFLEX_ATNR, {**atnr, "timestamp_ns": now_ns})

        moro = self.moro.evaluate(z_accel_g, pitch_rate_dps)
        if moro: self._bus.publish(T.REFLEX_MORO, {**moro, "timestamp_ns": now_ns})

    def check_audio(self, rms_db: float):
        asr = self.asr.evaluate(rms_db)
        if asr: self._bus.publish(T.REFLEX_ASR, {**asr, "timestamp_ns": time.time_ns()})

    def check_grasp(self, pressure_l: float, pressure_r: float):
        g = self.grasp.evaluate(pressure_l, pressure_r)
        if g: self._bus.publish(T.REFLEX_GRASP, {**g, "timestamp_ns": time.time_ns()})

    def check_optic_flow(self, h_flow: float, v_flow: float):
        okr = self.okr.evaluate(h_flow, v_flow)
        if okr: self._bus.publish(T.REFLEX_OKR, {**okr, "timestamp_ns": time.time_ns()})
