"""
brain_nodes/autonomic/ans_node.py — v0.13

Hypothalamic / ANS node — Jetson TK1
Homeostatic drives: Hunger (low battery) + Thermal Stress (overheating)
Full HPA axis model → cortisol → PFC fear/stress modulation

BIOLOGICAL BASIS
────────────────
The hypothalamus is the master homeostatic controller:

  Lateral hypothalamus (LH): hunger/satiety → orexin → arousal
  Paraventricular nucleus (PVN): HPA axis master — CRH → pituitary
  Ventromedial hypothalamus (VMH): satiety, glucose sensing
  Preoptic area (POA): thermoregulation → sweating/shivering
  Suprachiasmatic nucleus (SCN): circadian master clock

HPA AXIS (Hypothalamic-Pituitary-Adrenal):
  PVN → CRH → anterior pituitary → ACTH → adrenal cortex → cortisol
  Cortisol:
    - Raises blood glucose (gluconeogenesis)
    - Suppresses immune system
    - Enhances amygdala sensitivity (fear amplification)
    - Reduces hippocampal neurogenesis (chronic stress → memory impairment)
    - Negative feedback to PVN (prevents runaway)
  Timescale: minutes to hours

HOMEOSTATIC DRIVES → HARDWARE SIGNALS
  Hunger  ≡ battery voltage dropping below threshold
           System voltage: Jetson Nano nominal 5V, critical < 4.5V
           Maps to: LH orexin firing → PFC "obtain resources" salience
                    Reduced motor exploration (conserve energy)

  Thermal stress ≡ SoC/CPU temperature > 65°C
           Maps to: POA thermoregulation → sympathetic → sweat
                    PVN CRH release → cortisol surge
                    PFC stress signal → interrupt current task
                    Amygdala sensitisation (stress + amygdala = anxiety)

CORTISOL MODEL (Tsai et al. 2005 pharmacokinetic):
  d[CRH]/dt  = k_CRH · stressor - k_CRH_deg · [CRH]
  d[ACTH]/dt = k_ACTH · [CRH] - k_ACTH_deg · [ACTH]
  d[Cort]/dt = k_Cort · [ACTH] - k_Cort_deg · [Cort] - k_FB · [Cort]²
  Negative feedback: cortisol inhibits PVN CRH release.
"""

import time, json, logging, threading, subprocess
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

from bubo.shared.bus.neural_bus import NeuralBus, NeuralMessage, T
from bubo.shared.neuromodulators.neuromod_system import NeuromodulatorSystem

logger = logging.getLogger("ANS_Hypothalamus")


# ══════════════════════════════════════════════════════════════════════════════
# HARDWARE SENSORS  (Jetson TK1 + cluster-wide via SSH)
# ══════════════════════════════════════════════════════════════════════════════

class HardwareSensors:
    """
    Read real hardware metrics from the Jetson TK1 host system.
    Cached with timestamps to avoid hammering sysfs.
    """

    THERMAL_ZONE   = "/sys/class/thermal/thermal_zone0/temp"
    BATTERY_UCAP   = "/sys/class/power_supply/BAT0/charge_now"
    BATTERY_UFULL  = "/sys/class/power_supply/BAT0/charge_full"
    INA3221_BASE   = "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0"
    CACHE_TTL_S    = 0.5

    def __init__(self):
        self._cache = {}; self._t_cache = {}

    def _read(self, path: str, default=0.0) -> float:
        now = time.time()
        if path in self._cache and (now - self._t_cache[path]) < self.CACHE_TTL_S:
            return self._cache[path]
        try:
            val = float(Path(path).read_text().strip())
        except Exception:
            val = default
        self._cache[path] = val; self._t_cache[path] = now
        return val

    def cpu_temp_C(self) -> float:
        return self._read(self.THERMAL_ZONE, 45000.0) / 1000.0

    def battery_fraction(self) -> float:
        cap  = self._read(self.BATTERY_UCAP,  3000000.0)
        full = self._read(self.BATTERY_UFULL, 4000000.0)
        return float(np.clip(cap / max(full, 1.0), 0.0, 1.0))

    def supply_voltage_V(self) -> float:
        """INA3221 rail voltage for Jetson 5V rail (channel 0)."""
        p = f"{self.INA3221_BASE}/in_voltage0_input"
        return self._read(p, 5000.0) / 1000.0   # mV → V

    def gpu_temp_C(self) -> float:
        return self._read("/sys/class/thermal/thermal_zone1/temp", 40000.0) / 1000.0

    def cluster_temps(self, node_ips: list) -> dict:
        """Quick SSH poll for CPU temps across cluster nodes."""
        temps = {}
        for ip in node_ips[:4]:   # poll first 4 to avoid blocking too long
            try:
                result = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=1",
                     "-o", "StrictHostKeyChecking=no",
                     "-i", "/home/brain/.ssh/brain_cluster_id_ed25519",
                     f"brain@{ip}",
                     "cat /sys/class/thermal/thermal_zone0/temp"],
                    capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    temps[ip] = float(result.stdout.strip()) / 1000.0
            except Exception:
                pass
        return temps


# ══════════════════════════════════════════════════════════════════════════════
# HPA AXIS MODEL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HPAState:
    CRH:     float = 0.1    # corticotropin-releasing hormone (0–1 normalised)
    ACTH:    float = 0.05   # adrenocorticotropic hormone
    cortisol: float = 0.15  # plasma cortisol (0–1)


class HPAAxis:
    """
    Simplified 3-compartment ODE model of the HPA axis.
    Integrates with Euler method at 1 Hz (sufficient for hormonal timescales).

    Stressors drive CRH; CRH drives ACTH; ACTH drives cortisol.
    Cortisol provides negative feedback at PVN (limits runaway).

    Cortisol effects on downstream systems:
      → Amygdala: sensitisation (+0.3 per unit cortisol above baseline)
      → Hippocampus: encoding suppression (>0.7 cortisol → theta amplitude ↓)
      → PFC: working memory interference (>0.6 cortisol → WM capacity ↓)
      → NE release: potentiation (cortisol + NE → fear memory consolidation)
    """

    K_CRH    = 0.8    # stressor → CRH production rate
    K_CRH_D  = 0.15   # CRH degradation
    K_ACTH   = 0.6    # CRH → ACTH
    K_ACTH_D = 0.25   # ACTH degradation
    K_CORT   = 0.5    # ACTH → cortisol
    K_CORT_D = 0.10   # cortisol degradation
    K_FB     = 0.40   # cortisol → PVN negative feedback
    BASELINE  = HPAState()

    def __init__(self):
        self._s = HPAState()

    def step(self, stressor: float, dt: float) -> HPAState:
        """Euler integration of 3-compartment ODE."""
        s = self._s
        fb = self.K_FB * s.cortisol**2   # nonlinear feedback

        dCRH  = self.K_CRH  * stressor - self.K_CRH_D  * s.CRH  - fb
        dACTH = self.K_ACTH * s.CRH    - self.K_ACTH_D * s.ACTH
        dCort = self.K_CORT * s.ACTH   - self.K_CORT_D * s.cortisol \
                - self.K_FB * s.cortisol

        s.CRH      = float(np.clip(s.CRH      + dCRH  * dt, 0.0, 1.0))
        s.ACTH     = float(np.clip(s.ACTH     + dACTH * dt, 0.0, 1.0))
        s.cortisol = float(np.clip(s.cortisol + dCort * dt, 0.0, 1.0))
        return HPAState(s.CRH, s.ACTH, s.cortisol)

    @property
    def state(self) -> HPAState: return self._s


# ══════════════════════════════════════════════════════════════════════════════
# HOMEOSTATIC DRIVE MODELS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HomeostaticState:
    """Current homeostatic drive levels — all 0-1."""
    hunger:     float = 0.0    # 0=full, 1=critical depletion
    thermal:    float = 0.0    # 0=normal, 1=critical overheat
    fatigue:    float = 0.0    # 0=rested, 1=exhausted
    cortisol:   float = 0.15   # from HPA
    stress:     float = 0.0    # combined stressor
    # Derived physiological
    heart_rate:    float = 70.0
    resp_rate:     float = 12.0
    blood_pressure:float = 120.0
    pupil_mm:      float = 3.0
    skin_conduct:  float = 0.0   # sweat (sympathetic)
    reflex_gain:   float = 1.0


class HungerDrive:
    """
    Battery level → hunger signal.
    Models lateral hypothalamus orexin system.

    Thresholds (Jetson Nano 5V nominal supply):
      > 4.8V  : sated (hunger = 0)
      4.5-4.8V: mild hunger (hunger = 0.3–0.7)
      4.2-4.5V: strong hunger → interrupt motor task, seek charging
      < 4.2V  : critical → enter low-power state, emergency signal

    Orexin effect on PFC:
      Moderate hunger (0.3): increased exploration / goal-seeking salience
      High hunger (0.7): reduced working memory capacity (food > task)
      Critical (1.0): all non-survival goals suspended
    """

    V_SATED   = 4.80   # V
    V_MILD    = 4.50   # V
    V_STRONG  = 4.20   # V
    V_CRIT    = 3.90   # V

    def __init__(self):
        self._prev_bat = 1.0
        self._hunger   = 0.0

    def update(self, voltage_V: float, battery_frac: float) -> float:
        if voltage_V > self.V_SATED:
            target = 0.0
        elif voltage_V > self.V_MILD:
            target = float((self.V_SATED - voltage_V) / (self.V_SATED - self.V_MILD) * 0.5)
        elif voltage_V > self.V_STRONG:
            target = float(0.5 + (self.V_MILD - voltage_V) / (self.V_MILD - self.V_STRONG) * 0.4)
        elif voltage_V > self.V_CRIT:
            target = float(0.9 + (self.V_STRONG - voltage_V) / (self.V_STRONG - self.V_CRIT) * 0.09)
        else:
            target = 1.0

        # Also factor raw battery fraction (slow drift)
        target = max(target, float(np.clip(1.0 - battery_frac * 1.2, 0, 1)))
        # Low-pass filter (hunger builds and decays slowly)
        self._hunger = float(0.95 * self._hunger + 0.05 * target)
        return self._hunger


class ThermalStressDrive:
    """
    CPU/GPU temperature → thermal stress signal.
    Models hypothalamic preoptic area (POA) thermoregulation.

    Temperature thresholds:
      < 55°C  : normal
      55–70°C : moderate stress → fans, reduce compute, warn PFC
      70–80°C : high stress → throttle, emergency signal
      > 80°C  : critical → cortisol surge, suspend heavy processes

    Thermal-cortisol coupling:
      High CPU temp acts as a physiological stressor → HPA activation.
      Models the fact that heat stress is a genuine physiological stressor.
    """

    T_NORMAL = 55.0
    T_WARM   = 70.0
    T_HOT    = 80.0
    T_CRIT   = 85.0

    def __init__(self):
        self._stress = 0.0

    def update(self, cpu_C: float, gpu_C: float) -> float:
        peak = max(cpu_C, gpu_C)
        if peak < self.T_NORMAL:
            target = 0.0
        elif peak < self.T_WARM:
            target = float((peak - self.T_NORMAL) / (self.T_WARM - self.T_NORMAL) * 0.4)
        elif peak < self.T_HOT:
            target = float(0.4 + (peak - self.T_WARM) / (self.T_HOT - self.T_WARM) * 0.4)
        elif peak < self.T_CRIT:
            target = float(0.8 + (peak - self.T_HOT) / (self.T_CRIT - self.T_HOT) * 0.18)
        else:
            target = 1.0
        self._stress = float(0.8 * self._stress + 0.2 * target)
        return self._stress


# ══════════════════════════════════════════════════════════════════════════════
# ANS NODE v0.13
# ══════════════════════════════════════════════════════════════════════════════

class ANSNode:
    """
    Hypothalamic / ANS node — Jetson TK1

    v0.13 additions:
      - Hardware battery voltage → hunger drive (LH orexin model)
      - CPU/GPU temperature → thermal stress drive (POA model)
      - HPA axis ODE → cortisol → PFC stress + amygdala sensitisation
      - Cluster-wide temperature polling (SSH, every 30s)
      - Homeostatic signals broadcast to PFC (T.HYPO_STATE):
          hunger, thermal, cortisol, stress
          + PFC modulation: WM_capacity_scale, fear_scale, exploration_bias
    """

    HZ = 1   # 1 Hz — hormonal and homeostatic timescales

    NODE_IPS = ["192.168.1.10","192.168.1.11","192.168.1.30","192.168.1.34"]

    def __init__(self, config):
        self.name = "ANS_Hypothalamus"; self.cfg = config
        self.bus  = NeuralBus(self.name, config["pub_port"], config["sub_endpoints"])
        self.nm   = NeuromodulatorSystem()
        self.hw   = HardwareSensors()
        self.hpa  = HPAAxis()
        self.hunger  = HungerDrive()
        self.thermal = ThermalStressDrive()

        self._fear       = 0.0
        self._arousal    = 0.2
        self._reward     = 0.0
        self._cluster_temps: dict = {}
        self._t_cluster  = 0.0
        self._running    = False
        self._lock       = threading.Lock()

        # History for HRV / physiological variability
        self._hr_hist = deque(maxlen=60)

    # ── Handlers ─────────────────────────────────────────────────────────────

    def _on_cea(self, msg):
        with self._lock:
            self._fear    = float(msg.payload.get("cea_activation", 0))
            self._arousal = float(msg.payload.get("flee", 0) + msg.payload.get("freeze", 0))

    def _on_reward(self, msg):
        self._reward = float(msg.payload.get("rpe", 0.0))

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _loop(self):
        interval = 1.0 / self.HZ
        while self._running:
            t0 = time.time()

            # ── Read hardware ──────────────────────────────────────────────
            cpu_C   = self.hw.cpu_temp_C()
            gpu_C   = self.hw.gpu_temp_C()
            batt_V  = self.hw.supply_voltage_V()
            batt_f  = self.hw.battery_fraction()

            # Cluster temp poll (every 30s — don't block main loop)
            if time.time() - self._t_cluster > 30.0:
                self._t_cluster = time.time()
                threading.Thread(
                    target=lambda: self._cluster_temps.update(
                        self.hw.cluster_temps(self.NODE_IPS)),
                    daemon=True).start()

            max_cluster_C = max(self._cluster_temps.values(), default=cpu_C)

            # ── Homeostatic drives ─────────────────────────────────────────
            hunger_level  = self.hunger.update(batt_V, batt_f)
            thermal_level = self.thermal.update(max(cpu_C, max_cluster_C), gpu_C)

            # Combined stressor for HPA
            with self._lock:
                fear_level = self._fear
            stressor = float(np.clip(
                0.35 * thermal_level +
                0.25 * hunger_level  +
                0.30 * fear_level    +
                0.10 * self._arousal, 0, 1))

            hpa = self.hpa.step(stressor, dt=1.0)

            # ── Neuromodulators ────────────────────────────────────────────
            nm_state = self.nm.step(
                reward=self._reward,
                novelty=float(np.random.exponential(0.05)),
                fear=fear_level,
                attention=min(1.0, self._arousal + 0.3),
                punishment=float(thermal_level > 0.6))

            # ── Physiological state (ANS) ──────────────────────────────────
            sym = float(np.clip(
                0.5 * self._arousal + 0.3 * fear_level +
                0.15 * thermal_level + 0.05 * hunger_level, 0, 1))
            hr  = 70  + 130 * sym
            rr  = 12  + 18  * sym
            bp  = 120 + 60  * sym
            pup = 3.0 + 5.0 * sym
            sweat = float(np.clip((cpu_C - 37.5) / 5.0, 0, 1)) if cpu_C > 37.5 else 0.0
            self._hr_hist.append(hr)

            # ── PFC modulation factors (derived from cortisol + drives) ────
            # High cortisol → reduce WM capacity (stress impairs cognition)
            wm_scale    = float(np.clip(1.0 - 0.5 * max(hpa.cortisol - 0.4, 0), 0.3, 1.0))
            # Cortisol potentiates amygdala → fear scale increases
            fear_scale  = float(1.0 + 0.8 * hpa.cortisol)
            # Hunger → increase exploration salience (seek resources)
            explore_bias= float(0.3 * hunger_level)
            # Thermal → suppress motor actions (reduce heat generation)
            motor_inhibit = float(0.5 * thermal_level)

            hs = HomeostaticState(
                hunger=hunger_level, thermal=thermal_level,
                fatigue=0.0, cortisol=hpa.cortisol, stress=stressor,
                heart_rate=hr, resp_rate=rr, blood_pressure=bp,
                pupil_mm=pup, skin_conduct=sweat,
                reflex_gain=float(np.clip(0.8 + 0.4*sym, 0.5, 1.5)))

            # Log critical states
            if hunger_level > 0.7:
                logger.warning(f"HUNGER CRITICAL: {batt_V:.2f}V ({hunger_level:.2f})")
            if thermal_level > 0.6:
                logger.warning(f"THERMAL STRESS: CPU={cpu_C:.1f}°C ({thermal_level:.2f})")

            # ── Publish ────────────────────────────────────────────────────
            self.bus.publish(T.HYPO_STATE, {
                # Homeostatic drives
                "hunger":          hunger_level,
                "thermal":         thermal_level,
                "stressor":        stressor,
                # HPA
                "CRH":             hpa.CRH,
                "ACTH":            hpa.ACTH,
                "cortisol":        hpa.cortisol,
                # PFC modulation
                "wm_capacity_scale":  wm_scale,
                "fear_scale":         fear_scale,
                "exploration_bias":   explore_bias,
                "motor_inhibit":      motor_inhibit,
                # Hardware readings
                "cpu_temp_C":      cpu_C,
                "gpu_temp_C":      gpu_C,
                "battery_V":       batt_V,
                "battery_frac":    batt_f,
                "cluster_max_C":   max_cluster_C,
            })
            self.bus.publish(T.ANS_SYMPATH, {
                "heart_rate_bpm":    round(hr,1),
                "resp_rate_bpm":     round(rr,1),
                "blood_pressure_mm": round(bp,1),
                "pupil_mm":          round(pup,2),
                "sweat_rate":        round(sweat,3),
                "sympathetic_drive": round(sym,3),
                "reflex_gain":       hs.reflex_gain,
            })
            self.bus.publish(T.DA_VTA,     {"DA":  nm_state.DA})
            self.bus.publish(T.NE_LC,      {"NE":  nm_state.NE})
            self.bus.publish(T.SERO_RAPHE, {"5HT": nm_state.sero})
            self.bus.publish(T.ACH_NBM,    {"ACh": nm_state.ACh})

            elapsed = time.time() - t0
            time.sleep(max(0, interval - elapsed))

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.AMYG_CEA_OUT, self._on_cea)
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info(
            f"{self.name} v0.13 | HPA-ODE | "
            f"Hunger(V_thresh={self.hunger.V_MILD}V) | "
            f"Thermal(T_warn={self.thermal.T_WARM}°C)")

    def stop(self): self._running=False; self.bus.stop()


if __name__ == "__main__":
    with open("/etc/brain/config.json") as f: cfg = json.load(f)["ans"]
    n = ANSNode(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
