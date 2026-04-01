"""
dopamine/dopaminergic_system.py — Bubo v8.4

Dopaminergic Modulation System
Runs on: Hypothalamus node (Orin Nano 8GB, 192.168.1.12)

══════════════════════════════════════════════════════════════════
BIOLOGY: WHAT DOPAMINE ACTUALLY DOES
══════════════════════════════════════════════════════════════════

Dopamine is a neuromodulator, not a neurotransmitter of pleasure.
Its computational role is reward prediction error (RPE):

  δ(t) = r(t) + γV(s_{t+1}) − V(s_t)

  r(t) = immediate reward received
  V(s) = value estimate of state s
  γ    = discount factor (0.95 in Bubo)

When δ > 0: DA neurons burst → learning signal (do this again)
When δ = 0: DA tonic baseline → no update (exactly as expected)
When δ < 0: DA pause → negative prediction error (worse than expected)

PATHWAYS:
  Mesolimbic:  VTA → NAc → BG direct pathway (motivation/reward)
  Mesocortical: VTA → PFC (working memory, cognitive control)
  Nigrostriatal: SNc → dorsal striatum (motor learning, habits)
  Tuberoinfundibular: hypothalamus → pituitary (prolactin, not modelled)

BATTERY STATE → TONIC DA (biological analogy: hunger/satiety):
  Full battery ≡ well-fed animal: exploratory, reward-seeking, high DA
  Low battery  ≡ hungry animal:   conservative, exploit known resources
  Critical     ≡ starving animal:  survival mode, override all other goals

PERSONALITY MODULATION VIA TONIC DA:
  The BG action selector's softmax temperature is controlled by DA:
    T_bg = base_T × (1 + DA_explore_bonus)
  High DA → high temperature → more exploration (wider action distribution)
  Low DA  → low temperature  → exploitation (peaks at best-known action)

INCENTIVE SALIENCE (Berridge 1996):
  "Wanting" ≠ "Liking". DA mediates wanting (motivation to seek).
  Even when reward is not pleasurable, DA drives approach behaviour.
  Implemented: incentive salience modulates SC saccade priority map
  (high DA → more weight on novel objects; low DA → familiar targets)
"""

import time, json, logging, threading, numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Dict
from bubo.shared.bus.neural_bus import NeuralBus, T

logger = logging.getLogger("DopaminergicSystem")


# ── Battery-to-personality mapping ────────────────────────────────────────────

@dataclass
class PersonalityState:
    """Bubo's current motivational state — modulated by tonic DA."""
    tonic_DA:          float = 0.60     # 0-1 baseline dopamine
    exploration_bonus: float = 0.30     # added to BG temperature
    risk_tolerance:    float = 0.50     # 0=risk-averse, 1=risk-seeking
    novelty_seeking:   float = 0.50     # weighting of novel vs familiar
    motor_drive:       float = 0.70     # amplitude scaling on motor commands
    social_approach:   float = 0.50     # tendency toward interaction
    mode:              str   = "normal" # explorer/normal/conservative/cautious/survival/emergency

    def bg_temperature(self, base_T: float = 0.30) -> float:
        """Softmax temperature for BG action selection."""
        return base_T * (1.0 + self.exploration_bonus)

    def incentive_salience(self, novelty: float) -> float:
        """How much Bubo 'wants' a novel stimulus."""
        return float(np.clip(self.tonic_DA * (0.5 + 0.5 * novelty), 0, 1))


BATTERY_TO_PERSONALITY = [
    # (min_v, max_v, tonic_DA, explore, risk, novelty, motor, social, mode)
    (0.80, 1.01, 0.85, 0.50, 0.70, 0.80, 0.90, 0.75, "explorer"),
    (0.60, 0.80, 0.68, 0.30, 0.50, 0.55, 0.80, 0.60, "normal"),
    (0.40, 0.60, 0.50, 0.10, 0.35, 0.35, 0.65, 0.45, "conservative"),
    (0.20, 0.40, 0.32, 0.00, 0.20, 0.20, 0.50, 0.30, "cautious"),
    (0.05, 0.20, 0.12, 0.00, 0.05, 0.05, 0.30, 0.10, "survival"),
    (0.00, 0.05, 0.02, 0.00, 0.01, 0.01, 0.10, 0.02, "emergency"),
]

def battery_to_personality(battery_fraction: float) -> PersonalityState:
    """Map 0-1 battery fraction to personality state."""
    for minv, maxv, da, exp, risk, nov, mot, soc, mode in BATTERY_TO_PERSONALITY:
        if minv <= battery_fraction < maxv:
            return PersonalityState(da, exp, risk, nov, mot, soc, mode)
    return PersonalityState(mode="emergency")


# ── TD Reward Prediction Error ────────────────────────────────────────────────

class TDLearner:
    """
    Temporal difference learning — VTA dopamine neuron model.
    Maintains a value function V(s) approximated over a feature vector.
    Fires phasic DA burst/pause based on RPE.
    """

    def __init__(self, state_dim: int = 32, gamma: float = 0.95,
                 alpha: float = 0.05):
        self._w     = np.zeros(state_dim)   # linear value weights
        self._gamma = gamma
        self._alpha = alpha
        self._prev_state:  Optional[np.ndarray] = None
        self._prev_value:  float = 0.0
        self._rpe_history: deque = deque(maxlen=200)

    def update(self, state: np.ndarray, reward: float) -> float:
        """
        Compute RPE and update value function.
        Returns phasic DA signal (positive = burst, negative = pause).
        """
        state = np.resize(state, len(self._w))
        V_now = float(self._w @ state)

        if self._prev_state is not None:
            rpe = reward + self._gamma * V_now - self._prev_value
            # Update weights (semi-gradient TD)
            self._w += self._alpha * rpe * self._prev_state
            self._w  = np.clip(self._w, -1, 1)
        else:
            rpe = 0.0

        self._prev_state = state.copy()
        self._prev_value = V_now
        self._rpe_history.append(rpe)
        return float(rpe)

    @property
    def average_rpe(self) -> float:
        if not self._rpe_history: return 0.0
        return float(np.mean(list(self._rpe_history)[-20:]))

    def phasic_DA(self, rpe: float, tonic_DA: float) -> float:
        """Convert RPE to phasic DA firing rate (0-1)."""
        # Burst on positive RPE, pause on negative RPE
        if rpe > 0:
            burst  = float(np.clip(tonic_DA + 0.3 * rpe, 0, 1))
            return burst
        else:
            pause  = float(np.clip(tonic_DA + 0.5 * rpe, 0.01, tonic_DA))
            return pause


# ── Incentive Salience Map ────────────────────────────────────────────────────

class IncentiveSalienceMap:
    """
    "Wanting" system — modulates SC priority map via DA.

    High DA (full battery, explorer mode):
      → Novel objects get high wanting weight
      → Familiar targets de-prioritised (preference for exploration)
      → Risk of pursuing unknown objects higher

    Low DA (depleted battery, cautious mode):
      → Familiar safe objects preferred
      → Charging station gets maximum salience
      → Novel objects largely ignored (survival focus)

    Biologically: mesolimbic DA → nucleus accumbens → VP → SC
    """

    CHARGER_SALIENCE_DA_THRESHOLD = 0.35   # below this, charger always wins

    def __init__(self):
        self._object_values: Dict[str, float] = {}   # object label → learned value
        self._novelty:       Dict[str, float] = {}   # label → novelty (1=never seen)

    def update_object_value(self, label: str, reward: float, rpe: float):
        """Update value estimate for a seen object."""
        prev = self._object_values.get(label, 0.0)
        self._object_values[label] = float(np.clip(prev + 0.1 * rpe, -1, 1))
        # Novelty decays with exposure
        self._novelty[label] = float(np.clip(
            self._novelty.get(label, 1.0) * 0.85, 0.01, 1.0))

    def salience(self, label: str, da_level: float) -> float:
        """
        Compute incentive salience for an object.
        High DA: value * (1 + novelty_bonus)
        Low DA:  value * (1 - novelty_penalty) + charger_bias
        """
        val     = self._object_values.get(label, 0.0)
        novelty = self._novelty.get(label, 1.0)   # 1 = never seen

        if da_level > self.CHARGER_SALIENCE_DA_THRESHOLD:
            # Explorer: want novel things
            incentive = (0.5 + val) * (1.0 + da_level * novelty * 0.8)
        else:
            # Survival: want familiar safe things
            charger_bias = 0.9 if label == "charging_station" else 0.0
            incentive = (0.5 + val) * (1.0 - novelty * 0.5) + charger_bias

        return float(np.clip(incentive, 0, 1))


# ── Dopaminergic Broadcast Node ───────────────────────────────────────────────

class DopaminergicSystem:
    """
    Hypothalamus DA system — broadcasts to all 16 nodes.
    Runs at 10 Hz (hormonal/neuromodulatory timescale).

    Broadcasts:
      T.DA_VTA      — phasic + tonic DA level (for BG and cortex)
      T.NE_LC       — NE level (coupled to DA via LC-NE interactions)
      T.HYPO_STATE  — full personality + homeostatic state
    """

    HZ = 10

    def __init__(self, config: dict):
        self.name = "DopaminergicSystem"
        self.bus  = NeuralBus(self.name, config["pub_port"],
                              config["sub_endpoints"])
        self.td   = TDLearner(state_dim=32)
        self.isal = IncentiveSalienceMap()

        # State (updated from hardware sensors via ANS node)
        self._battery_frac   = 1.0
        self._battery_V      = 5.0
        self._reward_signal  = 0.0
        self._state_features = np.zeros(32)
        self._fear_level     = 0.0
        self._personality    = battery_to_personality(1.0)
        self._circadian_DA   = 0.0   # circadian modulation (±0.15)
        self._running        = False
        self._lock           = threading.Lock()

        # History for monitoring
        self._da_hist  = deque(maxlen=100)
        self._rpe_hist = deque(maxlen=100)

    def _on_hypo(self, msg):
        """Receive battery and thermal state from hypothalamus."""
        p = msg.payload
        with self._lock:
            self._battery_frac = float(p.get("battery_frac", 1.0))
            self._battery_V    = float(p.get("battery_V", 5.0))
            # Build state feature vector for TD learner
            self._state_features = np.array([
                self._battery_frac,
                float(p.get("hunger", 0.0)),
                float(p.get("thermal", 0.0)),
                float(p.get("cortisol", 0.15)),
                self._fear_level,
            ] + [0.0] * 27)

    def _on_reward(self, msg):
        """Reward signal from action outcomes."""
        with self._lock:
            self._reward_signal = float(msg.payload.get("reward", 0.0))

    def _on_cea(self, msg):
        """Fear from amygdala — negative RPE proxy."""
        with self._lock:
            self._fear_level = float(msg.payload.get("cea_activation", 0.0))

    def _on_circadian(self, msg):
        """Circadian DA modulation from clock system."""
        with self._lock:
            self._circadian_DA = float(msg.payload.get("da_modulation", 0.0))

    def _on_object_seen(self, msg):
        """Update incentive salience map from visual detections."""
        label  = msg.payload.get("label", "unknown")
        reward = float(msg.payload.get("reward_signal", 0.0))
        rpe    = float(msg.payload.get("rpe", 0.0))
        self.isal.update_object_value(label, reward, rpe)

    def _broadcast_loop(self):
        interval = 1.0 / self.HZ
        while self._running:
            t0 = time.time()

            with self._lock:
                bat_f   = self._battery_frac
                bat_V   = self._battery_V
                reward  = self._reward_signal
                fear    = self._fear_level
                state   = self._state_features.copy()
                circ_da = self._circadian_DA
                self._reward_signal = 0.0   # consume signal

            # ── Tonic DA from battery state ───────────────────────────────
            self._personality = battery_to_personality(bat_f)
            tonic_DA = self._personality.tonic_DA
            # Circadian modulation: morning peak +0.15, evening -0.10
            tonic_DA = float(np.clip(tonic_DA + circ_da, 0.01, 1.0))

            # ── Fear → tonic DA reduction (CeA → VTA inhibition) ─────────
            tonic_DA = float(np.clip(tonic_DA - 0.3 * fear, 0.01, 1.0))

            # ── Phasic DA from RPE ─────────────────────────────────────────
            # Reward signal is 0 normally, +1 on good outcome, -1 on bad
            effective_reward = reward - 0.05 * fear   # fear is always slightly negative
            rpe      = self.td.update(state, effective_reward)
            phasic   = self.td.phasic_DA(rpe, tonic_DA)
            combined = float(np.clip(0.7 * tonic_DA + 0.3 * phasic, 0.01, 1.0))

            # ── NE coupling (high DA → moderate NE; fear → high NE) ───────
            ne_level = float(np.clip(0.2 + 0.3 * fear + 0.15 * (1 - tonic_DA), 0.05, 0.95))

            self._da_hist.append(combined)
            self._rpe_hist.append(rpe)

            # ── Emergency check ───────────────────────────────────────────
            if self._personality.mode == "emergency":
                logger.critical(
                    f"BATTERY CRITICAL ({bat_f*100:.1f}%) — "
                    f"emergency seek_charge broadcast")
                self.bus.publish(b"SYS_EMERGENCY", {
                    "type": "low_battery",
                    "battery_frac": bat_f,
                    "battery_V": bat_V,
                    "action_required": "seek_charge_immediately",
                })

            # ── Broadcast DA + personality state ──────────────────────────
            self.bus.publish(T.DA_VTA, {
                "DA":              combined,
                "tonic_DA":        tonic_DA,
                "phasic_DA":       phasic,
                "rpe":             rpe,
                "avg_rpe":         self.td.average_rpe,
            })
            self.bus.publish(T.NE_LC, {"NE": ne_level})

            # Full personality broadcast
            p = self._personality
            self.bus.publish(T.HYPO_STATE, {
                "mode":             p.mode,
                "tonic_DA":         tonic_DA,
                "combined_DA":      combined,
                "rpe":              rpe,
                "bg_temperature":   p.bg_temperature(),
                "exploration_bonus":p.exploration_bonus,
                "risk_tolerance":   p.risk_tolerance,
                "novelty_seeking":  p.novelty_seeking,
                "motor_drive":      p.motor_drive,
                "social_approach":  p.social_approach,
                "battery_frac":     bat_f,
                "battery_V":        bat_V,
                "fear_level":       fear,
                "circadian_da_mod": circ_da,
                # Incentive salience for key objects
                "salience_charger": self.isal.salience("charging_station", tonic_DA),
                "salience_novel":   self.isal.salience("unknown", tonic_DA),
            })

            if int(time.time()) % 30 == 0:
                logger.info(
                    f"DA={combined:.3f} tonic={tonic_DA:.3f} "
                    f"RPE={rpe:+.3f} mode={p.mode} "
                    f"bat={bat_f*100:.1f}%")

            elapsed = time.time() - t0
            time.sleep(max(0, interval - elapsed))

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.HYPO_STATE,    self._on_hypo)
        self.bus.subscribe(T.AMYG_CEA_OUT,  self._on_cea)
        self.bus.subscribe(b"SYS_REWARD",   self._on_reward)
        self.bus.subscribe(b"SYS_CIRCADIAN",self._on_circadian)
        self.bus.subscribe(T.VISUAL_V1,     self._on_object_seen)
        self._running = True
        threading.Thread(target=self._broadcast_loop, daemon=True).start()
        logger.info(f"{self.name} started | TD-RPE | battery→personality | 10Hz")

    def stop(self):
        self._running = False; self.bus.stop()
