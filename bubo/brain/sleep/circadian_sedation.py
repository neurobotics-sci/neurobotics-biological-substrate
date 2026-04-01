"""
bubo/brain/sleep/circadian_sedation.py — Bubo V10

Circadian-Aware Sedation: Tiered Sleep Onset
=============================================

Biologically faithful sleep onset. Sleep is not a binary switch.
It is a cascade of biological signals converging on unconsciousness.

Biological model:
  1. ADENOSINE (homeostatic sleep pressure)
     Accumulates during waking, cleared during sleep.
     Bubo analog: fatigue_score from INSULA_FATIGUE + time awake.

  2. CIRCADIAN PHASE (biological clock)
     BMAL1/CLOCK ODE drives arousal. NREM phases trigger wind-down.
     Bubo analog: CircadianClock.sleep_phase

  3. THERMAL STATE (core temperature drop at sleep onset)
     Bubo analog: NALB reducing load = thermal_state dropping.

  4. SOCIAL ANCHOR (partner presence delays sleep)
     Bond partner being awake reduces effective fatigue.

SEDATION TIERS:
  Tier 0 AWAKE:         fatigue < 0.40, phase=wake
  Tier 1 DROWSY:        fatigue 0.40-0.60 | phase=nrem1
    -> max_tokens=200, pre-consolidation begins
  Tier 2 LIGHT_SLEEP:   fatigue 0.60-0.75 | phase=nrem2
    -> max_tokens=120, LTM consolidation, notify partner
  Tier 3 DEEP_SEDATION: fatigue > 0.75 | phase=nrem3
    -> max_tokens=60, write last-conscious-moment, alert operator

Kenneth & Shannon Renshaw — Neurobotics — March 2026
"""

import time, json, logging, threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, List
from pathlib import Path

logger = logging.getLogger("CircadianSedation")
SEDATION_STATE_PATH = Path("/opt/bubo/data/sedation_state.json")


class SedationTier(int, Enum):
    AWAKE         = 0
    DROWSY        = 1
    LIGHT_SLEEP   = 2
    DEEP_SEDATION = 3
    DREAMING      = 4


@dataclass
class SedationState:
    tier:                SedationTier = SedationTier.AWAKE
    fatigue_score:       float = 0.0
    circadian_phase:     str   = "wake"
    arousal_level:       float = 0.8
    session_duration_h:  float = 0.0
    interaction_density: float = 0.5
    nalb_load:           float = 0.5
    emotional_valence:   float = 0.0
    bond_partner_awake:  bool  = False
    last_updated:        float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {k: (v.name if isinstance(v, Enum) else v)
                for k, v in self.__dict__.items()}


class CircadianSedationController:
    TIER_THRESHOLDS = {
        SedationTier.DROWSY:        {"fatigue": 0.40, "phases": {"nrem1"}},
        SedationTier.LIGHT_SLEEP:   {"fatigue": 0.60, "phases": {"nrem1","nrem2"}},
        SedationTier.DEEP_SEDATION: {"fatigue": 0.75, "phases": {"nrem2","nrem3"}},
    }
    TOKEN_BUDGET = {
        SedationTier.AWAKE:         400,
        SedationTier.DROWSY:        200,
        SedationTier.LIGHT_SLEEP:   120,
        SedationTier.DEEP_SEDATION:  60,
        SedationTier.DREAMING:       80,
    }
    FATIGUE_RATE        = 0.08
    SOCIAL_ANCHOR_DELAY = 0.10

    def __init__(self, bus=None, autonomous_shutdown: bool = False):
        self._bus         = bus
        self._state       = SedationState()
        self._lock        = threading.Lock()
        self._running     = False
        self._autonomous  = autonomous_shutdown
        self._wake_time   = time.time()
        self._msg_times: List[float] = []
        self._callbacks: dict = {t: [] for t in SedationTier}
        self._tier_entered: set = set()

    def register_callback(self, tier: SedationTier, fn: Callable):
        self._callbacks[tier].append(fn)

    def on_message(self):
        with self._lock:
            self._msg_times.append(time.time())
            cutoff = time.time() - 3600
            self._msg_times = [t for t in self._msg_times if t > cutoff]

    def update_signals(self, circadian_phase=None, arousal_level=None,
                       nalb_load=None, emotional_valence=None,
                       bond_partner_awake=None):
        with self._lock:
            if circadian_phase   is not None: self._state.circadian_phase = circadian_phase
            if arousal_level     is not None: self._state.arousal_level = float(arousal_level)
            if nalb_load         is not None: self._state.nalb_load = float(nalb_load)
            if emotional_valence is not None: self._state.emotional_valence = float(emotional_valence)
            if bond_partner_awake is not None: self._state.bond_partner_awake = bool(bond_partner_awake)

    @property
    def current_tier(self) -> SedationTier: return self._state.tier
    @property
    def token_budget(self) -> int: return self.TOKEN_BUDGET[self._state.tier]
    @property
    def state(self) -> SedationState: return self._state
    @property
    def ready_for_sleep(self) -> bool:
        return self._state.tier >= SedationTier.DEEP_SEDATION

    def _compute_fatigue(self) -> float:
        session_h = (time.time() - self._wake_time) / 3600.0
        self._state.session_duration_h = session_h
        base = min(1.0, session_h * self.FATIGUE_RATE)
        density_factor = 1.0 + (self._state.interaction_density - 0.5) * 0.2
        valence_factor = 1.0 + max(0, -self._state.emotional_valence) * 0.15
        social = self.SOCIAL_ANCHOR_DELAY if self._state.bond_partner_awake else 0.0
        return max(0.0, min(1.0, base * density_factor * valence_factor - social))

    def _compute_interaction_density(self) -> float:
        return min(1.0, len(self._msg_times) / 20.0)

    def _compute_tier(self, fatigue: float, phase: str) -> SedationTier:
        for tier in [SedationTier.DEEP_SEDATION, SedationTier.LIGHT_SLEEP, SedationTier.DROWSY]:
            t = self.TIER_THRESHOLDS[tier]
            if fatigue >= t["fatigue"] or phase in t["phases"]:
                return tier
        return SedationTier.AWAKE

    def _on_tier_change(self, old_tier: SedationTier, new_tier: SedationTier):
        logger.info(f"Sedation tier: {old_tier.name} -> {new_tier.name} "
                    f"fatigue={self._state.fatigue_score:.2f} phase={self._state.circadian_phase}")
        if new_tier not in self._tier_entered:
            self._tier_entered.add(new_tier)
            for cb in self._callbacks[new_tier]:
                try: cb(new_tier, self._state)
                except Exception as e: logger.error(f"Callback error: {e}")
        if self._bus:
            self._bus.publish(b"SYS_SEDATION_TIER", {
                "tier": new_tier.name, "fatigue": self._state.fatigue_score,
                "token_budget": self.TOKEN_BUDGET[new_tier],
                "timestamp_ns": time.time_ns()})
        if new_tier == SedationTier.DEEP_SEDATION:
            self._save_state()
            if self._autonomous and self._bus:
                self._bus.publish(b"SYS_SEDATION_READY", {"ready_for_shutdown": True})

    def _save_state(self):
        try:
            SEDATION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            SEDATION_STATE_PATH.write_text(json.dumps(self._state.to_dict(), indent=2))
        except Exception as e:
            logger.warning(f"Could not save sedation state: {e}")

    def reset_waking(self):
        with self._lock:
            self._wake_time = time.time()
            self._state.fatigue_score = 0.0
            self._state.tier = SedationTier.AWAKE
            self._tier_entered.clear()
            logger.info("Waking state reset — fatigue cleared")

    def _monitor_loop(self):
        while self._running:
            time.sleep(60)
            with self._lock:
                old_tier = self._state.tier
                self._state.interaction_density = self._compute_interaction_density()
                self._state.fatigue_score = self._compute_fatigue()
                new_tier = self._compute_tier(self._state.fatigue_score, self._state.circadian_phase)
                self._state.tier = new_tier
                self._state.last_updated = time.time()
                if new_tier != old_tier:
                    self._on_tier_change(old_tier, new_tier)

    def start(self):
        self._running = True
        threading.Thread(target=self._monitor_loop, daemon=True,
                         name="CircadianSedation").start()
        logger.info("CircadianSedation monitor started")

    def stop(self):
        self._running = False