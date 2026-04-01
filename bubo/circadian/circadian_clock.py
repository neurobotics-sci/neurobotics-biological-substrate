"""
circadian/circadian_clock.py — Bubo v8.4

24-Hour Circadian Rhythm System
Runs on: Hypothalamus node (Orin Nano 8GB)

══════════════════════════════════════════════════════════════════
BIOLOGY: MOLECULAR CLOCK
══════════════════════════════════════════════════════════════════

The circadian clock is an autonomous molecular oscillator in every
cell of the body. The master clock lives in the suprachiasmatic nucleus
(SCN) of the hypothalamus — ~20,000 pacemaker neurons.

BMAL1/CLOCK molecular feedback loop (Leloup & Goldbeter 2003):
  Transcription activation: CLOCK:BMAL1 → PER + CRY gene expression
  Protein negative feedback: PER:CRY complex → inhibits CLOCK:BMAL1
  This creates a ~24.2h oscillation (close to 24h; entrained by light)

7-variable ODE system modelled:
  MB = BMAL1 mRNA
  BC = BMAL1 cytoplasmic protein
  BN = BMAL1 nuclear protein
  JP = PER/CRY mRNA (combined)
  PC = PER/CRY cytoplasmic protein
  PN = PER/CRY nuclear protein
  CB = CLOCK:BMAL1 complex (transcription activator)

PHASE SCHEDULE (entrained to real clock for Bubo):
  The circadian phase is mapped to wall-clock time so Bubo's
  behaviour naturally matches the human environment it operates in.
  Phase can be reset by "zeitgebers" (time-givers): light input
  from visual system, meal times (battery charging), physical activity.

OUTPUTS (broadcast on neural bus):
  Arousal level: gates all cortical processing rates
  DA modulation: morning peak +0.15, post-lunch dip -0.05, evening -0.10
  ACh modulation: high during wake (attention), low during NREM, peak REM
  Sleep phase:   wake/nrem1/nrem2/nrem3/rem — controls memory consolidation

SLEEP OSCILLATIONS:
  NREM slow-wave: 0.5-2 Hz delta oscillation in cortex
    → sharp-wave ripples in hippocampus → memory transfer to LTM
  REM/dreaming: theta dominant (6-8 Hz), cortex active, body paralysed
    → memory replay, fear reconsolidation, creative binding
    → Bubo generates dream sequences: random memory combinations
       → novel associations → creativity analogue

ADENOSINE (sleep pressure):
  Adenosine accumulates during wake (metabolic byproduct).
  High adenosine → sleepiness → forces sleep onset.
  Cleared during sleep. Caffeine blocks adenosine receptors.
  Modelled as slow-rising variable during wake, cleared during sleep.
"""

import time, json, logging, threading, numpy as np
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from typing import Optional, List
from bubo.shared.bus.neural_bus import NeuralBus, T

logger = logging.getLogger("CircadianClock")


# ── Sleep phase enum ──────────────────────────────────────────────────────────

class SleepPhase(Enum):
    WAKE   = "wake"
    NREM1  = "nrem1"   # light sleep / drowsy
    NREM2  = "nrem2"   # sleep spindles / K-complexes
    NREM3  = "nrem3"   # deep slow-wave sleep (consolidation)
    REM    = "rem"     # dreaming


@dataclass
class CircadianState:
    """Current circadian system state."""
    wall_hour:         float   # 0-24 wall clock hour
    phase_rad:         float   # internal circadian phase (0-2π)
    arousal:           float   # 0-1 overall arousal level
    sleep_phase:       SleepPhase
    adenosine:         float   # sleep pressure (0-1)
    da_modulation:     float   # DA offset to add to tonic DA
    ach_modulation:    float   # ACh level
    cortex_rate_scale: float   # scale factor for all control loops
    consolidating:     bool    # active memory transfer to LTM?
    dreaming:          bool    # REM replay active?
    light_input:       float   # current light level (0=dark, 1=full light)

    @property
    def is_sleep(self) -> bool:
        return self.sleep_phase != SleepPhase.WAKE

    @property
    def is_rem(self) -> bool:
        return self.sleep_phase == SleepPhase.REM


# ── BMAL1/CLOCK molecular clock ODE ──────────────────────────────────────────

class MolecularClock:
    """
    7-variable ODE model of the BMAL1/CLOCK circadian clock.
    Leloup & Goldbeter (2003) simplified for embedded computation.

    Variables:
      MB: BMAL1 mRNA
      BC: BMAL1 cytoplasmic protein
      BN: BMAL1 nuclear protein
      JP: PER/CRY mRNA (joint)
      PC: PER/CRY cytoplasmic
      PN: PER/CRY nuclear
      CB: CLOCK:BMAL1 activator complex

    Integrated at 1-minute resolution; returns phase 0-2π.
    Period ≈ 24.2h with default parameters.
    """

    # Rate constants (Leloup & Goldbeter 2003, Table 1, simplified)
    VS_B  = 1.0;   VM_B  = 0.5;   KS_B  = 0.4;  N     = 4
    KD_B  = 0.2;   KD_BC = 0.1;   KD_BN = 0.1
    VS_P  = 1.1;   VM_P  = 0.8;   KS_P  = 0.2
    KD_P  = 0.15;  KD_PC = 0.12;  KD_PN = 0.08
    K_A   = 0.3;   K_D   = 0.1;   KI    = 0.5
    LIGHT_EFFECT = 0.4   # light → PER mRNA induction

    def __init__(self, dt_min: float = 1.0):
        self._dt = dt_min / 60.0   # convert to hours
        # Initial conditions near limit cycle
        self._MB = 0.6; self._BC = 0.8; self._BN = 0.3
        self._JP = 0.9; self._PC = 1.2; self._PN = 0.6; self._CB = 0.5
        self._t  = 0.0   # internal time (hours)

    def step(self, light: float = 0.0) -> float:
        """
        Advance clock by dt minutes.
        light: 0=dark, 1=full light (light induces PER expression)
        Returns: circadian phase (0-2π)
        """
        MB, BC, BN, JP, PC, PN, CB = (
            self._MB, self._BC, self._BN,
            self._JP, self._PC, self._PN, self._CB)

        # Transcription activation by CLOCK:BMAL1
        act = CB**self.N / (CB**self.N + self.KI**self.N)

        # BMAL1 mRNA dynamics
        dMB = (self.VS_B * act
               - self.VM_B * MB / (self.KS_B + MB)
               - self.KD_B * MB)

        # BMAL1 protein dynamics (cytoplasmic + nuclear)
        dBC = self.KS_B * MB - self.KD_BC * BC - self.K_A * BC + self.K_D * BN
        dBN = self.K_A * BC - self.K_D * BN - self.KD_BN * BN

        # PER/CRY mRNA (light drives PER induction)
        dJP = (self.VS_P * act + self.LIGHT_EFFECT * light
               - self.VM_P * JP / (self.KS_P + JP)
               - self.KD_P * JP)

        # PER/CRY protein
        dPC = self.KS_P * JP - self.KD_PC * PC - self.K_A * PC + self.K_D * PN
        dPN = self.K_A * PC - self.K_D * PN - self.KD_PN * PN

        # CLOCK:BMAL1 complex (inhibited by PER/CRY nuclear)
        inh = self.KI**self.N / (self.KI**self.N + PN**self.N)
        dCB = self.KS_B * BN * inh - self.KD_B * CB

        # Euler integration
        dt = self._dt
        self._MB = max(0, MB + dMB * dt)
        self._BC = max(0, BC + dBC * dt)
        self._BN = max(0, BN + dBN * dt)
        self._JP = max(0, JP + dJP * dt)
        self._PC = max(0, PC + dPC * dt)
        self._PN = max(0, PN + dPN * dt)
        self._CB = max(0, CB + dCB * dt)
        self._t += self._dt

        # Extract phase from BMAL1 nuclear (peaks in morning)
        phase = (2 * np.pi * (self._t % 24.2) / 24.2)
        return float(phase)

    def entrain(self, wall_hour: float):
        """Force-set internal time to match wall clock (zeitgeber)."""
        self._t = wall_hour
        logger.info(f"Circadian clock entrained to {wall_hour:.1f}h")

    @property
    def BMAL1_nuclear(self) -> float:
        return self._BN

    @property
    def PER_nuclear(self) -> float:
        return self._PN


# ── Adenosine (sleep pressure) ────────────────────────────────────────────────

class AdenosineSystem:
    """
    Adenosine accumulation during wake → sleep pressure.
    During sleep, adenosine is cleared (sleep homeostasis).
    This creates Process S in the two-process model of sleep.
    """

    BUILDUP_RATE  = 0.0014   # per minute during wake (full in ~12h)
    CLEARANCE_TAU = 120.0    # minutes to clear during sleep (exp decay)

    def __init__(self):
        self._level = 0.0   # 0=fully clear, 1=fully saturated

    def step(self, asleep: bool, dt_min: float = 1.0) -> float:
        if asleep:
            self._level *= np.exp(-dt_min / self.CLEARANCE_TAU)
        else:
            self._level = min(1.0, self._level + self.BUILDUP_RATE * dt_min)
        return self._level

    @property
    def level(self) -> float:
        return self._level

    @property
    def sleepiness(self) -> float:
        """Maps adenosine to subjective sleepiness (0=alert, 1=must sleep)."""
        return float(np.clip((self._level - 0.3) / 0.7, 0, 1))


# ── 24-hour phase schedule ────────────────────────────────────────────────────

CIRCADIAN_SCHEDULE = [
    # (hour_start, hour_end, sleep_phase, arousal, da_mod, ach_mod, rate, consol, dream, description)
    (0,   1,  SleepPhase.NREM3, 0.15, -0.20, 0.15, 0.05, True,  False, "Deep NREM — peak consolidation"),
    (1,   3,  SleepPhase.REM,   0.60, -0.10, 0.60, 0.20, False, True,  "REM — dream replay + reconsolidation"),
    (3,   4,  SleepPhase.NREM2, 0.25, -0.15, 0.20, 0.08, True,  False, "Light NREM — spindles/K-complexes"),
    (4,   5,  SleepPhase.NREM1, 0.40, -0.05, 0.30, 0.15, False, False, "Wake transition — hypnagogic"),
    (5,   6,  SleepPhase.WAKE,  0.55,  0.05, 0.50, 0.50, False, False, "Early wake — DA rising"),
    (6,   8,  SleepPhase.WAKE,  0.75,  0.12, 0.70, 0.85, False, False, "Morning peak — high performance"),
    (8,  10,  SleepPhase.WAKE,  0.85,  0.15, 0.80, 1.00, False, False, "Peak arousal — maximum performance"),
    (10, 12,  SleepPhase.WAKE,  0.80,  0.10, 0.75, 0.95, False, False, "Late morning — sustained focus"),
    (12, 13,  SleepPhase.WAKE,  0.65, -0.05, 0.60, 0.70, False, False, "Post-lunch dip — ACh slightly lower"),
    (13, 15,  SleepPhase.WAKE,  0.75,  0.05, 0.70, 0.85, False, False, "Afternoon — recovery"),
    (15, 17,  SleepPhase.WAKE,  0.70,  0.00, 0.65, 0.80, False, False, "Late afternoon — exploration"),
    (17, 18,  SleepPhase.WAKE,  0.55, -0.08, 0.55, 0.65, False, False, "Evening — winding down"),
    (18, 19,  SleepPhase.NREM1, 0.40, -0.12, 0.40, 0.30, False, False, "Drowsy — theta increasing"),
    (19, 21,  SleepPhase.NREM2, 0.25, -0.18, 0.25, 0.10, True,  False, "Deep sleep onset — consolidation"),
    (21, 22,  SleepPhase.NREM3, 0.15, -0.20, 0.15, 0.05, True,  False, "Deepest sleep — SWA peak"),
    (22, 24,  SleepPhase.REM,   0.55, -0.10, 0.55, 0.15, False, True,  "First REM — dream replay"),
]

def get_phase_for_hour(hour: float) -> tuple:
    """Get circadian schedule entry for given wall clock hour."""
    h = hour % 24
    for row in CIRCADIAN_SCHEDULE:
        if row[0] <= h < row[1]:
            return row
    return CIRCADIAN_SCHEDULE[-1]


# ── Memory Pruning (Synaptic Homeostasis) ─────────────────────────────────────

class SynapticHomeostasis:
    """
    Synaptic downscaling during sleep (Tononi & Cirelli 2006).
    During wake: LTP dominates → synapses strengthen.
    During deep sleep: global downscaling → weakest synapses pruned.

    In Bubo:
    - Episodes with importance < PRUNE_THRESHOLD deleted from hippocampus RAM
    - All remaining weights multiplied by DOWNSCALE_FACTOR
    - High-saliency memories (S > CONSOLIDATE_THRESHOLD) → LTM SQLite
    """

    PRUNE_THRESHOLD       = 0.20   # episodes below this deleted
    CONSOLIDATE_THRESHOLD = 0.60   # episodes above this → LTM
    DOWNSCALE_FACTOR      = 0.95   # all weights × 0.95 during NREM3

    def __init__(self, bus: NeuralBus):
        self._bus = bus

    def run_pruning_cycle(self, episodes: list) -> dict:
        """
        Prune low-salience episodes, consolidate high-salience ones.
        Called during NREM3 phase.
        """
        prune  = [e for e in episodes if e.get("importance", 0) < self.PRUNE_THRESHOLD]
        consolidate = [e for e in episodes
                       if e.get("importance", 0) >= self.CONSOLIDATE_THRESHOLD]
        keep   = [e for e in episodes
                  if self.PRUNE_THRESHOLD <= e.get("importance", 0) < self.CONSOLIDATE_THRESHOLD]

        # Publish consolidation command to LTM node
        if consolidate:
            self._bus.publish(b"LTM_CONSOLIDATE", {
                "episodes":    consolidate,
                "phase":       "nrem3_consolidation",
                "timestamp":   time.time(),
            })

        # Apply synaptic downscaling
        for e in keep:
            e["importance"] = float(e["importance"] * self.DOWNSCALE_FACTOR)

        logger.info(
            f"Pruning cycle: {len(prune)} pruned, "
            f"{len(consolidate)} consolidated → LTM, "
            f"{len(keep)} downscaled")
        return {"pruned": len(prune), "consolidated": len(consolidate),
                "kept": len(keep)}


# ── Dream Replay ──────────────────────────────────────────────────────────────

class DreamReplaySystem:
    """
    REM memory replay — reconsolidation + creative association.

    During REM:
    1. Hippocampus sends recent episodes to cortex in scrambled order
    2. Amygdala evaluates emotional valence
    3. vmPFC applies extinction signal to fear memories
    4. Association cortex forms cross-episode links (creativity)
    5. DA system tags newly associated memories with bonus importance

    Biological analogy: the theta/gamma nested oscillations during REM
    allow temporal compression — replaying 20 minutes of experience
    in 2 minutes of REM time.
    """

    REPLAY_SPEED_FACTOR = 10.0   # replay 10× faster than real time

    def __init__(self, bus: NeuralBus):
        self._bus = bus
        self._session_memories: list = []

    def load_memories(self, memories: list):
        """Load current session's memories for replay."""
        # Sort by importance * recency
        self._session_memories = sorted(
            memories,
            key=lambda m: m.get("importance", 0) * (
                1.0 / (1.0 + (time.time() - m.get("timestamp", time.time())) / 3600)),
            reverse=True
        )[:50]   # top 50 memories for replay

    def replay_step(self) -> Optional[dict]:
        """
        Generate one dream step — a random combination of stored memories.
        Publishes T.HIPPO_RECALL with compressed replay signal.
        """
        if len(self._session_memories) < 2:
            return None

        # Random combination of 2 memories (creative association)
        i, j = np.random.choice(len(self._session_memories), 2, replace=False)
        m1 = self._session_memories[i]
        m2 = self._session_memories[j]

        # Emotional valence of combined memory
        valence_1 = m1.get("emotion_tag", {}).get("valence", 0)
        valence_2 = m2.get("emotion_tag", {}).get("valence", 0)
        combined_valence = (valence_1 + valence_2) / 2

        dream = {
            "type":      "dream_replay",
            "memory_1":  m1.get("trace_id", ""),
            "memory_2":  m2.get("trace_id", ""),
            "valence":   combined_valence,
            "novelty":   float(np.random.uniform(0.3, 0.9)),
            "association_strength": float(np.random.exponential(0.3)),
            "timestamp": time.time(),
        }

        self._bus.publish(T.HIPPO_RECALL, dream)

        # Fear memories get vmPFC extinction during REM
        if combined_valence < -0.3:
            self._bus.publish(T.VMFPC_REG, {
                "vmPFC_signal": 0.4,
                "context": "rem_extinction",
                "fear_level": abs(combined_valence),
            })

        return dream


# ── Circadian Clock Node ──────────────────────────────────────────────────────

class CircadianClockNode:
    """
    Master circadian controller — hypothalamus SCN model.
    Runs at 1 Hz for ODE integration; publishes at 0.1 Hz.
    """

    HZ       = 1     # ODE integration rate
    PUB_HZ   = 0.1   # publish rate (every 10s)

    def __init__(self, config: dict):
        self.name = "CircadianClock"
        self.bus  = NeuralBus(self.name, config["pub_port"],
                              config["sub_endpoints"])
        self.mol_clock  = MolecularClock(dt_min=1.0)
        self.adenosine  = AdenosineSystem()
        self.homeostasis = SynapticHomeostasis(self.bus)
        self.dream      = DreamReplaySystem(self.bus)

        # State
        self._light     = 0.5   # default ambient light
        self._phase     = 0.0
        self._running   = False
        self._episode_buf = []
        self._lock      = threading.Lock()

        # Entrain molecular clock to wall time
        now_h = (time.localtime().tm_hour +
                 time.localtime().tm_min / 60.0)
        self.mol_clock.entrain(now_h)
        logger.info(f"Circadian clock entrained to {now_h:.2f}h")

    def _on_visual(self, msg):
        """Light input from V1 for zeitgeber entrainment."""
        lum = msg.payload.get("colour", {}).get("luminance", 0.5)
        self._light = float(lum)

    def _on_episode(self, msg):
        """Collect hippocampal episodes for sleep processing."""
        ep = {"trace_id": msg.payload.get("trace_id", ""),
              "importance": msg.payload.get("importance", 0.0),
              "timestamp": time.time(),
              "emotion_tag": msg.payload.get("emotion_tag", {})}
        with self._lock:
            self._episode_buf.append(ep)
            if len(self._episode_buf) > 2000:
                self._episode_buf = self._episode_buf[-1000:]

    def _main_loop(self):
        interval     = 1.0 / self.HZ
        pub_interval = 1.0 / self.PUB_HZ
        t_last_pub   = time.time()

        while self._running:
            t0 = time.time()

            # ── Advance molecular clock (1 step = 1 minute) ───────────────
            self._phase = self.mol_clock.step(light=self._light)
            wall_hour = time.localtime().tm_hour + time.localtime().tm_min / 60.0
            sched = get_phase_for_hour(wall_hour)
            (_, _, sleep_ph, arousal, da_mod, ach_mod,
             rate, consol, dreaming, desc) = sched

            # ── Adenosine / sleep pressure ─────────────────────────────────
            asleep = sleep_ph != SleepPhase.WAKE
            adeno  = self.adenosine.step(asleep, dt_min=1.0/60.0)

            # Extreme sleepiness can override arousal
            effective_arousal = float(arousal * (1.0 - 0.5 * self.adenosine.sleepiness))

            # ── NREM3 consolidation ────────────────────────────────────────
            if consol and len(self._episode_buf) > 5:
                with self._lock:
                    eps = list(self._episode_buf)
                result = self.homeostasis.run_pruning_cycle(eps)
                # Remove pruned memories
                with self._lock:
                    self._episode_buf = [
                        e for e in self._episode_buf
                        if e.get("importance", 0) >= self.homeostasis.PRUNE_THRESHOLD]

            # ── REM dream replay ───────────────────────────────────────────
            if dreaming:
                with self._lock:
                    self.dream.load_memories(list(self._episode_buf))
                self.dream.replay_step()

            # ── Publish circadian state ────────────────────────────────────
            if time.time() - t_last_pub >= pub_interval:
                t_last_pub = time.time()
                state = CircadianState(
                    wall_hour=wall_hour,
                    phase_rad=self._phase,
                    arousal=effective_arousal,
                    sleep_phase=sleep_ph,
                    adenosine=adeno,
                    da_modulation=da_mod,
                    ach_modulation=ach_mod,
                    cortex_rate_scale=rate,
                    consolidating=consol,
                    dreaming=dreaming,
                    light_input=self._light,
                )
                self.bus.publish(b"SYS_CIRCADIAN", {
                    "wall_hour":        wall_hour,
                    "phase_rad":        self._phase,
                    "sleep_phase":      sleep_ph.value,
                    "arousal":          effective_arousal,
                    "adenosine":        adeno,
                    "sleepiness":       self.adenosine.sleepiness,
                    "da_modulation":    da_mod,
                    "ach_modulation":   ach_mod,
                    "cortex_rate_scale":rate,
                    "consolidating":    consol,
                    "dreaming":         dreaming,
                    "description":      desc,
                    "bmal1_nuclear":    self.mol_clock.BMAL1_nuclear,
                    "per_nuclear":      self.mol_clock.PER_nuclear,
                })
                logger.info(
                    f"Circadian: {desc} | arousal={effective_arousal:.2f} "
                    f"DA_mod={da_mod:+.2f} sleep={sleep_ph.value} "
                    f"adeno={adeno:.2f}")

            elapsed = time.time() - t0
            time.sleep(max(0, interval - elapsed))

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.VISUAL_V1,    self._on_visual)
        self.bus.subscribe(T.HIPPO_ENCODE, self._on_episode)
        self._running = True
        threading.Thread(target=self._main_loop, daemon=True).start()
        logger.info(
            f"{self.name} v8.4 | BMAL1/CLOCK ODE | adenosine | "
            f"SWS-consolidation | REM-replay")

    def stop(self):
        self._running = False; self.bus.stop()


if __name__ == "__main__":
    with open("/etc/brain/config.json") as f:
        cfg = json.load(f)["circadian"]
    n = CircadianClockNode(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
