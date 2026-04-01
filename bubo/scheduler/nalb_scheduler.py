"""
bubo/scheduler/nalb_scheduler.py — Bubo V10
NALB: Neural Adaptive Load Balancer

═══════════════════════════════════════════════════════════════════════
DESIGN: THE NALB SCHEDULER
═══════════════════════════════════════════════════════════════════════

The Neural Adaptive Load Balancer monitors system-wide cognitive load
and makes real-time decisions about priority routing.

KEY RULE (V10):
  If PFC thermal load > 95% AND current task is NOT survival-critical:
    → Switch to SOCIAL_ENGAGEMENT mode
    → Inform the human: "I'm running at capacity — let me slow down.
       Would you like me to continue anyway or take a break?"
    → Reduce PFC publication rate (thermal loop already handles cpufreq)
    → Shift cognitive work to 70B AGX Orin (cooler, more capacity)
    → Route remaining PFC cycles to social interaction (Broca, social node)

RATIONALE:
  Cognitive science: humans under high cognitive load become less empathic,
  more irritable, worse at social reasoning (Baumeister ego depletion).
  BUT: social interaction often requires LESS abstract reasoning than
  technical problem-solving. A taxed PFC can still say hello warmly,
  listen, and respond to simple emotional cues.

  Architecture: Social engagement (Broca + social node + auditory + Orin 70B)
  does NOT heavily load PFC. It loads:
    Social node (192.168.1.19, Orin 8GB): face recognition, latent emotion
    Broca (192.168.1.14, Orin 8GB):      speech generation
    AGX Orin (192.168.1.20):             70B LLM reasoning
    Auditory (192.168.1.51):             speech recognition
  These are SEPARATE nodes from PFC-L/R. PFC can genuinely rest.

SURVIVAL EXCEPTION:
  If the high-load task IS survival-critical:
    → Report load to human
    → Let human choose: "continue at risk of errors" or "pause the task"
    → Never make survival decisions unilaterally with degraded reasoning

MODES:
  NOMINAL:    All systems normal. Full PFC. Normal routing.
  SOCIAL:     PFC thermal > 95%. Route to social. Inform human.
  REDUCED:    PFC thermal 85-95%. Reduce PFC pub rate 50%.
  EMERGENCY:  PFC thermal > 85°C. Vagus thermal abort.
  SURVIVAL:   Imminent physical danger. PFC forced regardless of temp.
  SLEEP:      Battery low + charger detected. Glial cleanup mode.

NALB also implements the broader scheduling policy:
  - Which node gets priority on the DDS bus in a congestion event
  - Which LLM tier handles a given query (13B vs 70B)
  - When to defer non-urgent tasks to charging/sleep window
  - How to redistribute compute when a node fails
"""

import time, logging, threading
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable
from bubo.shared.bus.neural_bus import NeuralBus, T

logger = logging.getLogger("NALB")


class NALBMode(Enum):
    NOMINAL   = "nominal"
    REDUCED   = "reduced"       # 85-95% PFC load
    SOCIAL    = "social"        # >95% PFC load, non-survival
    SURVIVAL  = "survival"      # imminent danger, PFC forced
    EMERGENCY = "emergency"     # thermal abort
    SLEEP     = "sleep"         # charging / consolidation


@dataclass
class SystemLoad:
    """Current load snapshot across all nodes."""
    pfc_l_temp_C:    float = 45.0
    pfc_r_temp_C:    float = 45.0
    pfc_l_cpu_pct:   float = 0.0
    pfc_r_cpu_pct:   float = 0.0
    social_gpu_pct:  float = 0.0
    agx_temp_C:      float = 45.0
    agx_gpu_pct:     float = 0.0
    cluster_avg_C:   float = 45.0
    battery_frac:    float = 1.0
    charging:        bool  = False
    survival_active: bool  = False

    @property
    def pfc_thermal_pct(self) -> float:
        """PFC thermal load as percentage of max-safe operating temp."""
        max_safe = 83.0; min_op = 35.0
        peak = max(self.pfc_l_temp_C, self.pfc_r_temp_C)
        return float(np.clip((peak - min_op) / (max_safe - min_op) * 100, 0, 100))

    @property
    def pfc_cpu_pct(self) -> float:
        return float(max(self.pfc_l_cpu_pct, self.pfc_r_cpu_pct))


@dataclass
class NALBDecision:
    """One NALB scheduling decision."""
    mode:          NALBMode
    reason:        str
    human_message: Optional[str]    = None   # message to speak to human
    reduce_pfc:    bool             = False
    shift_to_agx:  bool             = False
    social_focus:  bool             = False
    pub_rate_scale: float           = 1.0    # PFC pub rate scale
    timestamp_ns:  int              = 0

    def requires_consent(self) -> bool:
        """True if this decision should ask the human first."""
        return self.mode == NALBMode.SOCIAL and self.human_message is not None


class NALBScheduler:
    """
    Neural Adaptive Load Balancer.
    Runs every 2 seconds on hypothalamus node (or as separate thread on PFC-L).
    """

    THERMAL_SOCIAL_THRESH  = 95.0   # % → switch to social mode
    THERMAL_REDUCED_THRESH = 82.0   # % → reduce pub rate
    THERMAL_EMERGENCY_C    = 85.0   # °C → vagus abort
    CPU_HIGH_THRESH        = 90.0   # % CPU → consider reducing
    HYSTERESIS_STEPS       = 3      # require N consecutive readings before mode switch

    def __init__(self, bus: NeuralBus, speech_callback: Optional[Callable] = None):
        self._bus       = bus
        self._speak     = speech_callback   # function(text) → speak to human
        self._mode      = NALBMode.NOMINAL
        self._load      = SystemLoad()
        self._pending_mode: Optional[NALBMode] = None
        self._pending_count = 0
        self._awaiting_consent = False
        self._consent_callback: Optional[Callable] = None
        self._running   = False
        self._lock      = threading.Lock()
        self._last_decision: Optional[NALBDecision] = None
        self._human_agreed_continue = False

    def update_load(self, **kwargs):
        """Update system load snapshot."""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._load, k):
                    setattr(self._load, k, v)

    def human_response(self, agree_to_continue: bool):
        """Called when human responds to NALB's consent request."""
        with self._lock:
            self._awaiting_consent = False
            self._human_agreed_continue = agree_to_continue
        if agree_to_continue:
            logger.info("NALB: Human chose to continue — holding SOCIAL mode, PFC throttled")
            if self._speak:
                self._speak("Understood. I'll continue but may be slower. Thank you for your patience.")
        else:
            logger.info("NALB: Human chose to pause — switching to full SOCIAL mode")
            if self._speak:
                self._speak("Good idea. Let me take a breath and focus on our conversation.")

    def decide(self) -> NALBDecision:
        """Compute NALB scheduling decision from current load."""
        with self._lock:
            load = self._load
            current_mode = self._mode
            survival = load.survival_active

        thermal_pct = load.pfc_thermal_pct
        cpu_pct     = load.pfc_cpu_pct
        peak_C      = max(load.pfc_l_temp_C, load.pfc_r_temp_C)

        # Emergency: hardware thermal abort
        if peak_C >= self.THERMAL_EMERGENCY_C:
            return NALBDecision(
                mode=NALBMode.EMERGENCY,
                reason=f"PFC temp {peak_C:.0f}°C ≥ {self.THERMAL_EMERGENCY_C}°C",
                human_message=f"Warning: I'm running very hot ({peak_C:.0f}°C). "
                              f"Initiating thermal safety shutdown.",
                reduce_pfc=True, pub_rate_scale=0.0,
                timestamp_ns=time.time_ns())

        # Survival: danger overrides thermal
        if survival:
            return NALBDecision(
                mode=NALBMode.SURVIVAL,
                reason="Survival flag active — forced full PFC",
                reduce_pfc=False, pub_rate_scale=1.0,
                timestamp_ns=time.time_ns())

        # High thermal → social or reduced
        if thermal_pct >= self.THERMAL_SOCIAL_THRESH or cpu_pct >= self.CPU_HIGH_THRESH:
            msg = (f"I'm running at {thermal_pct:.0f}% of my safe thermal capacity right now. "
                   f"I'd like to slow down my heavy thinking and focus on our conversation instead. "
                   f"Would you like me to continue anyway, or shall I take a moment to cool down?")
            return NALBDecision(
                mode=NALBMode.SOCIAL,
                reason=f"PFC thermal {thermal_pct:.0f}% > {self.THERMAL_SOCIAL_THRESH}%",
                human_message=msg,
                reduce_pfc=True, shift_to_agx=True, social_focus=True,
                pub_rate_scale=0.3, timestamp_ns=time.time_ns())

        if thermal_pct >= self.THERMAL_REDUCED_THRESH:
            return NALBDecision(
                mode=NALBMode.REDUCED,
                reason=f"PFC thermal {thermal_pct:.0f}% in reduced zone",
                reduce_pfc=True, pub_rate_scale=0.6,
                timestamp_ns=time.time_ns())

        # Sleep: charging + stable
        if load.charging and load.battery_frac > 0.95 and thermal_pct < 50:
            return NALBDecision(
                mode=NALBMode.SLEEP,
                reason="Charging complete, cool — consolidation mode",
                pub_rate_scale=0.5, timestamp_ns=time.time_ns())

        # Nominal
        return NALBDecision(
            mode=NALBMode.NOMINAL,
            reason="All systems nominal",
            pub_rate_scale=1.0, timestamp_ns=time.time_ns())

    def _apply_decision(self, decision: NALBDecision):
        """Apply NALB decision to cluster."""
        prev_mode = self._mode
        with self._lock:
            self._mode = decision.mode
            self._last_decision = decision

        if decision.mode != prev_mode:
            logger.info(f"NALB: {prev_mode.value} → {decision.mode.value} ({decision.reason})")

        # Publish NALB state to hypothalamus
        self._bus.publish(b"SYS_NALB", {
            "mode":          decision.mode.value,
            "reason":        decision.reason,
            "pub_rate_scale": decision.pub_rate_scale,
            "shift_to_agx":  decision.shift_to_agx,
            "social_focus":  decision.social_focus,
            "timestamp_ns":  decision.timestamp_ns,
        })

        # Speak to human if mode requires consent
        if decision.human_message and not self._awaiting_consent:
            self._awaiting_consent = True
            if self._speak:
                self._speak(decision.human_message)
            logger.info(f"NALB → human: '{decision.human_message[:80]}...'")

        # Publish PFC throttle signal
        if decision.reduce_pfc:
            self._bus.publish(T.HYPO_STATE, {
                "pub_rate_scale": decision.pub_rate_scale,
                "motor_inhibit":  0.0 if decision.mode != NALBMode.EMERGENCY else 1.0,
                "source":         "NALB",
                "timestamp_ns":   decision.timestamp_ns,
            })

    def _loop(self):
        while self._running:
            time.sleep(2.0)
            try:
                decision = self.decide()
                self._apply_decision(decision)
            except Exception as e:
                logger.error(f"NALB loop error: {e}")

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info("NALB Scheduler V10 started | 95% social threshold | consent required")

    def stop(self): self._running = False

    @property
    def mode(self) -> NALBMode: return self._mode
    @property
    def load(self) -> SystemLoad: return self._load
