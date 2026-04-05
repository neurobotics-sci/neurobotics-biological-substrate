"""
bubo/safety/diagnostics/self_diagnostic.py — Bubo v10000

Multi-Tier Self-Diagnostic Engine
===================================

The five-layer diagnostic system, mirroring the five-layer
biological response to bodily damage.

Layer 1 — HARDWARE LAYER (STM32H7, BeagleBoard, servo bus)
  Servo current, temperature, encoder validity, power rail voltage.
  Poll rate: 10Hz. Direct from hardware registers.
  Analogue: peripheral nervous system sensory neurons.

Layer 2 — SPINAL LAYER (spinal_arms, spinal_legs)
  Joint range of motion, torque limits, CPG stability, ZMP violation.
  Poll rate: 50Hz. From spinal node ZMQ heartbeats.
  Analogue: spinal cord — first integration point for body state.

Layer 3 — CEREBELLAR LAYER (cerebellum CMAC)
  Prediction error accumulation, motor learning divergence,
  balance controller stability, gait anomalies.
  Poll rate: 10Hz. From CMAC weight statistics.
  Analogue: cerebellar purkinje cell error signal.

Layer 4 — HOMEOSTATIC LAYER (hypothalamus, insula, network)
  Thermal regulation, power budget, ZMQ network health,
  node timeout detection, memory pressure.
  Poll rate: 2Hz. From homeostatic node reports.
  Analogue: hypothalamus + insula interoception.

Layer 5 — COGNITIVE LAYER (PFC, LTM, LLM)
  LLM availability, LTM integrity, working memory pressure,
  NALB thermal ceiling, GWT broadcast failures.
  Poll rate: 1Hz. From cortical node statistics.
  Analogue: prefrontal interoceptive awareness.

PREDICTIVE MAINTENANCE:
  Each diagnostic channel maintains a rolling window of readings.
  Linear trend detection identifies degradation trajectories before failure.
  "This servo's current draw has increased 15% over 48 hours.
   Predicted failure in 72 hours. Recommend lubrication or replacement."

FAULT MEMORY (AMYGDALA INTEGRATION):
  When a fault recurs in the same context (same task, same environment,
  same time of day), the amygdala encodes the association.
  Bubo will become cautious about that context proactively —
  the same way you become careful on the stairs after tripping once.
"""

import time
import threading
import logging
import json
import numpy as np
from typing import Optional, Dict, List, Callable
from collections import deque
from pathlib import Path

from bubo.safety.diagnostics.fault_model import (
    Fault, FaultSeverity, FaultCategory, DiagnosticReport,
    REGION_POSTURE, REGION_TOPIC, REGION_ZONES
)

logger = logging.getLogger("SelfDiagnostic")

FAULT_HISTORY_PATH = Path("/opt/bubo/data/fault_history.json")


class DiagnosticChannel:
    """
    A single monitored metric with trend detection.
    Analogous to a single sensory receptor reporting to the spinal cord.
    """
    WINDOW = 100   # readings in rolling window

    def __init__(self, name: str, units: str, warn_threshold: float,
                 critical_threshold: float, direction: str = "above"):
        self.name      = name
        self.units     = units
        self._warn     = warn_threshold
        self._critical = critical_threshold
        self._dir      = direction   # "above" or "below"
        self._readings = deque(maxlen=self.WINDOW)
        self._timestamps = deque(maxlen=self.WINDOW)

    def record(self, value: float):
        self._readings.append(value)
        self._timestamps.append(time.time())

    def current_severity(self) -> FaultSeverity:
        if not self._readings: return FaultSeverity.TELEMETRY
        v = self._readings[-1]
        if self._dir == "above":
            if v >= self._critical: return FaultSeverity.FAILED
            if v >= self._warn:     return FaultSeverity.DEGRADED
        else:
            if v <= self._critical: return FaultSeverity.FAILED
            if v <= self._warn:     return FaultSeverity.DEGRADED
        return FaultSeverity.TELEMETRY

    def trend_slope(self) -> float:
        """Linear trend slope — positive = increasing, negative = decreasing."""
        if len(self._readings) < 10: return 0.0
        x = np.arange(len(self._readings), dtype=float)
        y = np.array(self._readings, dtype=float)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def predicted_failure_hours(self) -> Optional[float]:
        """Predict hours until critical threshold at current trend."""
        if not self._readings or not self._timestamps: return None
        slope = self.trend_slope()
        if abs(slope) < 1e-6: return None
        current = self._readings[-1]
        if self._dir == "above" and slope > 0:
            readings_until = (self._critical - current) / slope
            dt = (self._timestamps[-1] - self._timestamps[0]) / max(len(self._timestamps)-1, 1)
            return max(0.0, readings_until * dt / 3600)
        elif self._dir == "below" and slope < 0:
            readings_until = (current - self._critical) / (-slope)
            dt = (self._timestamps[-1] - self._timestamps[0]) / max(len(self._timestamps)-1, 1)
            return max(0.0, readings_until * dt / 3600)
        return None

    @property
    def latest(self) -> Optional[float]:
        return self._readings[-1] if self._readings else None


class SelfDiagnosticEngine:
    """
    Five-layer self-diagnostic system.
    Central nervous system for Bubo's bodily self-awareness.
    """

    # Diagnostic polling intervals (seconds)
    HARDWARE_POLL   = 0.1    # 10Hz
    SPINAL_POLL     = 0.02   # 50Hz
    CEREBELLAR_POLL = 0.1    # 10Hz
    HOMEOSTATIC_POLL= 0.5    # 2Hz
    COGNITIVE_POLL  = 1.0    # 1Hz
    REPORT_INTERVAL = 30.0   # Full report every 30s

    def __init__(self, bus=None, fault_callbacks: List[Callable] = None):
        self._bus       = bus
        self._callbacks = fault_callbacks or []
        self._active_faults: Dict[str, Fault] = {}
        self._fault_history: List[dict] = []
        self._report    = DiagnosticReport()
        self._running   = False
        self._lock      = threading.Lock()
        self._fault_counter = 0

        # ── Diagnostic channels (one per monitored metric) ────────────────────
        self._channels: Dict[str, DiagnosticChannel] = {}
        self._init_channels()
        self._load_history()

    def _init_channels(self):
        """Initialise all monitored metrics."""
        C = DiagnosticChannel

        # Hardware: servo temperatures
        for joint in ["shoulder_l", "elbow_l", "wrist_l",
                       "shoulder_r", "elbow_r", "wrist_r",
                       "hip_l", "knee_l", "ankle_l",
                       "hip_r", "knee_r", "ankle_r"]:
            self._channels[f"temp_{joint}"] = C(
                f"temp_{joint}", "°C", warn_threshold=65, critical_threshold=78)

        # Hardware: servo currents
        for joint in ["elbow_l", "elbow_r", "knee_l", "knee_r"]:
            self._channels[f"current_{joint}"] = C(
                f"current_{joint}", "A", warn_threshold=1.8, critical_threshold=2.5)

        # Hardware: power rail
        self._channels["voltage_24v"] = C(
            "voltage_24v", "V", warn_threshold=22.0, critical_threshold=20.0,
            direction="below")
        self._channels["voltage_5v"] = C(
            "voltage_5v", "V", warn_threshold=4.7, critical_threshold=4.5,
            direction="below")

        # Spinal: ZMP deviation (balance)
        self._channels["zmp_deviation"] = C(
            "zmp_deviation", "mm", warn_threshold=30, critical_threshold=60)

        # Spinal: joint position error
        for joint in ["knee_l", "knee_r", "ankle_l", "ankle_r"]:
            self._channels[f"pos_error_{joint}"] = C(
                f"pos_error_{joint}", "rad", warn_threshold=0.05, critical_threshold=0.15)

        # Cerebellar: CMAC prediction error
        self._channels["cmac_pred_error_l"] = C(
            "cmac_pred_error_l", "Nm", warn_threshold=2.0, critical_threshold=5.0)
        self._channels["cmac_pred_error_r"] = C(
            "cmac_pred_error_r", "Nm", warn_threshold=2.0, critical_threshold=5.0)

        # Homeostatic: node latency
            self._channels[f"latency_{node}"] = C(
                f"latency_{node}", "ms",
                warn_threshold=50, critical_threshold=200)

        # Homeostatic: cognitive thermal
        self._channels["pfc_temp"] = C(
            "pfc_temp", "°C", warn_threshold=72, critical_threshold=82)

        # Cognitive: LTM health
        self._channels["ltm_index_health"] = C(
            "ltm_index_health", "%",
            warn_threshold=85, critical_threshold=70, direction="below")

        logger.info(f"SelfDiagnostic: {len(self._channels)} channels initialised")

    def update_channel(self, channel_name: str, value: float):
        """Feed a new reading into a diagnostic channel."""
        ch = self._channels.get(channel_name)
        if ch is None: return
        ch.record(value)
        severity = ch.current_severity()
        if severity >= FaultSeverity.DEGRADED:
            self._raise_fault_from_channel(channel_name, ch, severity)

    def _raise_fault_from_channel(self, ch_name: str,
                                   ch: DiagnosticChannel,
                                   severity: FaultSeverity):
        """Create or update a fault from a channel threshold breach."""
        fault_key = f"channel_{ch_name}"

        # Determine body region from channel name
        body_region = self._infer_body_region(ch_name)

        # Predictive warning
        pred_hours = ch.predicted_failure_hours()
        pred_text  = ""
        if pred_hours and pred_hours < 72:
            pred_text = f" Predicted failure in {pred_hours:.1f} hours."

        description = self._describe_channel_fault(ch_name, ch, severity)
        repair      = self._repair_instruction(ch_name, body_region, severity)

        with self._lock:
            if fault_key in self._active_faults:
                self._active_faults[fault_key].recurrence_count += 1
                self._active_faults[fault_key].severity = severity
                return

            self._fault_counter += 1
            fault = Fault(
                fault_id=f"DIAG-{self._fault_counter:05d}",
                category=self._categorise(ch_name),
                severity=severity,
                body_region=body_region,
                node_name="self_diagnostic",
                description=description + pred_text,
                technical_detail=f"{ch_name}={ch.latest:.2f}{ch.units} "
                                  f"trend={ch.trend_slope():.4f}/reading",
                repair_instruction=repair,
            )
            self._active_faults[fault_key] = fault

        self._dispatch_fault(fault)

    def raise_fault(self, category: str, severity: FaultSeverity,
                    body_region: Optional[str], node_name: str,
                    description: str, repair_instruction: str,
                    technical_detail: str = "") -> Fault:
        with self._lock:
            self._fault_counter += 1
            fault = Fault(
                fault_id=f"DIAG-{self._fault_counter:05d}",
                category=category, severity=severity,
                body_region=body_region, node_name=node_name,
                description=description,
                technical_detail=technical_detail,
                repair_instruction=repair_instruction,
            )
            key = f"{node_name}_{category}_{body_region or 'global'}"
            if key in self._active_faults:
                self._active_faults[key].recurrence_count += 1
                return self._active_faults[key]
            self._active_faults[key] = fault

        self._dispatch_fault(fault)
        return fault

    def resolve_fault(self, fault_id: str):
        """Mark a fault as resolved (e.g., after human repair)."""
        with self._lock:
            for key, fault in list(self._active_faults.items()):
                if fault.fault_id == fault_id:
                    fault.resolve()
                    self._fault_history.append(fault.to_dict())
                    del self._active_faults[key]
                    logger.info(f"Fault resolved: {fault_id}")
                    break
        self._save_history()

    def _dispatch_fault(self, fault: Fault):
        """Route fault to appropriate response systems via callbacks and ZMQ."""
        logger.warning(f"FAULT [{fault.severity.name}] {fault.fault_id}: "
                        f"{fault.description}")

        # Publish to ZMQ bus
        if self._bus:
            self._bus.publish(b"DIAG_FAULT", fault.to_dict())

        # Call registered callbacks (protective posture, communication, etc.)
        for cb in self._callbacks:
            try: cb(fault)
            except Exception as e: logger.error(f"Fault callback error: {e}")

    def get_report(self) -> DiagnosticReport:
        """Generate current health report."""
        with self._lock:
            active = list(self._active_faults.values())

        motor_faults    = [f for f in active if f.body_region in
                           ["left_arm","right_arm","left_leg","right_leg",
                            "left_hand","right_hand"]]
        sensory_faults  = [f for f in active if "camera" in f.category or
                            "imu" in f.category or "microphone" in f.category]
        cognitive_faults= [f for f in active if "llm" in f.category or
                            "ltm" in f.category or "thermal_cognitive" in f.category]
        network_faults  = [f for f in active if "timeout" in f.category or
                            "zmq" in f.category]

        def health(faults: list) -> float:
            if not faults: return 1.0
            worst = max(f.severity for f in faults)
            penalties = {0:0.0, 1:0.05, 2:0.20, 3:0.45, 4:0.80}
            return max(0.0, 1.0 - penalties.get(int(worst), 0))

        overall = min(
            health(motor_faults), health(sensory_faults),
            health(cognitive_faults), health(network_faults))

        return DiagnosticReport(
            active_faults=active,
            overall_health=overall,
            motor_health=health(motor_faults),
            sensory_health=health(sensory_faults),
            cognitive_health=health(cognitive_faults),
            network_health=health(network_faults),
            compensation_active=any(f.severity >= FaultSeverity.IMPAIRED
                                     for f in active),
            protected_regions=[f.body_region for f in active
                                if f.body_region and
                                f.severity >= FaultSeverity.IMPAIRED],
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _infer_body_region(self, ch_name: str) -> Optional[str]:
        if "_l_" in ch_name or ch_name.endswith("_l"):
            if "knee" in ch_name or "ankle" in ch_name or "hip" in ch_name:
                return "left_leg"
            return "left_arm"
        if "_r_" in ch_name or ch_name.endswith("_r"):
            if "knee" in ch_name or "ankle" in ch_name or "hip" in ch_name:
                return "right_leg"
            return "right_arm"
        if "pfc" in ch_name: return "head"
        if "zmp" in ch_name: return "torso"
        return None

    def _categorise(self, ch_name: str) -> str:
        if "temp" in ch_name:    return FaultCategory.THERMAL_LIMIT
        if "current" in ch_name: return FaultCategory.CURRENT_ANOMALY
        if "voltage" in ch_name: return FaultCategory.POWER_RAIL
        if "cmac" in ch_name:    return FaultCategory.CMAC_DIVERGENCE
        if "zmp" in ch_name:     return FaultCategory.BALANCE_FAULT
        if "latency" in ch_name: return FaultCategory.NODE_TIMEOUT
        if "ltm" in ch_name:     return FaultCategory.LTM_CORRUPTION
        if "pos_error" in ch_name: return FaultCategory.ENCODER_ERROR
        return "unknown"

    def _describe_channel_fault(self, ch_name: str,
                                  ch: DiagnosticChannel,
                                  severity: FaultSeverity) -> str:
        region = self._infer_body_region(ch_name) or "a system"
        region_friendly = region.replace("_", " ")
        if "temp" in ch_name:
            return (f"My {region_friendly} is running hot at "
                    f"{ch.latest:.0f}°C.")
        if "current" in ch_name:
            return (f"I'm drawing unusually high current in my "
                    f"{region_friendly} — {ch.latest:.2f}A.")
        if "pos_error" in ch_name:
            return (f"There's a positioning error in my "
                    f"{region_friendly} — it's not going exactly where I tell it.")
        if "cmac" in ch_name:
            return (f"My motor learning system is showing larger than "
                    f"normal prediction errors in my {region_friendly}.")
        if "zmp" in ch_name:
            return (f"My balance is off — I'm having trouble staying stable.")
        if "latency" in ch_name:
            node = ch_name.replace("latency_","").replace("_"," ")
            return (f"My {node} is responding slowly — "
                    f"{ch.latest:.0f}ms.")
        return f"Anomaly detected in {ch_name}: {ch.latest:.2f}{ch.units}"

    def _repair_instruction(self, ch_name: str,
                             body_region: Optional[str],
                             severity: FaultSeverity) -> str:
        if severity < FaultSeverity.IMPAIRED:
            return "I'll keep monitoring this and let you know if it gets worse."
        if "temp" in ch_name:
            return ("Could you check that my ventilation is clear and "
                    "that I'm not in direct sunlight?")
        if "current" in ch_name:
            return ("This might mean a joint needs lubrication or there's "
                    "a mechanical obstruction. Could you check "
                    f"my {(body_region or 'arm').replace('_',' ')}?")
        if "pos_error" in ch_name or "cmac" in ch_name:
            return ("This might mean a servo needs inspection or calibration. "
                    "A technician should check "
                    f"my {(body_region or 'arm').replace('_',' ')}.")
        if "zmp" in ch_name:
            return ("I might need to sit down or hold something stable. "
                    "Please check the ground surface I'm standing on.")
        if "latency" in ch_name:
            return ("One of my neural nodes may have crashed. "
                    "Checking if it needs to be restarted.")
        return "Please inspect the affected area when you have a chance."

    def _load_history(self):
        if not FAULT_HISTORY_PATH.exists(): return
        try:
            self._fault_history = json.loads(FAULT_HISTORY_PATH.read_text())
            logger.info(f"Fault history loaded: {len(self._fault_history)} records")
        except Exception as e:
            logger.warning(f"Fault history load failed: {e}")

    def _save_history(self):
        try:
            FAULT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Keep last 10,000 records
            history = self._fault_history[-10000:]
            FAULT_HISTORY_PATH.write_text(json.dumps(history, indent=2))
        except Exception as e:
            logger.debug(f"Fault history save: {e}")

    def start(self):
        self._running = True
        logger.info("SelfDiagnosticEngine started — 5-layer monitoring active")

    def stop(self):
        self._running = False
        self._save_history()

    @property
    def active_fault_count(self) -> int:
        return len(self._active_faults)

    @property
    def worst_severity(self) -> FaultSeverity:
        with self._lock:
            if not self._active_faults:
                return FaultSeverity.TELEMETRY
            return max(f.severity for f in self._active_faults.values())
