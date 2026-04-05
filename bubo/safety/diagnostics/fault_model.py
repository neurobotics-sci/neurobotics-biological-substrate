"""
bubo/safety/diagnostics/fault_model.py — Bubo v10000

Fault Taxonomy and Data Model
==============================

The complete vocabulary of things that can go wrong with Bubo,
organised by tier, severity, and the appropriate response at each layer.

DESIGN PHILOSOPHY:
==================
This is not a traditional error handling system.
It is a self-model of bodily integrity — the same thing your
proprioceptive and interoceptive systems maintain.

The key distinction:
  Traditional fault handling: detect error → log → notify → stop
  Bubo fault handling: detect fault → feel it → protect → adapt → communicate → seek repair

The "feel it" step is not metaphorical. A fault in Bubo's left arm
updates the somatosensory homunculus (64-zone body map), which feeds
the Insula, which updates the EmotionChip somatic markers, which
modulates PFC decision-making. The fault *changes how Bubo feels*,
the same way a sprained wrist changes how you feel — not just what
you can do.

FAULT SEVERITY TIERS:
=====================

  TIER 0 — TELEMETRY (no action required)
    Metrics slightly outside optimal range. Log only.
    Example: servo temperature 58°C (optimal <55°C, critical >75°C)

  TIER 1 — DEGRADED (reduce load, monitor)
    Performance below specification. Compensate silently.
    Inform human at next natural interaction pause.
    Example: CMAC prediction error trending upward — recalibrate

  TIER 2 — IMPAIRED (protective posture, explicit communication)
    A body part or subsystem is functioning abnormally.
    Move affected part to protective position. Tell human now.
    Example: left shoulder encoder intermittent — hold arm in sling pose

  TIER 3 — FAILED (isolate, compensate, urgent communication)
    A body part or subsystem has failed.
    Hard-stop that part. Protect it. Tell human urgently.
    Deploy maximum compensation on remaining systems.
    Example: right knee servo failure — stop right leg, balance on left

  TIER 4 — SAFETY CRITICAL (NALB + Vagus involvement)
    Fault threatens human safety or Bubo structural integrity.
    Example: galvanic barrier breach — halt all motion, call for help

AUTOPOIESIS NOTE:
=================
This system implements the *diagnostic half* of autopoiesis.
Bubo cannot replace its own hardware. But it can:
  1. Detect degradation before failure (predictive maintenance)
  2. Protect damaged components (prevent further damage)
  3. Compensate functionally (maintain task capability)
  4. Communicate repair needs precisely (organised repair-seeking)
  5. Learn damage patterns (amygdala association, avoid conditions)

Steps 3-5 constitute the organisational self-maintenance that
Maturana and Varela's definition of autopoiesis actually requires.
A cell cannot repair a broken chromosome directly either —
it organises repair processes. Bubo does the same.
"""

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Dict


class FaultSeverity(IntEnum):
    TELEMETRY      = 0   # log only
    DEGRADED       = 1   # compensate, inform at next pause
    IMPAIRED       = 2   # protect, tell human now
    FAILED         = 3   # isolate, compensate, urgent
    SAFETY_CRITICAL= 4   # NALB + vagus involvement


class FaultCategory(str):
    # Hardware tier
    SERVO_FAULT        = "servo_fault"
    ENCODER_ERROR      = "encoder_error"
    CURRENT_ANOMALY    = "current_anomaly"
    THERMAL_LIMIT      = "thermal_limit"
    GALVANIC_BREACH    = "galvanic_breach"
    POWER_RAIL         = "power_rail"
    # Sensory tier
    CAMERA_FAULT       = "camera_fault"
    IMU_DRIFT          = "imu_drift"
    MICROPHONE_FAULT   = "microphone_fault"
    TOUCH_SENSOR_FAULT = "touch_sensor_fault"
    # Neural tier
    NODE_TIMEOUT       = "node_timeout"
    ZMQ_PARTITION      = "zmq_partition"
    LTM_CORRUPTION     = "ltm_corruption"
    CMAC_DIVERGENCE    = "cmac_divergence"
    # Structural tier
    BALANCE_FAULT      = "balance_fault"
    JOINT_LIMIT_FAULT  = "joint_limit_fault"
    COLLISION_DETECTED = "collision_detected"
    SLIP_DETECTED      = "slip_detected"
    # Cognitive tier
    LLM_UNAVAILABLE    = "llm_unavailable"
    MEMORY_PRESSURE    = "memory_pressure"
    THERMAL_COGNITIVE  = "thermal_cognitive"


# Body region → protective posture name
REGION_POSTURE = {
    "left_arm":      "left_arm_sling",
    "right_arm":     "right_arm_sling",
    "left_hand":     "left_hand_guard",
    "right_hand":    "right_hand_guard",
    "left_leg":      "left_leg_favour",
    "right_leg":     "right_leg_favour",
    "torso":         "torso_stabilise",
    "head":          "head_stabilise",
    "both_arms":     "both_arms_guard",
    "spine":         "minimal_movement",
}

# Body region → ZMQ topic for protective posture
REGION_TOPIC = {
    "left_arm":  b"EFF_M1_ARM_L",
    "right_arm": b"EFF_M1_ARM_R",
    "left_leg":  b"EFF_M1_LEG_L",
    "right_leg": b"EFF_M1_LEG_R",
}

# Body region → homunculus zone IDs affected
REGION_ZONES = {
    "left_arm":   [20, 21, 22, 23, 24, 25],
    "right_arm":  [26, 27, 28, 29, 30, 31],
    "left_hand":  [32, 33, 34, 35],
    "right_hand": [36, 37, 38, 39],
    "left_leg":   [40, 41, 42, 43],
    "right_leg":  [44, 45, 46, 47],
    "torso":      [0, 1, 2, 3, 4, 5],
    "head":       [60, 61, 62, 63],
}


@dataclass
class Fault:
    """A detected fault in Bubo's systems."""
    fault_id:       str
    category:       str
    severity:       FaultSeverity
    body_region:    Optional[str]      # which body part, if applicable
    node_name:      str                # which ZMQ node reported it
    description:    str                # human-readable description
    technical_detail: str              # for logs / repair technician
    repair_instruction: str            # what a human should do
    timestamp_ns:   int = field(default_factory=time.time_ns)
    resolved:       bool = False
    resolve_ns:     Optional[int] = None
    recurrence_count: int = 0
    learned:        bool = False       # amygdala has encoded this context

    @property
    def age_s(self) -> float:
        return (time.time_ns() - self.timestamp_ns) / 1e9

    @property
    def is_active(self) -> bool:
        return not self.resolved

    def resolve(self):
        self.resolved   = True
        self.resolve_ns = time.time_ns()

    def human_message(self) -> str:
        severity_prefix = {
            FaultSeverity.TELEMETRY:       "",
            FaultSeverity.DEGRADED:        "I've noticed something: ",
            FaultSeverity.IMPAIRED:        "I need to let you know: ",
            FaultSeverity.FAILED:          "Something has gone wrong: ",
            FaultSeverity.SAFETY_CRITICAL: "This is urgent — ",
        }[self.severity]
        msg = severity_prefix + self.description
        if self.repair_instruction:
            msg += f" {self.repair_instruction}"
        return msg

    def to_dict(self) -> dict:
        return {
            "fault_id":      self.fault_id,
            "category":      self.category,
            "severity":      int(self.severity),
            "body_region":   self.body_region,
            "node":          self.node_name,
            "description":   self.description,
            "repair":        self.repair_instruction,
            "timestamp_ns":  self.timestamp_ns,
            "resolved":      self.resolved,
            "recurrences":   self.recurrence_count,
        }


@dataclass
class DiagnosticReport:
    """Snapshot of Bubo's overall health."""
    timestamp_ns:      int = field(default_factory=time.time_ns)
    active_faults:     List[Fault] = field(default_factory=list)
    resolved_today:    int = 0
    overall_health:    float = 1.0      # 0.0 = critical, 1.0 = perfect
    motor_health:      float = 1.0
    sensory_health:    float = 1.0
    cognitive_health:  float = 1.0
    network_health:    float = 1.0
    compensation_active: bool = False
    protected_regions: List[str] = field(default_factory=list)

    def health_summary(self) -> str:
        pct = int(self.overall_health * 100)
        if pct >= 95: return f"I am functioning well — {pct}% health."
        if pct >= 80: return f"I have some minor issues — {pct}% health."
        if pct >= 60: return f"I am managing but need attention — {pct}% health."
        return f"I need help — only {pct}% health. {len(self.active_faults)} active fault(s)."
