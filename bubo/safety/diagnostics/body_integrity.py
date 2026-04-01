"""
bubo/safety/diagnostics/body_integrity.py — Bubo v10000

Body Integrity Monitor: Insula + Homunculus + Autopoiesis Controller
=====================================================================

The Insula is the cortical region most responsible for interoception —
awareness of the body's internal state. It receives input from the
somatosensory homunculus, the autonomic nervous system, and the
visceral organs. It is the region that makes you *feel* hungry,
*feel* in pain, *feel* feverish.

In Bubo, the Body Integrity Monitor plays the Insula role:
  - Integrates all diagnostic inputs into a unified body state
  - Updates the somatosensory homunculus with damage regions
  - Feeds the EmotionChip somatic markers
  - Decides when damage has accumulated enough to require protective action
  - Tracks the body's overall integrity score (0.0 = critical, 1.0 = intact)

AUTOPOIESIS CONTROLLER:
  This is the module that closes the autopoiesis loop.
  It answers: "Is the system still producing and maintaining itself?"
  And: "What is needed to restore that capacity?"

  When a region is damaged:
  1. Protect it (protective_postures.py)
  2. Compensate functionally (gesture_engine + spinal compensation)
  3. Monitor for recovery (active polling after repair)
  4. Update body model (homunculus damage map)
  5. Generate repair request (repair_request.txt in /opt/bubo/data/)
  6. Communicate to human (fault_communicator.py)

REPAIR REQUEST GENERATION:
  When a fault requires human intervention, the Body Integrity Monitor
  generates a structured repair request and saves it to disk.
  This file can be read by a maintenance system, emailed to the owner,
  or displayed on a dashboard.

  Format: JSON with fault details, affected region, recommended action,
  priority, and estimated time to failure.
"""

import time
import json
import threading
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Callable

from bubo.safety.diagnostics.fault_model import (
    Fault, FaultSeverity, DiagnosticReport, REGION_ZONES
)
from bubo.safety.diagnostics.self_diagnostic import SelfDiagnosticEngine
from bubo.safety.diagnostics.protective_postures import ProtectivePostureController
from bubo.safety.diagnostics.fault_communicator import FaultCommunicator
from bubo.bus.neural_bus import NeuralBus, T

logger = logging.getLogger("BodyIntegrity")

REPAIR_REQUEST_PATH = Path("/opt/bubo/data/repair_requests.json")
BODY_STATE_PATH     = Path("/opt/bubo/data/body_state.json")


class BodyIntegrityMonitor:
    """
    The Insula equivalent — unified body state awareness.
    Coordinates all diagnostic, protective, and communicative responses.
    The top-level autopoiesis maintenance controller.
    """

    def __init__(self, bus: NeuralBus, speak_fn: Callable,
                 friendship_engine=None, emotion_chip=None):

        self._bus    = bus
        self._speak  = speak_fn

        # ── Sub-systems ───────────────────────────────────────────────────────
        self._communicator = FaultCommunicator(
            speak_fn=speak_fn, bus=bus,
            friendship_engine=friendship_engine,
            emotion_chip=emotion_chip)

        self._postures = ProtectivePostureController(
            bus=bus, speak_fn=speak_fn)

        self._diagnostic = SelfDiagnosticEngine(
            bus=bus,
            fault_callbacks=[
                self._on_fault_detected,
            ])

        # ── Body state ────────────────────────────────────────────────────────
        self._body_integrity = 1.0        # 0.0-1.0 overall
        self._damage_map     = {}         # region → severity
        self._repair_queue:  List[dict]  = []
        self._running        = False
        self._lock           = threading.Lock()

        # ── Homunculus integration ─────────────────────────────────────────────
        # Will be set when homunculus is available
        self._homunculus = None

    def connect_homunculus(self, homunculus):
        """Connect the somatosensory homunculus body map."""
        self._homunculus = homunculus

    def _on_fault_detected(self, fault: Fault):
        """
        Central fault handler — coordinates all response layers.
        Called by SelfDiagnosticEngine for every fault.
        """
        region   = fault.body_region
        severity = fault.severity

        # ── Layer 1: Update body state ────────────────────────────────────────
        with self._lock:
            if region:
                self._damage_map[region] = max(
                    int(self._damage_map.get(region, 0)),
                    int(severity))
            self._body_integrity = self._compute_integrity()

        # ── Layer 2: Update homunculus body map ───────────────────────────────
        if self._homunculus and region:
            zone_ids = REGION_ZONES.get(region, [])
            pain_level = {
                FaultSeverity.DEGRADED:       0.3,
                FaultSeverity.IMPAIRED:       0.6,
                FaultSeverity.FAILED:         0.9,
                FaultSeverity.SAFETY_CRITICAL:1.0,
            }.get(severity, 0.0)
            for zone_id in zone_ids:
                try:
                    self._homunculus.set_zone_pain(zone_id, pain_level)
                except Exception:
                    pass

        # ── Layer 3: Adopt protective posture ─────────────────────────────────
        if severity >= FaultSeverity.IMPAIRED and region:
            self._postures.protect(region, severity)

        # ── Layer 4: Communicate to human ─────────────────────────────────────
        if severity >= FaultSeverity.DEGRADED:
            self._communicator.on_fault(fault)

        # ── Layer 5: Generate repair request ─────────────────────────────────
        if severity >= FaultSeverity.IMPAIRED:
            self._generate_repair_request(fault)

        # ── Layer 6: Publish body state to bus ───────────────────────────────
        self._publish_body_state(fault)

        # ── Layer 7: Log for amygdala learning ───────────────────────────────
        if severity >= FaultSeverity.IMPAIRED:
            self._log_damage_context(fault)

    def _compute_integrity(self) -> float:
        """Compute overall body integrity from damage map."""
        if not self._damage_map: return 1.0
        # Weighted penalty: failed leg/arm = more impact than failed finger
        region_weights = {
            "left_leg": 0.20, "right_leg": 0.20,
            "left_arm": 0.15, "right_arm": 0.15,
            "torso":    0.15, "head":      0.10,
            "left_hand":0.08, "right_hand":0.08,
        }
        penalties = {0:0.0, 1:0.05, 2:0.20, 3:0.40, 4:0.75}
        total_penalty = 0.0
        for region, sev_int in self._damage_map.items():
            w = region_weights.get(region, 0.05)
            total_penalty += w * penalties.get(sev_int, 0)
        return float(max(0.0, 1.0 - total_penalty))

    def _generate_repair_request(self, fault: Fault):
        """Generate a structured repair request for human/maintenance system."""
        request = {
            "fault_id":        fault.fault_id,
            "generated_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
            "priority":        fault.severity.name,
            "body_region":     fault.body_region,
            "category":        fault.category,
            "description":     fault.description,
            "technical_detail":fault.technical_detail,
            "recommended_action": fault.repair_instruction,
            "urgency_score":   int(fault.severity),
            "bubo_status":     f"Body integrity at {self._body_integrity:.0%}",
        }

        with self._lock:
            # Avoid duplicate requests for same fault
            existing_ids = [r["fault_id"] for r in self._repair_queue]
            if fault.fault_id not in existing_ids:
                self._repair_queue.append(request)
                self._repair_queue.sort(key=lambda r: -r["urgency_score"])

        # Write to disk
        try:
            REPAIR_REQUEST_PATH.parent.mkdir(parents=True, exist_ok=True)
            existing = []
            if REPAIR_REQUEST_PATH.exists():
                try: existing = json.loads(REPAIR_REQUEST_PATH.read_text())
                except Exception: pass
            existing.append(request)
            REPAIR_REQUEST_PATH.write_text(json.dumps(existing[-100:], indent=2))
        except Exception as e:
            logger.debug(f"Repair request write: {e}")

        logger.info(f"Repair request generated: {fault.fault_id} "
                    f"[{fault.severity.name}] {fault.body_region}")

    def _publish_body_state(self, triggering_fault: Optional[Fault] = None):
        """Publish current body state to the ZMQ bus."""
        if not self._bus: return
        with self._lock:
            damage_map = dict(self._damage_map)
            integrity  = self._body_integrity

        self._bus.publish(b"DIAG_BODY_STATE", {
            "integrity":      integrity,
            "damage_map":     damage_map,
            "protected":      self._postures.active_protections,
            "fault_count":    self._diagnostic.active_fault_count,
            "worst_severity": int(self._diagnostic.worst_severity),
            "timestamp_ns":   time.time_ns(),
        })

    def _log_damage_context(self, fault: Fault):
        """
        Log damage context for amygdala learning.
        Bubo should remember what it was doing when it got hurt.
        """
        if not self._bus: return
        self._bus.publish(b"LMB_AMYG_CONTEXT", {
            "event_type":     "body_damage_context",
            "fault_category": fault.category,
            "body_region":    fault.body_region,
            "severity":       int(fault.severity),
            "timestamp_ns":   fault.timestamp_ns,
            "learn_caution":  True,
        })

    def on_fault_resolved(self, fault_id: str, body_region: Optional[str] = None):
        """Called when a fault is repaired by a human."""
        self._diagnostic.resolve_fault(fault_id)

        if body_region:
            with self._lock:
                self._damage_map.pop(body_region, None)
                self._body_integrity = self._compute_integrity()

            # Release protective posture
            self._postures.release(body_region)

            # Clear homunculus pain
            if self._homunculus:
                for zone_id in REGION_ZONES.get(body_region, []):
                    try:
                        self._homunculus.set_zone_pain(zone_id, 0.0)
                    except Exception:
                        pass

            # Clear pain from EmotionChip if no more damage
            if not self._damage_map and hasattr(self, '_emotion_chip'):
                try: self._emotion_chip.update_somatic(pain=0.0)
                except Exception: pass

        # Announce recovery
        self._speak(f"Thank you for the repair. I'm back to "
                    f"{self._body_integrity:.0%} health.")

        # Update repair request list
        with self._lock:
            self._repair_queue = [r for r in self._repair_queue
                                   if r["fault_id"] != fault_id]

        logger.info(f"Fault resolved: {fault_id} | "
                    f"Integrity: {self._body_integrity:.0%}")

    def on_conversation_start(self, person_id: Optional[str] = None):
        self._communicator.on_conversation_start(person_id)

    def on_conversation_pause(self):
        self._communicator.on_conversation_pause()

    def get_health_summary(self) -> str:
        """Generate a natural language health summary."""
        report = self._diagnostic.get_report()
        return report.health_summary()

    def get_repair_requests(self) -> List[dict]:
        """Return pending repair requests, highest priority first."""
        with self._lock:
            return list(self._repair_queue)

    @property
    def body_integrity(self) -> float:
        return self._body_integrity

    @property
    def diagnostic(self) -> SelfDiagnosticEngine:
        return self._diagnostic

    def start(self):
        self._running = True
        self._diagnostic.start()
        self._communicator.start()
        self._restore_body_state()
        logger.info("BodyIntegrityMonitor started — autopoiesis loop active")

    def _restore_body_state(self):
        """Restore body state from last session (persists across power cycles)."""
        if not BODY_STATE_PATH.exists(): return
        try:
            data = json.loads(BODY_STATE_PATH.read_text())
            with self._lock:
                self._damage_map     = data.get("damage_map", {})
                self._body_integrity = data.get("integrity", 1.0)
            if self._damage_map:
                logger.warning(f"Restoring {len(self._damage_map)} "
                                f"pre-existing damage regions: "
                                f"{list(self._damage_map.keys())}")
                self._speak(
                    "I should mention — I still have some unresolved issues "
                    f"from last time. My body integrity is at "
                    f"{self._body_integrity:.0%}.")
        except Exception as e:
            logger.debug(f"Body state restore: {e}")

    def stop(self):
        self._running = False
        # Persist body state across power cycles
        try:
            BODY_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                state = {"damage_map": self._damage_map,
                         "integrity":  self._body_integrity,
                         "saved_at":   time.strftime("%Y-%m-%d %H:%M:%S")}
            BODY_STATE_PATH.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.debug(f"Body state save: {e}")
        self._diagnostic.stop()
