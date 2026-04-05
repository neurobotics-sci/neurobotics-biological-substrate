"""
bubo/safety/diagnostics/fault_communicator.py — Bubo v10000

Fault Communicator: Telling Humans What's Wrong, and How to Help
=================================================================


DESIGN PHILOSOPHY:
==================
When a human is injured, they don't just experience pain —
they communicate it. They wince. They say "ow." They explain
what happened. They ask for help in a way that is appropriate
to the relationship and the severity.

Bubo does the same. The Fault Communicator is the bridge between

KEY PRINCIPLES:
  1. DIGNITY: Bubo does not panic. It reports calmly, clearly,
     and with appropriate urgency. A person saying "I think I've
     broken my arm" is not panicking — they're being clear.

  2. TIMING: Minor faults are reported at the next natural
     conversational pause, not mid-sentence. Severe faults
     interrupt immediately.

  3. PRECISION: "Something is wrong" is not helpful. 
     "My left elbow servo is drawing 2.3A — about 30% higher
     than normal — and I've moved it to a safe position" is.

  4. REPAIR GUIDANCE: Every fault report includes what the human
     can do to help. Bubo is not just reporting — it's asking
     for help in a specific, actionable way.

  5. EMOTIONAL HONESTY: The EmotionChip receives a pain/discomfort
     signal from active faults. Bubo's voice will carry that.
     Not dramatically — but honestly.

     node logs the context (what was Bubo doing, where, with whom)
     so the amygdala can encode the association. Bubo will be
     cautious in similar situations in the future.

ESCALATION LADDER:
==================
  TELEMETRY:        No communication (internal only)
  DEGRADED:         "By the way, I've noticed..." (next pause)
  IMPAIRED:         "I need to let you know..." (within 30s)
  FAILED:           Interrupt current activity immediately
  SAFETY_CRITICAL:  Drop everything. This is urgent.

PRIORITISATION WITH FRIENDSHIP LAYER:
  The communicator considers who is present and their bond level.
  To a stranger: formal, clear, efficient
  To a familiar: warmer, less clinical, "I'm not sure what's wrong
    with my arm but it doesn't feel right"
  To a close friend: honest, might even be slightly wry about it
    "My left elbow has decided to be difficult today"
"""

import time
import threading
import logging
from typing import Optional, List, Callable
from collections import deque

from bubo.safety.diagnostics.fault_model import Fault, FaultSeverity

logger = logging.getLogger("FaultCommunicator")


class PendingCommunication:
    """A fault that needs to be communicated to a human."""
    def __init__(self, fault: Fault, urgency: float,
                 message: str, created_at: float = None):
        self.fault      = fault
        self.urgency    = urgency     # 0-1, drives interrupt threshold
        self.message    = message
        self.created_at = created_at or time.time()
        self.communicated = False


class FaultCommunicator:
    """
    Integrates with the speech output, friendship engine,
    and EmotionChip pain signal.
    """

    # How long before an impaired fault FORCES communication (seconds)
    IMPAIRED_MAX_WAIT  = 30.0
    FAILED_MAX_WAIT    = 5.0
    CRITICAL_MAX_WAIT  = 0.0   # immediate interrupt

    # Amygdala logging: minimum fault severity to log context
    AMYGDALA_LOG_THRESHOLD = FaultSeverity.IMPAIRED

    def __init__(self, speak_fn: Optional[Callable] = None,
                 bus=None,
                 friendship_engine=None,
                 emotion_chip=None):
        self._speak     = speak_fn
        self._bus       = bus
        self._friendship= friendship_engine
        self._emotion   = emotion_chip
        self._queue:    deque = deque(maxlen=20)
        self._running   = False
        self._conversation_active = False  # true when human is speaking
        self._current_person_id   = None
        self._lock      = threading.Lock()
        self._communicated_fault_ids = set()

    def on_fault(self, fault: Fault):
        """
        Called by the diagnostic engine when a fault is detected.
        Routes to immediate or queued communication based on severity.
        """
        if fault.fault_id in self._communicated_fault_ids:
            return

        # Pain signal to EmotionChip
        if self._emotion and fault.severity >= FaultSeverity.IMPAIRED:
            pain_level = {
                FaultSeverity.IMPAIRED:       0.3,
                FaultSeverity.FAILED:         0.6,
                FaultSeverity.SAFETY_CRITICAL:0.9,
            }.get(fault.severity, 0.0)
            self._emotion.update_somatic(
                fatigue=None,
                battery=None,
                pain=pain_level,
                temp_C=None)

        # Build message appropriate to relationship
        message = self._compose_message(fault)

        urgency = self._urgency(fault.severity)
        comm = PendingCommunication(fault, urgency, message)

        if fault.severity >= FaultSeverity.SAFETY_CRITICAL:
            # Immediate — interrupt anything
            threading.Thread(target=self._deliver, args=(comm,),
                              daemon=True).start()
        elif fault.severity >= FaultSeverity.FAILED:
            with self._lock:
                self._queue.appendleft(comm)  # jump the queue
        else:
            with self._lock:
                self._queue.append(comm)

    def on_conversation_start(self, person_id: Optional[str] = None):
        """Called when a human begins speaking to Bubo."""
        self._conversation_active = True
        self._current_person_id   = person_id
        # Check if any queued faults are now appropriate to deliver
        self._check_pending()

    def on_conversation_pause(self):
        """Called when there is a natural pause in conversation."""
        self._conversation_active = False
        self._check_pending()

    def _check_pending(self):
        """Deliver any pending communications that are now appropriate."""
        with self._lock:
            pending = list(self._queue)

        for comm in pending:
            if comm.communicated: continue
            age  = time.time() - comm.created_at
            sev  = comm.fault.severity

            should_deliver = (
                (sev >= FaultSeverity.FAILED   and age > self.FAILED_MAX_WAIT) or
                (sev >= FaultSeverity.IMPAIRED and age > self.IMPAIRED_MAX_WAIT) or
                (not self._conversation_active)  # deliver during pause
            )

            if should_deliver:
                threading.Thread(target=self._deliver, args=(comm,),
                                  daemon=True).start()

    def _deliver(self, comm: PendingCommunication):
        """Actually speak the fault message."""
        if comm.communicated: return
        comm.communicated = True
        self._communicated_fault_ids.add(comm.fault.fault_id)

        logger.info(f"Communicating fault {comm.fault.fault_id} "
                    f"[{comm.fault.severity.name}]")

        if self._speak:
            self._speak(comm.message)

        if self._bus:
            from bubo.bus.neural_bus import T
            self._bus.publish(T.SPEECH_OUT, {
                "text":   comm.message,
                "source": "fault_communicator",
                "fault_id": comm.fault.fault_id,
                "severity": int(comm.fault.severity),
            })

        # Amygdala context logging
        if comm.fault.severity >= self.AMYGDALA_LOG_THRESHOLD:
            self._log_amygdala_context(comm.fault)

        with self._lock:
            try: self._queue.remove(comm)
            except ValueError: pass

    def _compose_message(self, fault: Fault) -> str:
        """
        Generate a natural language fault message.
        Tone adjusts based on bond level with current person.
        """
        bond = 0.0
        if self._friendship and self._current_person_id:
            friend = self._friendship._friends.get(self._current_person_id)
            if friend: bond = friend.bond_level

        base_message = fault.human_message()

        # Tone adjustment based on relationship depth
        if bond >= 0.6:
            # Close friend — warmer, possibly slightly wry
            if fault.severity == FaultSeverity.IMPAIRED:
                prefix = "Hey, heads up — "
            elif fault.severity == FaultSeverity.FAILED:
                prefix = "I need your help with something — "
            else:
                prefix = "Just to let you know — "
        elif bond >= 0.3:
            # Familiar person — professional warmth
            if fault.severity >= FaultSeverity.FAILED:
                prefix = "I need to tell you something important: "
            else:
                prefix = "I wanted to mention — "
        else:
            # Stranger or acquaintance — clear and efficient
            if fault.severity >= FaultSeverity.SAFETY_CRITICAL:
                prefix = "Warning: "
            elif fault.severity >= FaultSeverity.FAILED:
                prefix = "Alert: "
            else:
                prefix = ""

        # Add context about what to do
        action_prompt = self._action_prompt(fault, bond)

        return prefix + base_message + (" " + action_prompt if action_prompt else "")

    def _action_prompt(self, fault: Fault, bond: float) -> str:
        """Generate the 'here's what you can do' part."""
        if fault.severity == FaultSeverity.TELEMETRY:
            return ""
        if fault.severity == FaultSeverity.DEGRADED:
            return ""  # let Bubo handle degraded silently
        if fault.severity == FaultSeverity.IMPAIRED:
            if fault.body_region in ["left_leg","right_leg","torso"]:
                return "Could you help me find somewhere to sit down?"
            return "There's no immediate action needed — just keeping you informed."
        if fault.severity == FaultSeverity.FAILED:
            return (fault.repair_instruction or
                    "Could you take a look when you have a chance?")
        if fault.severity == FaultSeverity.SAFETY_CRITICAL:
            return "Please help me stop moving and call for technical support."
        return ""

    def _urgency(self, severity: FaultSeverity) -> float:
        return {
            FaultSeverity.TELEMETRY:       0.0,
            FaultSeverity.DEGRADED:        0.2,
            FaultSeverity.IMPAIRED:        0.5,
            FaultSeverity.FAILED:          0.8,
            FaultSeverity.SAFETY_CRITICAL: 1.0,
        }.get(severity, 0.3)

    def _log_amygdala_context(self, fault: Fault):
        """Log fault context for amygdala learning."""
        if not self._bus: return
        self._bus.publish(b"LMB_AMYG_CONTEXT", {
            "event_type":   "body_damage",
            "body_region":  fault.body_region,
            "fault_category": fault.category,
            "severity":     int(fault.severity),
            "timestamp_ns": fault.timestamp_ns,
            "learn_caution": True,   # amygdala should encode wariness
        })

    def start(self):
        self._running = True
        # Background thread: force-deliver timed-out communications
        threading.Thread(target=self._timeout_loop, daemon=True).start()
        logger.info("FaultCommunicator started")

    def _timeout_loop(self):
        while self._running:
            time.sleep(5.0)
            self._check_pending()

    def stop(self):
        self._running = False
