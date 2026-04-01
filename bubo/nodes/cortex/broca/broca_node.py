"""
bubo/nodes/cortex/broca/broca_node.py — v10.17

Broca's Area — Orin Nano 8GB (192.168.1.14)

══════════════════════════════════════════════════════════════════
BIOLOGICAL BASIS
══════════════════════════════════════════════════════════════════

Broca's area comprises Brodmann areas 44 (pars opercularis) and 45
(pars triangularis) in the inferior frontal gyrus (IFG), left hemisphere.

FUNCTIONS:
1. SPEECH PRODUCTION:
   Phonological encoding → articulatory planning → EFF_SPEECH
   Bubo's speech output = internal state verbalisation:
   "battery low", "arm fatigued", "obstacle detected"
   Format: short declarative sentences encoding key state variables.

2. SYNTACTIC PROCESSING:
   Hierarchical phrase structure building (Merge operation — Chomsky).
   Not full language, but the ordering and combination of motor sub-programs.
   "reach → grasp → lift" has the same hierarchical structure as
   "subject → verb → object".

3. MOTOR SEQUENCE PLANNING:
   Broca's area is active during complex manual action sequences.
   Coordinates with premotor cortex for ordered sub-movement execution.
   Bubo: complex tool use, multi-step manipulation planning.

4. MIRROR NEURON INTEGRATION (pars opercularis):
   IFG houses a dense mirror neuron population.
   Observed actions → parsed syntactically → Bubo motor vocabulary.
   Enables imitation learning from human demonstration.

5. WORKING MEMORY FOR SEQUENCES:
   Maintains the currently planned motor sequence in a phonological buffer
   (Baddeley's model: phonological loop ≡ Broca's area).

OUTPUTS:
  T.EFF_SPEECH    → articulatory target (what to say)
  T.BROCA_SYNTAX  → motor sequence plan (ordered sub-movements)
  T.BROCA_MOTSEQ  → next motor segment in current sequence
  T.CTX_PFC_CMD   → complex multi-step action plan to BG

INPUTS:
  T.CTX_PFC_CMD   → PFC action intent (what goal to achieve)
  T.INSULA_STATE  → body state (verbalise if abnormal)
  T.INSULA_FATIGUE→ fatigue → generate "I need rest" utterance
  T.AMYG_CEA_OUT  → fear → "danger" utterance
  T.CTX_ASSOC     → mirror neuron signal → parse observed action
  T.DA_VTA        → motivation (low DA → shorter utterances, less initiative)
"""

import time, json, logging, threading, numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from bubo.shared.bus.neural_bus import NeuralBus, NeuralMessage, T

logger = logging.getLogger("BrocasArea")


# ── Internal state verbalisation ─────────────────────────────────────────────

STATE_UTTERANCES = {
    "battery_critical":   "battery critical — seek charging station",
    "battery_low":        "battery low",
    "thermal_stress":     "overheating — reducing motor activity",
    "fear_high":          "danger detected",
    "fatigue_arm":        "arm servo fatigue — rest required",
    "fatigue_leg":        "leg servo fatigue — stand by",
    "obstacle":           "obstacle in path",
    "task_complete":      "task complete",
    "rest_initiated":     "entering rest mode",
    "repair_required":    "repair behaviour initiated",
    "idle":               "",
}

@dataclass
class SpeechAct:
    """One speech output — internal state verbalisation."""
    utterance:    str
    priority:     float   # 0-1 (safety > self-preservation > task)
    trigger:      str
    timestamp:    float = field(default_factory=time.time)


# ── Motor Sequence Buffer (phonological loop analogue) ────────────────────────

@dataclass
class MotorSegment:
    """One step in a complex action sequence."""
    action:       str
    joint_targets: dict
    duration_ms:  float
    precondition: str = ""   # check before executing


class MotorSequenceBuffer:
    """
    Phonological loop analogue — holds planned motor sequence.
    Broca's area maintains ~7 segments (Miller's law applies to
    sequential motor programs as well as verbal memory).
    """

    CAPACITY = 7

    def __init__(self):
        self._buf: deque = deque(maxlen=self.CAPACITY)
        self._active: Optional[MotorSegment] = None

    def load(self, sequence: List[MotorSegment]):
        self._buf.clear()
        for seg in sequence[:self.CAPACITY]:
            self._buf.append(seg)

    def advance(self) -> Optional[MotorSegment]:
        """Pop and execute next segment."""
        if self._buf:
            self._active = self._buf.popleft()
            return self._active
        return None

    def clear(self): self._buf.clear(); self._active = None

    @property
    def remaining(self) -> int: return len(self._buf)

    @property
    def current(self) -> Optional[MotorSegment]: return self._active


# ── Syntactic motor planner ───────────────────────────────────────────────────

class SyntacticMotorPlanner:
    """
    Hierarchical action decomposition — IFG pars opercularis.
    Maps high-level goals → ordered motor sub-programs.
    Uses a simplified phrase-structure grammar:
      S → NP VP
      VP → V NP  (reach [target], grasp [object], place [location])
    Applied to motor actions:
      reach → approach_phase, arm_extend, wrist_orient
      grasp → fingers_open, contact, fingers_close, lift_check
      place  → move_to_target, lower, release, retract
    """

    # Action grammar (action → list of motor segments)
    ACTION_GRAMMAR: Dict[str, List[dict]] = {
        "reach_right": [
            {"action": "shoulder_init",  "joint_targets": {"shoulder_flex": 0.3},  "duration_ms": 200},
            {"action": "arm_extend",     "joint_targets": {"elbow_flex": -0.4},     "duration_ms": 400},
            {"action": "wrist_orient",   "joint_targets": {"wrist_flex": 0.1},      "duration_ms": 150},
        ],
        "grasp": [
            {"action": "fingers_open",   "joint_targets": {"finger_spread": 0.8},  "duration_ms": 150},
            {"action": "contact",        "joint_targets": {"finger_spread": 0.3},  "duration_ms": 200},
            {"action": "fingers_close",  "joint_targets": {"finger_spread": 0.0},  "duration_ms": 200},
            {"action": "lift_verify",    "joint_targets": {},                        "duration_ms": 100,
             "precondition": "object_in_gripper"},
        ],
        "rest_posture": [
            {"action": "arms_lower",     "joint_targets": {"shoulder_flex": -0.1, "elbow_flex": 0.1},
             "duration_ms": 1000},
            {"action": "head_neutral",   "joint_targets": {"neck_flex": 0.0},      "duration_ms": 500},
            {"action": "balance_lock",   "joint_targets": {"hip_flex": 0.0, "knee_flex": 0.02},
             "duration_ms": 500},
        ],
        "seek_charge": [
            {"action": "orient_charger", "joint_targets": {"gaze": [0, 0, 1]},    "duration_ms": 500},
            {"action": "approach",       "joint_targets": {"gait_mode": "walk"},   "duration_ms": 5000},
            {"action": "dock",           "joint_targets": {"shoulder_flex": 0.2},  "duration_ms": 2000},
        ],
    }

    def plan(self, goal: str) -> List[MotorSegment]:
        template = self.ACTION_GRAMMAR.get(goal, [])
        return [MotorSegment(**seg) for seg in template]


# ── Broca Node ────────────────────────────────────────────────────────────────

class BrocasAreaNode:
    """
    Broca's area — speech production, motor sequencing, mirror integration.
    """

    SEQ_HZ  = 20   # sequence execution check rate
    PUB_HZ  = 2    # speech act publish rate

    def __init__(self, config: dict):
        self.name   = "BrocasArea"
        self.bus    = NeuralBus(self.name, config["pub_port"],
                                config["sub_endpoints"])
        self.planner = SyntacticMotorPlanner()
        self.seq_buf = MotorSequenceBuffer()

        # State
        self._da_level      = 0.6
        self._fear_level    = 0.0
        self._fatigue_state = {}
        self._insula_state  = {}
        self._mirror_action = None
        self._pending_speech: Optional[SpeechAct] = None
        self._current_goal  = None
        self._running       = False
        self._lock          = threading.Lock()

    # ── Handlers ─────────────────────────────────────────────────────────────

    def _on_pfc_cmd(self, msg):
        """PFC goal → decompose into motor sequence."""
        action = msg.payload.get("action", "")
        src    = msg.payload.get("source", "pfc")
        if src == "mirror_neuron" and self._mirror_action:
            action = self._mirror_action
        if action and action != self._current_goal:
            segs = self.planner.plan(action)
            if segs:
                self.seq_buf.load(segs)
                self._current_goal = action
                logger.debug(f"Loaded sequence: {action} ({len(segs)} segments)")

    def _on_insula_fatigue(self, msg):
        """Fatigue → generate rest utterance + rest_posture sequence."""
        p = msg.payload
        with self._lock:
            self._fatigue_state = p
        if float(p.get("global_fatigue", 0)) > 0.70:
            speech = SpeechAct("arm servo fatigue — rest required",
                                priority=0.85, trigger="fatigue")
            with self._lock:
                if self._pending_speech is None or speech.priority > self._pending_speech.priority:
                    self._pending_speech = speech
            segs = self.planner.plan("rest_posture")
            self.seq_buf.load(segs)
            self._current_goal = "rest_posture"

    def _on_insula(self, msg):
        with self._lock: self._insula_state = msg.payload

    def _on_cea(self, msg):
        cea = float(msg.payload.get("cea_activation", 0))
        with self._lock: self._fear_level = cea
        if cea > 0.6:
            sp = SpeechAct("danger detected", priority=0.95, trigger="fear")
            with self._lock:
                if self._pending_speech is None or sp.priority > self._pending_speech.priority:
                    self._pending_speech = sp

    def _on_da(self, msg):
        with self._lock: self._da_level = float(msg.payload.get("DA", 0.6))

    def _on_hypo(self, msg):
        p = msg.payload
        mode = p.get("mode", "normal")
        priority = {"emergency": 0.99, "survival": 0.85,
                    "cautious": 0.4}.get(mode, 0.0)
        utt = {"emergency": "battery critical — seek charging station",
               "survival":  "battery low"}.get(mode)
        if utt:
            sp = SpeechAct(utt, priority=priority, trigger="battery")
            with self._lock:
                if self._pending_speech is None or sp.priority > self._pending_speech.priority:
                    self._pending_speech = sp

    def _on_assoc(self, msg):
        """Mirror neuron action observation."""
        with self._lock:
            self._mirror_action = msg.payload.get("mirror_action")

    def _on_rest_repair(self, msg):
        """Insula/safety rest trigger → plan rest posture."""
        segs = self.planner.plan("rest_posture")
        self.seq_buf.load(segs)
        self._current_goal = "rest_posture"
        sp = SpeechAct("entering rest mode", 0.80, "rest_repair")
        with self._lock: self._pending_speech = sp

    # ── Sequence execution loop ───────────────────────────────────────────────

    def _seq_loop(self):
        interval = 1.0 / self.SEQ_HZ
        t_last_speech = 0.0
        while self._running:
            t0 = time.time()

            # Execute next motor segment
            seg = self.seq_buf.advance()
            if seg:
                self.bus.publish(T.BROCA_MOTSEQ, {
                    "action":       seg.action,
                    "joint_targets":seg.joint_targets,
                    "duration_ms":  seg.duration_ms,
                    "goal":         self._current_goal,
                    "remaining":    self.seq_buf.remaining,
                })
                # If action has sub-sequence, publish to PFC as plan
                self.bus.publish(T.BROCA_SYNTAX, {
                    "segment":  seg.action,
                    "goal":     self._current_goal,
                    "depth":    1,
                    "is_leaf":  True,
                })

            # Publish speech if pending (rate-limited by priority)
            with self._lock:
                sp = self._pending_speech
            now = time.time()
            if sp and (now - t_last_speech) > (2.0 / max(sp.priority, 0.1)):
                da = self._da_level
                # Low DA → shorter utterances, less verbose
                utterance = sp.utterance if da > 0.3 else sp.utterance.split("—")[0].strip()
                self.bus.publish(T.EFF_SPEECH, {
                    "utterance":  utterance,
                    "priority":   sp.priority,
                    "trigger":    sp.trigger,
                    "da_level":   da,
                })
                self.bus.publish(T.BROCA_SPEECH_ACT, {
                    "text":      utterance,
                    "timestamp": now,
                })
                t_last_speech = now
                with self._lock: self._pending_speech = None

            elapsed = time.time() - t0
            time.sleep(max(0, interval - elapsed))

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.CTX_PFC_CMD,     self._on_pfc_cmd)
        self.bus.subscribe(T.INSULA_FATIGUE,  self._on_insula_fatigue)
        self.bus.subscribe(T.INSULA_STATE,    self._on_insula)
        self.bus.subscribe(T.AMYG_CEA_OUT,   self._on_cea)
        self.bus.subscribe(T.DA_VTA,          self._on_da)
        self.bus.subscribe(T.HYPO_STATE,      self._on_hypo)
        self.bus.subscribe(T.CTX_ASSOC,       self._on_assoc)
        self.bus.subscribe(T.REST_REPAIR,     self._on_rest_repair)
        self._running = True
        threading.Thread(target=self._seq_loop, daemon=True).start()
        logger.info(f"{self.name} v10.17 | IFG BA44/BA45 | motor-seq | speech-acts")

    def stop(self): self._running = False; self.bus.stop()


if __name__ == "__main__":
    import sys
    with open("/etc/brain/config.json") as f: cfg = json.load(f)["broca"]
    n = BrocasAreaNode(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
