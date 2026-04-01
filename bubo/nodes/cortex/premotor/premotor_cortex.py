"""
bubo/nodes/cortex/premotor/premotor_cortex.py — Bubo v6000

Premotor + Supplementary Motor Area (PM/SMA, BA6) — split from PFC.

BIOLOGY:
  BA6 (premotor cortex) is anterior to M1 (BA4) and divides into:
    Lateral PM (PMd/PMv): externally-cued movements, visually-guided reaching.
      PMd: dorsal PM → arm/reach movements triggered by visual cues.
      PMv: ventral PM → hand shaping, object manipulation, mirror neurons.
    Medial PM (SMA): internally-cued movements, sequences, bimanual coordination.
      SMA proper: movement initiation, preparing sequential actions.
      Pre-SMA: learning new sequences, switching between programs.

  KEY DISTINCTION from M1:
    M1: "fire this muscle now" (execution)
    PM:  "this type of movement in this context" (contingency coding)
    SMA: "these movements in this order" (sequence representation)
    PFC: "this goal, given these constraints" (goal selection)

  READY POTENTIAL (Bereitschaftspotential, Libet 1983):
    SMA activates ~1.5s before voluntary movement begins.
    This is the neural substrate of motor preparation / intention.
    Bubo implements this as: PM generates plans 200-500ms before M1 executes.

  MIRROR NEURON SYSTEM (PMv, Rizzolatti 1992):
    PMv neurons fire both when the monkey performs an action AND when it
    observes the same action. This is the basis of action understanding
    and imitation learning. Bubo: observed human movements → PM motor programs.

BUBO IMPLEMENTATION:
  PM/SMA runs on PFC-R (192.168.1.11) — separate module from PFC goal selection.
  Receives: T.CTX_PFC_CMD (goal/action), T.PARIETAL_TOOL (affordances),
            T.CTX_ASSOC (mirror neuron signal), T.HIPPO_CONTEXT (spatial context)
  Outputs:  T.PM_MOTOR_PLAN (joint targets → M1)
  Rate: 20Hz (planning is slower than execution)
"""

import time, json, logging, threading
import numpy as np
from bubo.shared.bus.neural_bus import NeuralBus, T
from bubo.shared.kinematics.hand_kinematics import OmniHand, GraspMode

logger = logging.getLogger("PremotorCortex")


# Movement program library (PM contingency codes)
MOVEMENT_PROGRAMS = {
    "reach_right": {
        "arm_r": [0.4, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
        "arm_l": [0.0]*7, "grasp_mode": None,
        "duration_ms": 400, "sequence": ["shoulder_init","arm_extend","wrist_orient"],
    },
    "reach_left": {
        "arm_l": [0.4, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
        "arm_r": [0.0]*7, "grasp_mode": None,
        "duration_ms": 400, "sequence": ["shoulder_init","arm_extend"],
    },
    "grasp_power": {
        "arm_r": [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        "grasp_mode": "power",
        "duration_ms": 300, "sequence": ["approach","open_hand","contact","close"],
    },
    "grasp_precision": {
        "arm_r": [0.4, 0.1, 0.0, 0.4, 0.0, 0.1, 0.0],
        "grasp_mode": "pinch_adj",
        "duration_ms": 350, "sequence": ["approach","orient","pinch"],
    },
    "place_object": {
        "arm_r": [0.3, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
        "grasp_mode": "open",
        "duration_ms": 600, "sequence": ["lower","release","retract"],
    },
    "bimanual_open": {
        "arm_l": [0.3, 0.2, 0.0, 0.3, 0.0, 0.0, 0.0],
        "arm_r": [0.3,-0.2, 0.0, 0.3, 0.0, 0.0, 0.0],
        "grasp_mode": None,
        "duration_ms": 500, "sequence": ["both_reach","spread"],
    },
    "rest_posture": {
        "arm_l": [-0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0],
        "arm_r": [-0.1,-0.1, 0.0, 0.1, 0.0, 0.0, 0.0],
        "leg_l": [0.0, 0.0, 0.0, 0.02, 0.0, 0.0],
        "leg_r": [0.0, 0.0, 0.0, 0.02, 0.0, 0.0],
        "grasp_mode": "open",
        "duration_ms": 1000, "sequence": ["lower_arms","stand"],
    },
    "withdraw": {
        "arm_l": [-0.3, 0.2, 0.0, 0.5, 0.0, 0.0, 0.0],
        "arm_r": [-0.3,-0.2, 0.0, 0.5, 0.0, 0.0, 0.0],
        "grasp_mode": "open",
        "duration_ms": 200, "sequence": ["retract"],
    },
}

# SMA sequence encoding: multi-step action programs
SMA_SEQUENCES = {
    "pick_and_place": ["reach_right","grasp_precision","place_object"],
    "push_door":      ["reach_right","grasp_power","step_forward","release"],
    "wave":           ["reach_right","bimanual_open"],
}


class PremotorCortexNode:
    """PM/SMA: contingency coding, sequence planning, mirror neuron integration."""
    HZ = 20

    def __init__(self, config: dict):
        self.name   = "PremotorCortex"
        self.bus    = NeuralBus(self.name, config["pub_port"], config["sub_endpoints"])

        self._current_action = "rest_posture"
        self._affordance     = {}
        self._mirror_action  = None
        self._sma_sequence   = []
        self._seq_step       = 0
        self._seq_timer      = 0.0
        self._da_level       = 0.6
        self._running        = False
        self._lock           = threading.Lock()

    def _on_pfc_cmd(self, msg):
        action = msg.payload.get("action","rest_posture")
        source = msg.payload.get("source","")
        with self._lock:
            if source == "mirror_neuron" and self._mirror_action:
                action = self._mirror_action
            if action != self._current_action:
                self._current_action = action
                # Check if this is a multi-step SMA sequence
                if action in SMA_SEQUENCES:
                    self._sma_sequence = list(SMA_SEQUENCES[action])
                    self._seq_step     = 0
                    self._seq_timer    = time.time()
                else:
                    self._sma_sequence = []

    def _on_tool(self, msg):
        with self._lock:
            self._affordance = msg.payload.get("affordance", {})

    def _on_assoc(self, msg):
        m = msg.payload.get("mirror_action")
        with self._lock:
            if m: self._mirror_action = m

    def _on_da(self, msg):
        with self._lock: self._da_level = float(msg.payload.get("DA", 0.6))

    def _generate_plan(self) -> dict:
        with self._lock:
            action = self._current_action
            seq    = list(self._sma_sequence)
            step   = self._seq_step
            aff    = dict(self._affordance)
            da     = self._da_level

        # If SMA sequence active, pick current step
        if seq and step < len(seq):
            prog_name = seq[step]
            dur = MOVEMENT_PROGRAMS.get(prog_name, {}).get("duration_ms", 300)
            if time.time() - self._seq_timer > dur / 1000.0:
                with self._lock:
                    self._seq_step += 1
                    self._seq_timer = time.time()
        else:
            prog_name = action

        prog = MOVEMENT_PROGRAMS.get(prog_name, MOVEMENT_PROGRAMS["rest_posture"])

        # Select grasp mode from affordance if available
        grasp = prog.get("grasp_mode")
        if aff.get("type") and not grasp:
            aff_type = aff["type"]
            if "power" in aff_type:   grasp = "power"
            elif "precision" in aff_type: grasp = "pinch_adj"
            elif "key" in aff_type:   grasp = "key"

        plan = {
            "arm_l":    prog.get("arm_l",   [0.0]*7),
            "arm_r":    prog.get("arm_r",   [0.0]*7),
            "leg_l":    prog.get("leg_l",   [0.0]*6),
            "leg_r":    prog.get("leg_r",   [0.0]*6),
            "hand_l":   [0.0]*16,
            "hand_r":   [0.0]*16,
            "grasp_mode": grasp,
            "action":   prog_name,
            "da_scale": da,
            "timestamp_ns": time.time_ns(),
        }

        # Scale velocity by DA (high DA = more vigorous movements)
        scale = float(np.clip(0.5 + da * 0.5, 0.3, 1.0))
        plan["arm_l"] = [v * scale for v in plan["arm_l"]]
        plan["arm_r"] = [v * scale for v in plan["arm_r"]]

        return plan

    def _loop(self):
        iv = 1.0 / self.HZ
        while self._running:
            t0   = time.time()
            plan = self._generate_plan()
            self.bus.publish(b"CTX_PM_PLAN", plan)
            time.sleep(max(0, iv - (time.time() - t0)))

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.CTX_PFC_CMD,    self._on_pfc_cmd)
        self.bus.subscribe(T.PARIETAL_TOOL,  self._on_tool)
        self.bus.subscribe(T.CTX_ASSOC,      self._on_assoc)
        self.bus.subscribe(T.DA_VTA,         self._on_da)
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info(f"{self.name} v6000 | PM/SMA | sequences | mirror neurons | {self.HZ}Hz")

    def stop(self): self._running = False; self.bus.stop()

if __name__ == "__main__":
    with open("/etc/bubo/config.json") as f: cfg=json.load(f)["pfc_r"]
    n=PremotorCortexNode(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
