"""
bubo/nodes/cortex/m1/primary_motor_cortex.py — Bubo v6000

Primary Motor Cortex (M1, BA4) — split from PFC in v6000.

BIOLOGY:
  M1 (Brodmann area 4): direct descending motor commands to spinal cord.
  Each M1 neuron has a "preferred direction" and fires before voluntary movement.
  Corticospinal tract: M1 → internal capsule → pyramidal decussation → α-motor neurons.
  Somatotopic organisation: face (lateral), hand (central/largest area), leg (medial).
  The hand representation alone occupies ~35% of M1 in primates.

  M1 does NOT plan movements — it executes them. Planning is:
    - Premotor cortex (PM, BA6): what movement, in what context
    - SMA (supplementary motor area, BA6): sequences, bimanual coordination
    - PFC: goal selection → tells PM/SMA what to do

  M1 receives from: PM/SMA (movement plans), S1 (sensory feedback for online correction),
  basal ganglia (via thalamus), cerebellum (via thalamus).

BUBO IMPLEMENTATION:
  M1 runs on PFC-L (192.168.1.10) — same node, separate Python module.
  Receives T.PM_MOTOR_PLAN from premotor cortex.
  Sends T.EFF_M1_* (joint-space commands) to spinal nodes.
  Integrates cerebellar corrections (T.CRB_DELTA) before issuing commands.
  Handles online position correction from S1 touch/force feedback.
  At 50Hz — between PM (20Hz planning) and spinal (100Hz execution).
"""

import time, json, logging, threading
import numpy as np
from bubo.shared.bus.neural_bus import NeuralBus, T
from bubo.shared.homunculus.somatosensory_homunculus import SomatosensoryHomunculus
from bubo.shared.kinematics.hand_kinematics import OmniHand, GraspMode

logger = logging.getLogger("PrimaryMotorCortex")


class M1MotorMap:
    """
    Somatotopic motor map — models M1 columnar organisation.
    Each column has a preferred direction vector for its body segment.
    In v6000: hand columns are largest (35% of all columns = 32 hand joints).
    """
    COLUMNS_HEAD   = 8    # face, neck, tongue, eye muscles
    COLUMNS_ARM    = 14   # 7 DOF × 2 arms
    COLUMNS_HAND   = 32   # 16 DOF × 2 hands (Omnihand) — largest region
    COLUMNS_LEG    = 12   # 6 DOF × 2 legs
    COLUMNS_TRUNK  = 6    # torso, spine
    TOTAL          = COLUMNS_HEAD + COLUMNS_ARM + COLUMNS_HAND + COLUMNS_LEG + COLUMNS_TRUNK

    def __init__(self):
        self._positions = np.zeros(self.TOTAL)   # current joint positions
        self._targets   = np.zeros(self.TOTAL)   # commanded targets
        self._velocity  = np.zeros(self.TOTAL)   # velocity commands
        self._gain      = np.ones(self.TOTAL) * 0.3  # output gain per column

    def set_arm_targets(self, arm_l: np.ndarray, arm_r: np.ndarray):
        base = self.COLUMNS_HEAD
        self._targets[base:base+7]    = np.resize(arm_l, 7)
        self._targets[base+7:base+14] = np.resize(arm_r, 7)

    def set_hand_targets(self, hand_l: np.ndarray, hand_r: np.ndarray):
        base = self.COLUMNS_HEAD + self.COLUMNS_ARM
        self._targets[base:base+16]    = np.resize(hand_l, 16)
        self._targets[base+16:base+32] = np.resize(hand_r, 16)

    def set_leg_targets(self, leg_l: np.ndarray, leg_r: np.ndarray):
        base = self.COLUMNS_HEAD + self.COLUMNS_ARM + self.COLUMNS_HAND
        self._targets[base:base+6]    = np.resize(leg_l, 6)
        self._targets[base+6:base+12] = np.resize(leg_r, 6)

    def get_arm_commands(self) -> tuple:
        base = self.COLUMNS_HEAD
        return self._targets[base:base+7], self._targets[base+7:base+14]

    def get_hand_commands(self) -> tuple:
        base = self.COLUMNS_HEAD + self.COLUMNS_ARM
        return self._targets[base:base+16], self._targets[base+16:base+32]

    def get_leg_commands(self) -> tuple:
        base = self.COLUMNS_HEAD + self.COLUMNS_ARM + self.COLUMNS_HAND
        return self._targets[base:base+6], self._targets[base+6:base+12]

    def apply_cerebellar_correction(self, arm_corr: np.ndarray, leg_corr: np.ndarray):
        """Add cerebellar delta corrections (online error compensation)."""
        base_arm = self.COLUMNS_HEAD
        base_leg = self.COLUMNS_HEAD + self.COLUMNS_ARM + self.COLUMNS_HAND
        self._targets[base_arm:base_arm+14] += np.resize(arm_corr, 14) * 0.5
        self._targets[base_leg:base_leg+12] += np.resize(leg_corr, 12) * 0.5

    def apply_s1_correction(self, force_error: np.ndarray):
        """
        Online force feedback: if contact force exceeds target → reduce motor command.
        Biological: M1 online correction via S1 thalamocortical loop (~50ms).
        """
        base_hand = self.COLUMNS_HEAD + self.COLUMNS_ARM
        for i in range(32):
            if i < len(force_error) and abs(force_error[i]) > 0.1:
                self._targets[base_hand+i] -= float(force_error[i]) * 0.2


class PrimaryMotorCortexNode:
    """M1 node: receives premotor plans → issues joint-space motor commands."""
    HZ = 50

    def __init__(self, config: dict):
        self.name   = "M1_PrimaryMotorCortex"
        self.bus    = NeuralBus(self.name, config["pub_port"], config["sub_endpoints"])
        self.m1map  = M1MotorMap()
        self.homo   = SomatosensoryHomunculus()
        self.hand_l = OmniHand('L')
        self.hand_r = OmniHand('R')

        self._pm_arm_l  = np.zeros(7);   self._pm_arm_r  = np.zeros(7)
        self._pm_leg_l  = np.zeros(6);   self._pm_leg_r  = np.zeros(6)
        self._pm_hand_l = np.zeros(16);  self._pm_hand_r = np.zeros(16)
        self._cerb_arm  = np.zeros(14);  self._cerb_leg  = np.zeros(12)
        self._s1_force  = np.zeros(32)
        self._grasp_cmd  = None
        self._motor_inh  = 0.0
        self._running    = False
        self._lock       = threading.Lock()
        self._t_last     = time.time()

    def _on_pm_plan(self, msg):
        """Premotor plan → M1 target positions."""
        p = msg.payload
        with self._lock:
            al = p.get("arm_l", []); ar = p.get("arm_r", [])
            ll = p.get("leg_l", []); lr = p.get("leg_r", [])
            hl = p.get("hand_l", []); hr = p.get("hand_r", [])
            if al: self._pm_arm_l  = np.resize(np.array(al,  dtype=float), 7)
            if ar: self._pm_arm_r  = np.resize(np.array(ar,  dtype=float), 7)
            if ll: self._pm_leg_l  = np.resize(np.array(ll,  dtype=float), 6)
            if lr: self._pm_leg_r  = np.resize(np.array(lr,  dtype=float), 6)
            if hl: self._pm_hand_l = np.resize(np.array(hl,  dtype=float), 16)
            if hr: self._pm_hand_r = np.resize(np.array(hr,  dtype=float), 16)
            gc = p.get("grasp_mode")
            if gc: self._grasp_cmd = gc

    def _on_cerb(self, msg):
        with self._lock:
            self._cerb_arm = np.resize(np.array(msg.payload.get("arm_correction",[0]*14),dtype=float),14)
            self._cerb_leg = np.resize(np.array(msg.payload.get("leg_correction",[0]*12),dtype=float),12)

    def _on_s1(self, msg):
        force = msg.payload.get("hand_forces", [])
        with self._lock:
            self._s1_force = np.resize(np.array(force,dtype=float), 32) if force else np.zeros(32)

    def _on_hypo(self, msg):
        with self._lock: self._motor_inh = float(msg.payload.get("motor_inhibit", 0))

    def _loop(self):
        iv = 1.0 / self.HZ
        while self._running:
            t0 = time.time(); dt = max(t0 - self._t_last, 0.001); self._t_last = t0
            with self._lock:
                al=self._pm_arm_l.copy(); ar=self._pm_arm_r.copy()
                ll=self._pm_leg_l.copy(); lr=self._pm_leg_r.copy()
                hl=self._pm_hand_l.copy(); hr=self._pm_hand_r.copy()
                ca=self._cerb_arm.copy(); cl=self._cerb_leg.copy()
                sf=self._s1_force.copy(); mi=self._motor_inh
                gc=self._grasp_cmd; self._grasp_cmd=None

            scale = float(np.clip(1.0 - mi, 0.0, 1.0))

            # Set targets in M1 map
            self.m1map.set_arm_targets(al * scale, ar * scale)
            self.m1map.set_leg_targets(ll * scale, lr * scale)
            self.m1map.set_hand_targets(hl * scale, hr * scale)

            # Apply cerebellar online correction
            self.m1map.apply_cerebellar_correction(ca, cl)

            # Apply S1 force feedback
            self.m1map.apply_s1_correction(sf)

            # Handle grasp command → set hand joint targets
            if gc:
                try:
                    mode_l = GraspMode(gc); mode_r = GraspMode(gc)
                    self.hand_l.set_grasp(mode_l)
                    self.hand_r.set_grasp(mode_r)
                    hl = self.hand_l.joint_array()
                    hr = self.hand_r.joint_array()
                    self.m1map.set_hand_targets(hl * scale, hr * scale)
                except ValueError: pass

            # Retrieve final commands
            arm_l_cmd, arm_r_cmd = self.m1map.get_arm_commands()
            leg_l_cmd, leg_r_cmd = self.m1map.get_leg_commands()
            hand_l_cmd, hand_r_cmd = self.m1map.get_hand_commands()

            now_ns = time.time_ns()
            self.bus.publish(T.EFF_M1_ARM_L, {"joints": arm_l_cmd.tolist(),  "source":"M1","timestamp_ns":now_ns})
            self.bus.publish(T.EFF_M1_ARM_R, {"joints": arm_r_cmd.tolist(),  "source":"M1","timestamp_ns":now_ns})
            self.bus.publish(T.EFF_M1_LEG_L, {"joints": leg_l_cmd.tolist(),  "source":"M1","timestamp_ns":now_ns})
            self.bus.publish(T.EFF_M1_LEG_R, {"joints": leg_r_cmd.tolist(),  "source":"M1","timestamp_ns":now_ns})
            # Hand commands (new v6000)
            self.bus.publish(b"EFF_HAND_L",  {"joints": hand_l_cmd.tolist(), "grasp":self.hand_l.grasp_mode.value,"source":"M1","timestamp_ns":now_ns})
            self.bus.publish(b"EFF_HAND_R",  {"joints": hand_r_cmd.tolist(), "grasp":self.hand_r.grasp_mode.value,"source":"M1","timestamp_ns":now_ns})

            time.sleep(max(0, iv - (time.time() - t0)))

    def start(self):
        self.bus.start()
        self.bus.subscribe(b"CTX_PM_PLAN",   self._on_pm_plan)
        self.bus.subscribe(T.CEREBELL_DELTA, self._on_cerb)
        self.bus.subscribe(T.TOUCH_SA1,      self._on_s1)
        self.bus.subscribe(T.HYPO_STATE,     self._on_hypo)
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info(f"{self.name} v6000 | M1 somatotopic | Omnihand | {self.HZ}Hz")

    def stop(self): self._running = False; self.bus.stop()

if __name__ == "__main__":
    with open("/etc/bubo/config.json") as f: cfg=json.load(f)["pfc_l"]
    n=PrimaryMotorCortexNode(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
