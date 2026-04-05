"""
bubo/shared/kinematics/hand_kinematics.py — Bubo v6000

4-Finger Symmetric "Omnihand" — Novel Grasping Architecture

══════════════════════════════════════════════════════════════════════
DESIGN PHILOSOPHY: THE SYMMETRIC OMNIHAND
══════════════════════════════════════════════════════════════════════

Standard robot hands copy the human 4-finger + opposed-thumb layout.
This has a fundamental limitation: the thumb is anatomically privileged
but mechanically fragile. When the thumb fails, the hand loses most
of its grasping capability.

The Bubo Omnihand uses 4 IDENTICAL fingers arranged at 90° intervals
around a circular palm. Each finger is anatomically neutral — it can
act as a "thumb" (opposer) or "finger" (flexor) depending on task.

ADVANTAGES over human-style hand:
  1. Any finger can oppose any other finger(s)
  2. Cylindrical grasp: all 4 wrap around object simultaneously
  3. Pinch grasp: any 2 adjacent OR opposite fingers
  4. Power grasp: all 4 flex toward palm centre
  5. No single point of failure (lose 1 finger = 75% capability retained)
  6. Symmetric: no handedness issue for tool use

BIOLOGICAL INSPIRATION:
  - Chameleon zygodactyl foot: 2+2 symmetric toe opposition
  - Parrot beak: two independently movable gripping surfaces
  - Industrial parallel gripper: force distribution analysis
  - Human thumb: only the MCP+CMC (2 DOF base) enables opposition;
    the IP joints (1 DOF each) just flex. We apply this to all 4 fingers.

FINGER GEOMETRY (each finger identical):
  ┌──────────────────────────────────────────────────────┐
  │  Palm mounting point                                 │
  │         │                                            │
  │    [MCP_flex / MCP_abd]  ← 2 DOF at base            │
  │         │                                            │
  │    Proximal phalanx (35mm)                           │
  │         │                                            │
  │    [PIP_flex]           ← 1 DOF                      │
  │         │                                            │
  │    Middle phalanx (25mm)                             │
  │         │                                            │
  │    [DIP_flex]           ← 1 DOF                      │
  │         │                                            │
  │    Distal phalanx (20mm) — SENSOR PAD                │
  └──────────────────────────────────────────────────────┘

DOF per finger: 4 (MCP_flex, MCP_abd, PIP, DIP)
DOF per hand: 16 (4 fingers × 4 DOF)
DOF both hands: 32

PALM GEOMETRY (top view):
  Finger 0: North  (0°)
  Finger 1: East   (90°)
  Finger 2: South  (180°)
  Finger 3: West   (270°)

  Grasp modes:
  POWER:     all 4 flex inward (cylindrical object, power grip)
  PINCH_ADJ: F0+F1 (precision grasp, small object)
  PINCH_OPP: F0+F2 (opposition, thin flat object)
  TRIPOD:    F0+F1+F2 (stable tripod = pen grip)
  HOOK:      F1+F3 oppose, F0+F2 flex (hook/carry bag handle)
  KEY:       F0 press laterally against F1 (key grip)

SENSING (per fingertip):
  - 4 capacitive pressure zones (quadrant pressure map)
  - 1 thermistor (NTC, 10kΩ, -20°C to +80°C)
  - 1 force sensor (FSR 402, 0-100N)
  - MCP joint: magnetic encoder (AS5048A) for proprioception
  - PIP/DIP joints: Hall-effect sensor (position estimate)

SERVO SPECIFICATION (Dynamixel XL430 recommended):
  MCP_flex: XL430-W250-T, 1.5Nm stall, continuous rotation
  MCP_abd:  XL430-W250-T, 1.5Nm stall
  PIP:      XC330-T288-T (smaller, 0.5Nm, fits finger profile)
  DIP:      XC330-T288-T (0.5Nm)
  Total servos per hand: 16
  Total both hands: 32

WIRING NOTE:
  All finger servos daisy-chain on a single TTL bus (Dynamixel protocol)
  per hand. Hand L → spinal-arms UART1, Hand R → UART2.
  Sensor signals (ADC pressure, thermistors) route through Galvanic Barrier
  to BeagleBoard-equivalent (handled by spinal-arms node).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


# ── Joint limits (radians) ──────────────────────────────────────────────────

@dataclass
class FingerJointLimits:
    mcp_flex_min: float = -0.1     # slight hyperextension
    mcp_flex_max: float = 1.7      # ~97° flexion (biological max ~90°, robot ~97°)
    mcp_abd_min:  float = -0.6     # 34° abduction (toward adjacent finger)
    mcp_abd_max:  float = 0.6      # 34° abduction (away)
    pip_min:      float = 0.0
    pip_max:      float = 1.6      # ~92° flexion
    dip_min:      float = 0.0
    dip_max:      float = 1.2      # ~69° flexion


FINGER_LIMITS = FingerJointLimits()


# ── Finger state ─────────────────────────────────────────────────────────────

@dataclass
class FingerState:
    """Complete state of one finger (joints + sensors)."""
    finger_id:      int
    side:           str       # 'L' or 'R'
    name:           str       # 'N','E','S','W'
    angle_deg:      float     # palm mounting angle (0/90/180/270)

    # Joint positions (radians)
    mcp_flex:  float = 0.0
    mcp_abd:   float = 0.0
    pip:       float = 0.0
    dip:       float = 0.0

    # Sensor readings
    pressure_N:   np.ndarray = field(default_factory=lambda: np.zeros(4))
    temperature_C: float = 25.0
    force_N:      float = 0.0
    contact:      bool = False

    @property
    def joint_array(self) -> np.ndarray:
        return np.array([self.mcp_flex, self.mcp_abd, self.pip, self.dip])

    @property
    def total_pressure_N(self) -> float:
        return float(np.sum(self.pressure_N))


# ── Forward kinematics ───────────────────────────────────────────────────────

# Phalanx lengths (metres)
PROX_LEN   = 0.035   # proximal phalanx
MID_LEN    = 0.025   # middle phalanx
DIST_LEN   = 0.020   # distal phalanx
PALM_R     = 0.040   # palm radius (finger mounting offset from centre)


def finger_tip_position(state: FingerState) -> np.ndarray:
    """
    Compute 3-D fingertip position in hand frame (palm centre = origin, Z = up).
    Uses sequential rotation matrices for each joint.
    """
    phi   = np.radians(state.angle_deg)   # mounting angle on palm
    mount = np.array([PALM_R * np.cos(phi), PALM_R * np.sin(phi), 0.0])

    # MCP joint: flex rotates about X-axis (local), abd about Z-axis
    # Build rotation: first abduction (about Z), then flexion (about X)
    cos_abd = np.cos(state.mcp_abd); sin_abd = np.sin(state.mcp_abd)
    Rabd = np.array([[cos_abd,-sin_abd,0],[sin_abd,cos_abd,0],[0,0,1]])

    cos_f1 = np.cos(state.mcp_flex); sin_f1 = np.sin(state.mcp_flex)
    Rflex1 = np.array([[1,0,0],[0,cos_f1,-sin_f1],[0,sin_f1,cos_f1]])

    cos_f2 = np.cos(state.pip); sin_f2 = np.sin(state.pip)
    Rflex2 = np.array([[1,0,0],[0,cos_f2,-sin_f2],[0,sin_f2,cos_f2]])

    cos_f3 = np.cos(state.dip); sin_f3 = np.sin(state.dip)
    Rflex3 = np.array([[1,0,0],[0,cos_f3,-sin_f3],[0,sin_f3,cos_f3]])

    # Finger points outward from palm along local Y axis (radial direction)
    # Rotate the radial direction by phi so each finger points outward
    Rphi = np.array([[cos_abd*np.cos(phi), -np.sin(phi), 0],
                     [cos_abd*np.sin(phi),  np.cos(phi), 0],
                     [0,                   0,            1]])

    # Chain kinematics along finger axis (Z-down = finger extending)
    R    = Rabd @ Rflex1
    p1   = mount + R @ np.array([0, PROX_LEN, 0])
    R    = R @ Rflex2
    p2   = p1   + R @ np.array([0, MID_LEN,  0])
    R    = R @ Rflex3
    tip  = p2   + R @ np.array([0, DIST_LEN, 0])
    return tip


# ── Grasp planning ───────────────────────────────────────────────────────────

class GraspMode(Enum):
    OPEN       = "open"
    POWER      = "power"         # cylinder grasp, all 4 fingers
    PINCH_ADJ  = "pinch_adj"     # adjacent fingers (F0+F1)
    PINCH_OPP  = "pinch_opp"     # opposing fingers (F0+F2)
    TRIPOD     = "tripod"        # 3-finger precision (F0+F1+F2)
    HOOK       = "hook"          # carry/hook (DIP+PIP, MCP extended)
    KEY        = "key"           # lateral pinch (F0 presses on F1 side)
    ENVELOPE   = "envelope"      # all 4, conforming to object shape


GRASP_CONFIGS: Dict[GraspMode, List[Dict]] = {
    GraspMode.OPEN: [
        {"mcp_flex":0.0,"mcp_abd":0.0,"pip":0.0,"dip":0.0} for _ in range(4)],
    GraspMode.POWER: [
        {"mcp_flex":1.2,"mcp_abd":0.0,"pip":1.2,"dip":0.8} for _ in range(4)],
    GraspMode.PINCH_ADJ: [
        {"mcp_flex":0.4,"mcp_abd":-0.3,"pip":0.5,"dip":0.3},  # F0 (N) toward E
        {"mcp_flex":0.4,"mcp_abd": 0.3,"pip":0.5,"dip":0.3},  # F1 (E) toward N
        {"mcp_flex":0.0,"mcp_abd": 0.0,"pip":0.0,"dip":0.0},  # F2 open
        {"mcp_flex":0.0,"mcp_abd": 0.0,"pip":0.0,"dip":0.0},  # F3 open
    ],
    GraspMode.PINCH_OPP: [
        {"mcp_flex":0.6,"mcp_abd":0.0,"pip":0.8,"dip":0.4},   # F0 toward S
        {"mcp_flex":0.0,"mcp_abd":0.0,"pip":0.0,"dip":0.0},   # F1 open
        {"mcp_flex":0.6,"mcp_abd":0.0,"pip":0.8,"dip":0.4},   # F2 toward N
        {"mcp_flex":0.0,"mcp_abd":0.0,"pip":0.0,"dip":0.0},   # F3 open
    ],
    GraspMode.TRIPOD: [
        {"mcp_flex":0.5,"mcp_abd": 0.0,"pip":0.6,"dip":0.3},
        {"mcp_flex":0.5,"mcp_abd":-0.2,"pip":0.6,"dip":0.3},
        {"mcp_flex":0.5,"mcp_abd": 0.2,"pip":0.6,"dip":0.3},
        {"mcp_flex":0.0,"mcp_abd": 0.0,"pip":0.0,"dip":0.0},
    ],
    GraspMode.HOOK: [
        {"mcp_flex":0.0,"mcp_abd":0.0,"pip":1.4,"dip":1.0} for _ in range(4)],
    GraspMode.ENVELOPE: [
        {"mcp_flex":0.8,"mcp_abd":0.0,"pip":0.8,"dip":0.5} for _ in range(4)],
}


class OmniHand:
    """
    4-finger symmetric hand for one side (L or R).
    Manages all 16 joints + sensor state.
    """
    FINGER_NAMES  = ['N','E','S','W']
    FINGER_ANGLES = [0, 90, 180, 270]

    def __init__(self, side: str):
        assert side in ('L','R')
        self.side    = side
        self.fingers = [
            FingerState(i, side, self.FINGER_NAMES[i], self.FINGER_ANGLES[i])
            for i in range(4)
        ]
        self._grasp_mode = GraspMode.OPEN

    def set_grasp(self, mode: GraspMode):
        """Command all fingers to a predefined grasp configuration."""
        config = GRASP_CONFIGS[mode]
        for i, (finger, cfg) in enumerate(zip(self.fingers, config)):
            finger.mcp_flex = float(np.clip(cfg["mcp_flex"],
                                            FINGER_LIMITS.mcp_flex_min, FINGER_LIMITS.mcp_flex_max))
            finger.mcp_abd  = float(np.clip(cfg["mcp_abd"],
                                            FINGER_LIMITS.mcp_abd_min, FINGER_LIMITS.mcp_abd_max))
            finger.pip      = float(np.clip(cfg["pip"],
                                            FINGER_LIMITS.pip_min, FINGER_LIMITS.pip_max))
            finger.dip      = float(np.clip(cfg["dip"],
                                            FINGER_LIMITS.dip_min, FINGER_LIMITS.dip_max))
        self._grasp_mode = mode

    def joint_array(self) -> np.ndarray:
        """All 16 joint positions as flat array."""
        return np.concatenate([f.joint_array for f in self.fingers])

    def set_joints(self, arr: np.ndarray):
        """Set all 16 joints from flat array."""
        for i, f in enumerate(self.fingers):
            base = i * 4
            f.mcp_flex = float(np.clip(arr[base],   FINGER_LIMITS.mcp_flex_min, FINGER_LIMITS.mcp_flex_max))
            f.mcp_abd  = float(np.clip(arr[base+1], FINGER_LIMITS.mcp_abd_min,  FINGER_LIMITS.mcp_abd_max))
            f.pip      = float(np.clip(arr[base+2], FINGER_LIMITS.pip_min,      FINGER_LIMITS.pip_max))
            f.dip      = float(np.clip(arr[base+3], FINGER_LIMITS.dip_min,      FINGER_LIMITS.dip_max))

    def update_sensors(self, pressure: List[np.ndarray], temps: List[float], forces: List[float]):
        """Update sensor state from HAL readings."""
        for i, f in enumerate(self.fingers):
            f.pressure_N    = np.array(pressure[i]) if i < len(pressure) else np.zeros(4)
            f.temperature_C = float(temps[i]) if i < len(temps) else 25.0
            f.force_N       = float(forces[i]) if i < len(forces) else 0.0
            f.contact       = f.total_pressure_N > 0.5

    def contact_pattern(self) -> np.ndarray:
        """4-bit contact pattern: which fingers are touching."""
        return np.array([int(f.contact) for f in self.fingers])

    def tip_positions(self) -> List[np.ndarray]:
        """Fingertip positions in hand frame."""
        return [finger_tip_position(f) for f in self.fingers]

    def select_grasp_from_object(self, width_m: float, height_m: float, depth_m: float) -> GraspMode:
        """
        Heuristic grasp selection from object dimensions.
        Biological parallel: grasping affordance detection (parietal cortex IPL).
        """
        max_dim = max(width_m, height_m, depth_m)
        min_dim = min(width_m, height_m, depth_m)

        if max_dim > 0.08:     return GraspMode.POWER      # large object
        if min_dim < 0.005:    return GraspMode.KEY         # thin flat
        if min_dim < 0.015:    return GraspMode.PINCH_OPP   # thin object
        if max_dim < 0.025:    return GraspMode.PINCH_ADJ   # small precision
        if max_dim < 0.06:     return GraspMode.TRIPOD      # medium precision
        return GraspMode.ENVELOPE

    def grasp_force_distribution(self) -> Dict[str, float]:
        """Contact force distribution across active fingers."""
        total = sum(f.force_N for f in self.fingers)
        return {
            f.name: round(f.force_N / max(total, 0.001), 3)
            for f in self.fingers
        }

    @property
    def grasp_mode(self) -> GraspMode: return self._grasp_mode

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "grasp_mode": self._grasp_mode.value,
            "joints": self.joint_array().tolist(),
            "contact": self.contact_pattern().tolist(),
            "forces_N": [f.force_N for f in self.fingers],
            "temps_C":  [f.temperature_C for f in self.fingers],
        }


# ── SERVO IDs for Dynamixel bus ───────────────────────────────────────────────

def hand_servo_map(side: str) -> Dict[str, int]:
    """
    Dynamixel servo ID assignments for one hand.
    IDs 20-35 (left), 36-51 (right).
    F0..F3, each with: MCP_flex(+0), MCP_abd(+1), PIP(+2), DIP(+3)
    """
    base = 20 if side == 'L' else 36
    return {
        f"F{f}_{joint}": base + f*4 + j
        for f in range(4)
        for j, joint in enumerate(["MCP_flex","MCP_abd","PIP","DIP"])
    }
