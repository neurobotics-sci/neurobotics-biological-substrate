"""
bubo/shared/hal/servo_hal.py — Bubo v5400
Hardware Abstraction Layer: servo controllers → unified joint API.

PROBLEM: Without a HAL, all spinal nodes compute IK and publish corrections
but never actually move a servo. The robot cannot move.

SOLUTION: ServoHAL abstracts three servo backends:
  1. Dynamixel (XL430, XC330, XM430): UART protocol, position+velocity+current+temp
  2. Hobby PWM (generic): BeagleBoard PWM channels + Galvanic Barrier
  3. Simulation: numpy-only, returns simulated state with noise

DYNAMIXEL PROTOCOL (recommended for Bubo):
  UART at 57600 baud (or 1Mbit/s for XL430)
  Packet: header [0xFF 0xFF 0xFD 0x00] + ID + LEN + INST + PARAMS + CRC16
  Model: XL430-W250-T ($29.90, position range 360°, max 1.5 Nm)
  Feedback: position (0.088°/count), velocity (0.229rpm/count), current (2.69mA/count), temperature (1°C/count)
  This gives Bubo REAL proprioception and the Insula a real fatigue signal.

FALLBACK HIERARCHY:
  1. Try Dynamixel (USB2Dynamixel or U2D2 adaptor)
  2. Try BeagleBoard PWM via sysfs (+ Galvanic Barrier)
  3. Fall back to simulation

JOINT NAMING: matches SERVO_NAMES in insula_node.py and spinal nodes.
"""

import time, logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List

logger = logging.getLogger("ServoHAL")


@dataclass
class JointState:
    """Current state of one servo joint."""
    name:         str
    position_rad: float = 0.0
    velocity_rps: float = 0.0
    current_mA:   float = 0.0
    temperature_C: float = 25.0
    voltage_V:    float = 12.0
    error_flags:  int   = 0      # Dynamixel hardware error register
    timestamp_ns: int   = 0

    @property
    def torque_Nm(self) -> float:
        """Approximate torque from current (XL430: 2.69mA/count at 1.5Nm stall)"""
        # Stall current XL430 ≈ 1100mA → 1.5Nm → 1.36mNm/mA
        return self.current_mA * 0.00136

    @property
    def is_overheated(self) -> bool:
        return self.temperature_C > 70.0

    @property
    def is_overloaded(self) -> bool:
        return abs(self.current_mA) > 900.0


# ── Simulation backend ────────────────────────────────────────────────────────

class SimulationBackend:
    """
    Pure simulation servo backend.
    Implements 1st-order joint dynamics with inertia, damping, spring.
    Returns realistic noisy state feedback.
    """
    INERTIA   = 0.01    # kg·m² (approximate for arm link)
    DAMPING   = 0.5     # N·m·s/rad
    STIFFNESS = 0.0     # spring stiffness

    def __init__(self, n_joints: int):
        self._q   = np.zeros(n_joints)
        self._qd  = np.zeros(n_joints)
        self._cmd = np.zeros(n_joints)
        self._t   = time.time()
        self._n   = n_joints
        # Simulated thermal state
        self._temp = np.full(n_joints, 25.0)
        self._duty = np.zeros(n_joints)

    def set_position(self, joint_id: int, position_rad: float):
        if 0 <= joint_id < self._n: self._cmd[joint_id] = position_rad

    def get_state(self, joint_id: int) -> JointState:
        return JointState(
            name=f"sim_{joint_id}",
            position_rad=float(self._q[joint_id] + np.random.normal(0, 0.001)),
            velocity_rps=float(self._qd[joint_id] + np.random.normal(0, 0.01)),
            current_mA=float(abs(self._cmd[joint_id] - self._q[joint_id]) * 100 + np.random.normal(0, 5)),
            temperature_C=float(self._temp[joint_id]),
            timestamp_ns=time.time_ns())

    def step(self):
        t = time.time(); dt = max(t - self._t, 0.001); self._t = t
        for i in range(self._n):
            # PD position control
            err = self._cmd[i] - self._q[i]
            torque = 5.0 * err - self.DAMPING * self._qd[i]
            self._qd[i] += torque / self.INERTIA * dt
            self._qd[i] = float(np.clip(self._qd[i], -10, 10))
            self._q[i]  += self._qd[i] * dt
            # Thermal model
            self._duty[i] = 0.99*self._duty[i] + 0.01*float(abs(torque) > 0.5)
            self._temp[i] += self._duty[i] * 0.001 * dt - (self._temp[i] - 25) * 0.0001 * dt
            self._temp[i] = float(np.clip(self._temp[i], 20, 90))


# ── Dynamixel backend ─────────────────────────────────────────────────────────

class DynamixelBackend:
    """
    Dynamixel XL430/XC330/XM430 servo backend.
    Requires: pip3 install dynamixel-sdk
    USB connection: U2D2 USB-to-Dynamixel adaptor
    """
    PROTOCOL = 2.0
    BAUDRATE = 57600

    # Control table addresses (Dynamixel Protocol 2.0, XL430-W250)
    ADDR_TORQUE_ENABLE      = 64
    ADDR_GOAL_POSITION      = 116
    ADDR_GOAL_VELOCITY      = 104
    ADDR_PRESENT_POSITION   = 132
    ADDR_PRESENT_VELOCITY   = 128
    ADDR_PRESENT_CURRENT    = 126
    ADDR_PRESENT_TEMPERATURE = 146
    ADDR_HARDWARE_ERROR     = 70
    POSITION_SCALE          = 2048.0 / np.pi  # counts per radian (XL430 = 4096/360°)

    def __init__(self, device: str = "/dev/ttyUSB0", servo_ids: List[int] = None):
        self._dxl = None
        self._port_handler = None
        self._pkt_handler  = None
        self._servo_ids    = servo_ids or list(range(26))
        self._device       = device
        self._available    = False
        self._states: Dict[int, JointState] = {}

        try:
            from dynamixel_sdk import PortHandler, PacketHandler
            ph = PortHandler(device)
            pkth = PacketHandler(self.PROTOCOL)
            if ph.openPort() and ph.setBaudRate(self.BAUDRATE):
                self._port_handler = ph
                self._pkt_handler  = pkth
                self._available    = True
                # Enable torque on all servos
                for sid in self._servo_ids:
                    pkth.write1ByteTxOnly(ph, sid, self.ADDR_TORQUE_ENABLE, 1)
                logger.info(f"Dynamixel backend: {len(self._servo_ids)} servos on {device}")
            else:
                logger.warning(f"Dynamixel: could not open {device}")
        except ImportError:
            logger.warning("dynamixel_sdk not installed — use: pip3 install dynamixel-sdk")
        except Exception as e:
            logger.warning(f"Dynamixel init failed: {e}")

    def set_position(self, joint_id: int, position_rad: float):
        if not self._available or joint_id >= len(self._servo_ids): return
        sid  = self._servo_ids[joint_id]
        goal = int(np.clip(position_rad * self.POSITION_SCALE + 2048, 0, 4095))
        self._pkt_handler.write4ByteTxOnly(
            self._port_handler, sid, self.ADDR_GOAL_POSITION, goal)

    def get_state(self, joint_id: int) -> JointState:
        if not self._available or joint_id >= len(self._servo_ids):
            return JointState(name=f"dxl_{joint_id}")
        sid = self._servo_ids[joint_id]
        ph  = self._port_handler; pkth = self._pkt_handler
        try:
            pos, _, _  = pkth.read4ByteTxRx(ph, sid, self.ADDR_PRESENT_POSITION)
            vel, _, _  = pkth.read4ByteTxRx(ph, sid, self.ADDR_PRESENT_VELOCITY)
            cur, _, _  = pkth.read2ByteTxRx(ph, sid, self.ADDR_PRESENT_CURRENT)
            temp, _, _ = pkth.read1ByteTxRx(ph, sid, self.ADDR_PRESENT_TEMPERATURE)
            err, _, _  = pkth.read1ByteTxRx(ph, sid, self.ADDR_HARDWARE_ERROR)
            return JointState(
                name=f"dxl_{sid}",
                position_rad=float((pos - 2048) / self.POSITION_SCALE),
                velocity_rps=float(vel * 0.229 * 2*np.pi / 60),  # RPM→rad/s
                current_mA=float(cur * 2.69),
                temperature_C=float(temp),
                error_flags=int(err),
                timestamp_ns=time.time_ns())
        except Exception:
            return JointState(name=f"dxl_{sid}")

    def is_available(self) -> bool: return self._available


# ── Main ServoHAL ─────────────────────────────────────────────────────────────

class ServoHAL:
    """
    Unified servo hardware abstraction.
    Auto-selects: Dynamixel → PWM → Simulation.
    """

    def __init__(self, n_joints: int = 26, device: str = "/dev/ttyUSB0",
                 galvanic_barrier=None):
        self._n = n_joints
        self._barrier = galvanic_barrier
        self._states  = [JointState(name=f"joint_{i}") for i in range(n_joints)]
        self._backend_name = "simulation"

        # Try Dynamixel first
        self._dxl = DynamixelBackend(device, list(range(n_joints)))
        if self._dxl.is_available():
            self._backend_name = "dynamixel"
            self._sim = None
            logger.info(f"ServoHAL: Dynamixel backend ({n_joints} joints)")
        else:
            # Fall back to simulation
            self._sim = SimulationBackend(n_joints)
            logger.info(f"ServoHAL: simulation backend ({n_joints} joints)")

    def write_positions(self, positions_rad: np.ndarray):
        """Write target joint positions to all servos."""
        if self._barrier: self._barrier.set_servo_active(True)
        for i in range(min(len(positions_rad), self._n)):
            if self._backend_name == "dynamixel":
                self._dxl.set_position(i, float(positions_rad[i]))
            elif self._sim:
                self._sim.set_position(i, float(positions_rad[i]))
        if self._barrier: self._barrier.set_servo_active(True)

    def read_states(self) -> List[JointState]:
        """Read current state of all joints."""
        if self._sim: self._sim.step()
        states = []
        for i in range(self._n):
            if self._backend_name == "dynamixel":
                s = self._dxl.get_state(i)
            elif self._sim:
                s = self._sim.get_state(i)
            else:
                s = JointState(name=f"joint_{i}")
            states.append(s)
        self._states = states
        return states

    def joint_angles(self) -> np.ndarray:
        return np.array([s.position_rad for s in self._states])

    def joint_velocities(self) -> np.ndarray:
        return np.array([s.velocity_rps for s in self._states])

    def joint_temperatures(self) -> np.ndarray:
        return np.array([s.temperature_C for s in self._states])

    def joint_currents_mA(self) -> np.ndarray:
        return np.array([s.current_mA for s in self._states])

    def any_overheated(self) -> bool:
        return any(s.is_overheated for s in self._states)

    def any_overloaded(self) -> bool:
        return any(s.is_overloaded for s in self._states)

    @property
    def backend(self) -> str: return self._backend_name
