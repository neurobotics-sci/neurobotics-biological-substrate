"""
bubo/hw/servo/hal.py — Bubo Unified V10
Hardware Abstraction Layer: profile-aware servo/GPIO access.
Real on hardware profiles, simulation on AWS profiles.
"""
import time, logging
import numpy as np
from bubo.shared.profile import profile

logger = logging.getLogger("ServoHAL")

HW_ACTIVE = profile.hardware.servos


class DynamixelHAL:
    """
    Servo HAL: real Dynamixel calls on hardware, simulation on cloud.
    Brain modules call this identically regardless of profile.
    """
    def __init__(self, device: str = "/dev/ttyUSB0", baud: int = 1_000_000):
        self._device = device
        self._baud   = baud
        self._positions: dict = {}
        self._simulated = not HW_ACTIVE
        self._handler   = None

        if not self._simulated:
            self._init_hardware()
        else:
            logger.info("ServoHAL: simulation mode (profile has no servo hardware)")

    def _init_hardware(self):
        try:
            from dynamixel_sdk import PortHandler, PacketHandler
            self._ph  = PortHandler(self._device)
            self._pkt = PacketHandler(2.0)
            if self._ph.openPort() and self._ph.setBaudRate(self._baud):
                logger.info(f"ServoHAL: Dynamixel on {self._device} @ {self._baud}")
            else:
                logger.warning(f"ServoHAL: port open failed — falling back to simulation")
                self._simulated = True
        except ImportError:
            logger.warning("dynamixel_sdk not installed — simulation mode")
            self._simulated = True

    def set_goal_position(self, servo_id: int, position_rad: float):
        """Set servo goal position. No-op in simulation."""
        if self._simulated:
            self._positions[servo_id] = position_rad
            return
        try:
            counts = int((position_rad + np.pi) / (2*np.pi) * 4095)
            counts = max(0, min(4095, counts))
            self._pkt.write4ByteTxRx(self._ph, servo_id, 116, counts)
        except Exception as e:
            logger.debug(f"Servo {servo_id}: {e}")

    def get_position(self, servo_id: int) -> float:
        """Get current position in radians."""
        if self._simulated:
            return self._positions.get(servo_id, 0.0)
        try:
            val, _, _ = self._pkt.read4ByteTxRx(self._ph, servo_id, 132)
            return float(val) / 4095 * 2*np.pi - np.pi
        except Exception:
            return 0.0

    def set_joints(self, joint_map: dict):
        """Set multiple joints. joint_map: {servo_id: position_rad}"""
        for sid, pos in joint_map.items():
            self.set_goal_position(sid, pos)

    @property
    def is_simulated(self) -> bool:
        return self._simulated


class VagusNerveHAL:
    """
    Hardware kill switch HAL.
    On hardware: controls the physical NE556 relay.
    On cloud: logs the event (no physical relay).
    """
    def __init__(self, gpio_pin: int = 60):
        self._pin     = gpio_pin
        self._active  = profile.hardware.vagus_nerve
        self._gpio    = None
        self._fired   = False

        if self._active:
            try:
                import Adafruit_BBIO.GPIO as GPIO
                GPIO.setup(str(self._pin), GPIO.OUT)
                GPIO.output(str(self._pin), GPIO.HIGH)  # HIGH = relay closed = servos ON
                self._gpio = GPIO
                logger.info(f"VagusNerve: GPIO{self._pin} active")
            except ImportError:
                logger.warning("BeagleBone GPIO not available — vagus in simulation")
                self._active = False

    def fire(self, reason: str = "manual"):
        """Trigger emergency stop."""
        logger.critical(f"VAGUS NERVE FIRED: {reason}")
        self._fired = True
        if self._gpio:
            self._gpio.output(str(self._pin), self._gpio.LOW)
        else:
            logger.critical("[SIMULATION] Servo rail would be cut NOW")

    def reset(self):
        if not self._fired: return
        self._fired = False
        if self._gpio:
            self._gpio.output(str(self._pin), self._gpio.HIGH)
        logger.info("VagusNerve: reset")

    @property
    def fired(self) -> bool: return self._fired
