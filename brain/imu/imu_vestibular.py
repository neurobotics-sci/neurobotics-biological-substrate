"""
imu/imu_vestibular.py — v0.33
IMU → Vestibular Subsystem — BeagleBoard Black

BIOLOGICAL BASIS: VESTIBULAR SYSTEM
─────────────────────────────────────
The vestibular apparatus (membranous labyrinth, CN VIII) contains:

  Semicircular canals (3 pairs, orthogonal):
    - Detect ANGULAR velocity (rotation) — equivalent to gyroscope
    - Fluid-filled tubes; hair cells bend with rotational flow
    - Sensitive range: 0.1–10 Hz (head movements during locomotion)
    - Output: neural firing rate ∝ head angular velocity
    - Adaptation: canals adapt to sustained rotation (cupula deflection)

  Otolith organs (utricle + saccule):
    - Utricle: horizontal linear acceleration + static head tilt
    - Saccule: vertical linear acceleration
    - Output: firing rate ∝ linear acceleration + gravitational component
    - Key property: cannot distinguish tilt from linear acceleration
      (the Tilt-Translation ambiguity problem — same as accelerometer!)

COMPLEMENTARY FILTER (biological correlate):
  The brain resolves the tilt-translation ambiguity using:
  - Low-frequency (<0.2 Hz): trust otolith (tilt dominates)
  - High-frequency (>2 Hz): trust semicircular canals (translation dominates)
  - Crossover region: both contribute
  The complementary filter models this exactly:
    θ̂ = α(θ̂ + ω·dt) + (1-α)·θ_accel
    α = τ/(τ+dt) where τ ≈ 5s is the biological VOR time constant

VESTIBULO-OCULAR REFLEX (VOR):
  The VOR maintains stable gaze during head movement.
  When the head rotates rightward by θ, eyes rotate leftward by θ.
  VOR gain ≈ 0.95 (eyes move 95% of head angle, not quite compensatory).
  Latency: ~7ms (the fastest reflex in the body — shorter than visual pursuit).
  VOR correction is computed here and published for cervical_motor node.

VESTIBULO-SPINAL REFLEX (VSR):
  Compensatory postural reflexes in response to unexpected perturbations.
  Latency: ~60ms (vestibular → brainstem → spinal cord → muscle)
  VSR drives ankle/hip corrections during balance perturbation.
  Published to gait_engine via T.VESTIBULAR.

HARDWARE: MPU-6050 (primary) or BNO055 (preferred — includes fusion)
  Connected via I2C on BeagleBoard:
    SDA: P9.20 (I2C2_SDA)
    SCL: P9.19 (I2C2_SCL)
    VDD: P9.3  (3.3V)
    GND: P9.1
    INT: P8.12 (GPIO1_12 — data-ready interrupt, optional)
"""

import time, json, logging, threading, struct, math
import numpy as np
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger("IMU_Vestibular")


# ── Register maps ─────────────────────────────────────────────────────────────

class MPU6050_REG:
    """MPU-6050 register addresses."""
    I2C_ADDR     = 0x68    # AD0 low (0x69 if AD0 high)
    PWR_MGMT_1   = 0x6B
    SMPLRT_DIV   = 0x19    # sample rate = 8kHz / (1 + DIV)
    CONFIG       = 0x1A    # DLPF config
    GYRO_CONFIG  = 0x1B    # gyro full-scale range
    ACCEL_CONFIG = 0x1C    # accel full-scale range
    INT_ENABLE   = 0x38
    INT_STATUS   = 0x3A
    ACCEL_XOUT_H = 0x3B    # 6 bytes: AX_H, AX_L, AY_H, AY_L, AZ_H, AZ_L
    TEMP_OUT_H   = 0x41
    GYRO_XOUT_H  = 0x43    # 6 bytes: GX_H, GX_L, GY_H, GY_L, GZ_H, GZ_L
    WHO_AM_I     = 0x75    # should return 0x68

class BNO055_REG:
    """BNO055 register addresses (9-DOF fusion IMU)."""
    I2C_ADDR     = 0x28
    CHIP_ID      = 0x00    # should return 0xA0
    OPR_MODE     = 0x3D
    SYS_STATUS   = 0x39
    UNIT_SEL     = 0x3B
    EUL_HEADING  = 0x1A    # Euler angles: heading, roll, pitch (3×2 bytes, m°)
    LIA_DATA_X   = 0x28    # Linear acceleration (gravity-subtracted, 3×2 bytes)
    GYR_DATA_X   = 0x14    # Gyroscope data (3×2 bytes)
    NDOF_MODE    = 0x0C    # 9-DOF fusion mode


# ── I2C driver abstraction ────────────────────────────────────────────────────

class I2CDevice:
    """Thin wrapper around /dev/i2c-X or smbus2."""

    def __init__(self, bus: int = 2, addr: int = 0x68):
        self._addr = addr
        self._bus  = None
        try:
            import smbus2
            self._bus  = smbus2.SMBus(bus)
            self._smbus = True
            logger.info(f"I2C bus {bus} opened (smbus2), addr=0x{addr:02X}")
        except ImportError:
            logger.warning("smbus2 not available — IMU simulation mode")
            self._smbus = False
        except Exception as e:
            logger.warning(f"I2C open failed ({e}) — simulation mode")
            self._smbus = False

    def write_byte(self, reg: int, val: int):
        if self._smbus and self._bus:
            self._bus.write_byte_data(self._addr, reg, val)

    def read_bytes(self, reg: int, n: int) -> bytes:
        if self._smbus and self._bus:
            return bytes(self._bus.read_i2c_block_data(self._addr, reg, n))
        return bytes(n)   # zeros in sim mode

    def read_word_be(self, reg: int) -> int:
        """Read big-endian signed 16-bit."""
        d = self.read_bytes(reg, 2)
        val = struct.unpack(">h", d)[0]
        return val

    def available(self) -> bool:
        return self._smbus and self._bus is not None


# ── MPU-6050 driver ───────────────────────────────────────────────────────────

class MPU6050Driver:
    """
    MPU-6050 IMU driver.
    Gyro: ±500°/s range → 65.5 LSB/°/s
    Accel: ±4g range    → 8192 LSB/g
    DLPF: 42 Hz bandwidth (rejects high-frequency vibration)
    Sample rate: 100 Hz (DIV = 80 → 8kHz/80 = 100Hz)
    """

    GYRO_SCALE  = 1.0 / 65.5    # °/s per LSB (±500°/s range)
    ACCEL_SCALE = 1.0 / 8192.0  # g per LSB (±4g range)

    def __init__(self, bus: int = 2):
        self._dev = I2CDevice(bus, MPU6050_REG.I2C_ADDR)
        self._sim = not self._dev.available()
        if not self._sim:
            self._init_device()

    def _init_device(self):
        self._dev.write_byte(MPU6050_REG.PWR_MGMT_1,   0x01)  # clock: X gyro
        time.sleep(0.1)
        self._dev.write_byte(MPU6050_REG.SMPLRT_DIV,   79)    # 100 Hz
        self._dev.write_byte(MPU6050_REG.CONFIG,        0x03)  # DLPF 42Hz
        self._dev.write_byte(MPU6050_REG.GYRO_CONFIG,  0x08)  # ±500°/s
        self._dev.write_byte(MPU6050_REG.ACCEL_CONFIG, 0x08)  # ±4g
        self._dev.write_byte(MPU6050_REG.INT_ENABLE,   0x01)  # data-ready
        who = self._dev.read_bytes(MPU6050_REG.WHO_AM_I, 1)[0]
        logger.info(f"MPU-6050 WHO_AM_I=0x{who:02X} ({'OK' if who==0x68 else 'UNEXPECTED'})")

    def read(self) -> dict:
        """Returns raw IMU data in SI units."""
        if self._sim:
            return self._simulate()
        # Read 14 bytes: ACCEL (6) + TEMP (2) + GYRO (6)
        raw = self._dev.read_bytes(MPU6050_REG.ACCEL_XOUT_H, 14)
        ax, ay, az = struct.unpack(">hhh", raw[0:6])
        temp_raw   = struct.unpack(">h",   raw[6:8])[0]
        gx, gy, gz = struct.unpack(">hhh", raw[8:14])

        return {
            "accel_g":    [ax * self.ACCEL_SCALE,
                           ay * self.ACCEL_SCALE,
                           az * self.ACCEL_SCALE],
            "gyro_deg_s": [gx * self.GYRO_SCALE,
                           gy * self.GYRO_SCALE,
                           gz * self.GYRO_SCALE],
            "temp_C":     temp_raw / 340.0 + 36.53,
        }

    def _simulate(self) -> dict:
        """Gentle simulated motion for testing."""
        t = time.time()
        return {
            "accel_g":    [0.02 * np.sin(t * 2.1),
                           0.01 * np.sin(t * 1.7),
                           1.0 + 0.005 * np.sin(t * 3.3)],
            "gyro_deg_s": [1.0 * np.sin(t * 1.3),
                           0.5 * np.cos(t * 0.9),
                           0.3 * np.sin(t * 2.0)],
            "temp_C":     36.5 + 0.1 * np.sin(t * 0.01),
        }


# ── Complementary filter ──────────────────────────────────────────────────────

class VestibularFusion:
    """
    Mahony complementary filter — IMU sensor fusion.
    Matches biological vestibular-otolith integration.

    τ = 5.0s biological time constant (matches human VOR adaptation).
    High-pass gyroscope: eliminates drift (semicircular canal analogue).
    Low-pass accelerometer: corrects long-term bias (otolith analogue).

    Also computes:
      - Linear acceleration (gravity removed) for ZMP and SLAM
      - VOR compensation signal (horizontal and vertical)
      - VSR perturbation detection (sudden jerk → balance recovery)
    """

    VOR_GAIN   = 0.95   # biological VOR gain (eyes move 95% of head)
    TAU        = 5.0    # time constant seconds (biological value)
    JERK_THRESH = 0.15  # m/s² — threshold for unexpected perturbation

    def __init__(self):
        self._roll  = 0.0
        self._pitch = 0.0
        self._yaw   = 0.0   # gyro-integrated only (no magnetometer)
        self._t_last = time.time()
        self._vel   = np.zeros(3)   # velocity estimate (integration)
        self._prev_accel = np.zeros(3)
        self._jerk_hist  = deque(maxlen=20)

    def update(self, accel_g: list, gyro_deg_s: list) -> dict:
        """
        accel_g:    [ax, ay, az] in g (1g ≈ 9.81 m/s²)
        gyro_deg_s: [gx, gy, gz] in °/s
        """
        t  = time.time()
        dt = max(t - self._t_last, 0.001)
        self._t_last = t

        gx, gy, gz = [math.radians(v) for v in gyro_deg_s]
        ax, ay, az = accel_g

        alpha = self.TAU / (self.TAU + dt)

        # Gyro integration (high-pass)
        roll_gyro  = self._roll  + gx * dt
        pitch_gyro = self._pitch + gy * dt
        self._yaw  = (self._yaw + gz * dt)

        # Accel tilt estimate (low-pass) — valid only in near-static
        norm = math.sqrt(ax**2 + ay**2 + az**2)
        if abs(norm - 1.0) < 0.3:   # close to 1g = mostly static
            roll_accel  = math.degrees(math.atan2(ay, az))
            pitch_accel = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))
        else:
            roll_accel  = math.degrees(self._roll)
            pitch_accel = math.degrees(self._pitch)

        # Complementary fusion
        self._roll  = alpha * roll_gyro  + (1 - alpha) * math.radians(roll_accel)
        self._pitch = alpha * pitch_gyro + (1 - alpha) * math.radians(pitch_accel)

        # Linear acceleration (gravity removed)
        g_vec = np.array([
            -math.sin(self._pitch),
             math.sin(self._roll) * math.cos(self._pitch),
             math.cos(self._roll) * math.cos(self._pitch)
        ])
        accel_np  = np.array([ax, ay, az])
        lin_accel = (accel_np - g_vec) * 9.81   # m/s²

        # Jerk detection (unexpected perturbation → VSR)
        jerk = lin_accel - self._prev_accel / dt
        jerk_mag = float(np.linalg.norm(jerk))
        self._jerk_hist.append(jerk_mag)
        self._prev_accel = lin_accel.copy()
        perturbation = jerk_mag > self.JERK_THRESH

        # VOR: corrective eye/head movement
        vor_h = float(-self.VOR_GAIN * gz * dt * 180 / math.pi)
        vor_v = float(-self.VOR_GAIN * gx * dt * 180 / math.pi)

        return {
            "roll_deg":        float(math.degrees(self._roll)),
            "pitch_deg":       float(math.degrees(self._pitch)),
            "yaw_deg":         float(math.degrees(self._yaw)),
            "gyro_rps":        [gx, gy, gz],
            "accel_g":         list(accel_g),
            "linear_accel_g":  (lin_accel / 9.81).tolist(),
            "vor_horiz_deg":   vor_h,
            "vor_vert_deg":    vor_v,
            "jerk_mag":        jerk_mag,
            "perturbation":    perturbation,
            "features": [float(math.degrees(self._roll))/45,
                          float(math.degrees(self._pitch))/45,
                          float(gz), float(norm - 1.0)],
        }


# ── IMU Vestibular Node ───────────────────────────────────────────────────────

class IMUVestibularNode:
    """
    IMU driver + vestibular fusion → neural bus.
    Runs at 100 Hz (IMU sample rate).
    Publishes T.VESTIBULAR with full fusion output.
    """

    HZ = 100

    def __init__(self, config: dict):
        self.name = "IMU_Vestibular"
        self.cfg  = config
        self._imu = MPU6050Driver(bus=config.get("i2c_bus", 2))
        self._filt = VestibularFusion()
        self._running = False

        import zmq
        self._ctx = zmq.Context()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(f"tcp://*:{config.get('pub_port', 5645)}")

    def _publish(self, payload: dict, topic: bytes = b"AFF_VEST"):
        msg = json.dumps({
            "topic": topic.decode(),
            "timestamp_ms": time.time() * 1000,
            "source": self.name,
            "target": "broadcast",
            "payload": payload,
            "phase": 0.0,
            "neuromod": {"DA": 0.5, "NE": 0.2, "5HT": 0.5, "ACh": 0.5},
        }).encode()
        self._pub.send_multipart([topic, msg])

    def _loop(self):
        interval = 1.0 / self.HZ
        while self._running:
            t0 = time.time()
            raw = self._imu.read()
            fused = self._filt.update(raw["accel_g"], raw["gyro_deg_s"])
            fused["temp_C"] = raw.get("temp_C", 36.5)
            fused["sim_mode"] = self._imu._sim
            self._publish(fused)
            # Perturbation → also publish as balance alert
            if fused["perturbation"]:
                self._publish({"type": "perturbation",
                               "jerk_mag": fused["jerk_mag"],
                               **fused}, b"ANS_SYMP")
            elapsed = time.time() - t0
            time.sleep(max(0, interval - elapsed))

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info(
            f"{self.name} v0.33 | MPU-6050 I2C | "
            f"ComplementaryFilter(τ={VestibularFusion.TAU}s) | "
            f"VOR(gain={VestibularFusion.VOR_GAIN}) | "
            f"sim={self._imu._sim} | {self.HZ}Hz")

    def stop(self):
        self._running = False
        self._pub.close(); self._ctx.term()


if __name__ == "__main__":
    with open("/etc/brain/config.json") as f:
        cfg = json.load(f).get("imu_vestibular", {
            "i2c_bus": 2,
            "pub_port": 5645,
        })
    n = IMUVestibularNode(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
