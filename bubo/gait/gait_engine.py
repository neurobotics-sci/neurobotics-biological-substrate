"""
gait/gait_engine.py — v0.33
4-Limb Bipedal Gait Engine — BeagleBoard Black

BIOLOGICAL BASIS
─────────────────
Human bipedal locomotion is not simply two-legged walking.
It is a 4-limb coordinated pattern where arm swing is essential:
  - Arms provide ~30% of the propulsive energy (reaction cancellation)
  - Arm swing reduces trunk rotation by 75% (Elftman 1939)
  - Contralateral arm-leg coupling: L-arm swings with R-leg (trot pattern)
  - Without arm swing, metabolic cost rises ~12% and stability decreases

GAIT PHASES (biological):
  Stance phase: foot on ground (~62% of gait cycle at walking speed)
  Swing phase:  foot in air   (~38% of gait cycle)
  Double support: both feet on ground (~12% each side) — uniquely human
  Loading response → Mid-stance → Terminal stance → Pre-swing → Swing

ZERO MOMENT POINT (ZMP) stability:
  ZMP = point where the net ground reaction moment is zero.
  For stable walking: ZMP must remain within the support polygon.
  A biped falls when ZMP exits the convex hull of the stance foot/feet.
  ZMP is computed from accelerometer data (vestibular → ANS → cerebellum).

CPG COUPLING PATTERN (4-limb trot-walk hybrid):
  Diagonal pairs in phase: (RL, LL) and (RA, LA) offset by π
    Phase relationships (trot, 180° pairs):
      RL (right leg):   0°
      LA (left arm):    0°   ← contralateral coupling
      LL (left leg):   180°
      RA (right arm):  180°  ← contralateral coupling
  At low speed (walk): more asymmetric, longer double-support

BALANCE RECOVERY (IMU-driven):
  Trunk lean (roll/pitch from complementary filter) → PD controller
  → ankle dorsiflexion/plantarflexion correction
  → hip abduction/adduction (coronal plane balance)
  Gain scheduling: larger gains at faster speeds

FOOT TRAJECTORY (parabolic swing):
  Foot height: h(t) = 4·H_max · t·(1-t)    t ∈ [0,1] in swing phase
  H_max = 0.05m at walk, 0.12m at run
  Foot forward position follows linear interpolation from toe-off to heel-strike
"""

import time, json, logging, threading, numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Tuple

logger = logging.getLogger("GaitEngine")


# ── Biomechanical constants ───────────────────────────────────────────────────

@dataclass
class HumanoidGeometry:
    """Humanoid robot dimensions (approximate human adult proportions)."""
    height_m:       float = 1.70
    mass_kg:        float = 60.0
    leg_length_m:   float = 0.90   # greater trochanter to ground
    thigh_m:        float = 0.43   # femur length
    shank_m:        float = 0.41   # tibia + fibula length
    foot_length_m:  float = 0.27
    arm_length_m:   float = 0.72   # shoulder to fingertip
    upper_arm_m:    float = 0.32
    forearm_m:      float = 0.28   # includes hand
    shoulder_width_m: float = 0.40
    hip_width_m:    float = 0.28
    com_height_m:   float = 0.95   # centre of mass height (standing)
    ankle_height_m: float = 0.07   # ankle joint height
    g_ms2:          float = 9.81


@dataclass
class GaitParameters:
    """Gait control parameters — speed-dependent."""
    speed_ms:       float = 0.4    # walking speed m/s
    cadence_hz:     float = 0.9    # step frequency (steps/second)
    step_length_m:  float = 0.45   # stride length (0.5 × gait cycle length)
    swing_fraction: float = 0.38   # fraction of cycle in swing phase
    foot_clearance_m: float = 0.04 # max foot height during swing
    arm_swing_deg:  float = 30.0   # shoulder flexion/extension amplitude
    trunk_sway_deg: float = 4.0    # lateral trunk sway

    @classmethod
    def from_speed(cls, speed_ms: float) -> "GaitParameters":
        """Speed-dependent gait parameters (Winter 1991 regression)."""
        s = float(np.clip(speed_ms, 0.0, 2.5))
        return cls(
            speed_ms       = s,
            cadence_hz     = 0.5 + 0.4 * s,           # 0.5–1.5 Hz
            step_length_m  = max(0.0, 0.28 + 0.28 * s),
            swing_fraction = 0.38 + 0.04 * (s - 0.4),  # more swing at speed
            foot_clearance_m = 0.04 + 0.04 * s,
            arm_swing_deg  = 20 + 15 * s,
            trunk_sway_deg = max(0.5, 5 - 2 * s),       # less sway at speed
        )


# ── Limb phase state ──────────────────────────────────────────────────────────

@dataclass
class LimbState:
    """Phase and kinematic state of one limb."""
    name:       str
    phase:      float = 0.0        # 0–2π, continuous
    in_stance:  bool  = True
    foot_xy:    np.ndarray = field(default_factory=lambda: np.zeros(3))
    joint_angles: np.ndarray = field(default_factory=lambda: np.zeros(6))


# ── ZMP (Zero Moment Point) calculator ───────────────────────────────────────

class ZMPCalculator:
    """
    Computes the Zero Moment Point from IMU acceleration + current pose.
    ZMP stability criterion: ZMP must stay inside the support polygon.

    ZMP formula (Vukobratovic & Borovac 2004):
      x_ZMP = x_CoM - (z_CoM / g) × ẍ_CoM
      y_ZMP = y_CoM - (z_CoM / g) × ÿ_CoM

    where ẍ, ÿ = horizontal CoM accelerations from IMU (minus gravity).
    """

    def __init__(self, com_height_m: float = 0.95, g: float = 9.81):
        self._H = com_height_m
        self._g = g
        self._pos_hist = deque(maxlen=10)

    def compute(self, com_pos: np.ndarray,
                accel_ms2: np.ndarray) -> np.ndarray:
        """
        com_pos:   [x, y, z] centre of mass position (metres)
        accel_ms2: [ax, ay, az] linear acceleration (m/s², gravity-subtracted)
        Returns:   [x_zmp, y_zmp] ZMP position in world frame
        """
        z = max(com_pos[2], 0.1)
        x_zmp = com_pos[0] - (z / self._g) * accel_ms2[0]
        y_zmp = com_pos[1] - (z / self._g) * accel_ms2[1]
        return np.array([x_zmp, y_zmp])

    def is_stable(self, zmp: np.ndarray,
                  support_polygon: np.ndarray) -> bool:
        """
        Check if ZMP is inside the support polygon (convex hull of contact points).
        Simple 2-D point-in-polygon test using cross products.
        """
        n = len(support_polygon)
        for i in range(n):
            a = support_polygon[i]
            b = support_polygon[(i + 1) % n]
            cross = (b[0] - a[0]) * (zmp[1] - a[1]) - \
                    (b[1] - a[1]) * (zmp[0] - a[0])
            if cross < 0:
                return False
        return True


# ── Foot trajectory planner ───────────────────────────────────────────────────

class FootTrajectory:
    """
    Generates foot position in world frame during swing phase.
    Parabolic height profile: h(t) = 4·H·t·(1-t)
    Forward motion: linear from toe-off to heel-strike target.
    """

    def __init__(self, h_max: float = 0.05):
        self._H = h_max

    def position(self, t_norm: float,
                 start_xy: np.ndarray,
                 target_xy: np.ndarray) -> np.ndarray:
        """
        t_norm: 0=toe-off, 1=heel-strike
        Returns [x, y, z] foot position
        """
        t = float(np.clip(t_norm, 0, 1))
        xy = start_xy + t * (target_xy - start_xy)
        z  = 4.0 * self._H * t * (1.0 - t)   # parabolic
        return np.array([xy[0], xy[1], z])


# ── IK solver (biomimetic leg) ────────────────────────────────────────────────

class LegIKSolver:
    """
    Analytic inverse kinematics for 3-DOF leg (hip-knee-ankle in sagittal).
    Simplified 2-D IK extended to 3-D with hip abduction/adduction.

    Hip flexion: θ_h = atan2(foot_x, foot_z) + correction
    Knee:        θ_k from cosine rule
    Ankle:       θ_a for foot flat on ground (functional constraint)

    Biological constraint: knee extension limited (θ_k ∈ [-135°, 0°])
    """

    def __init__(self, thigh_m: float = 0.43, shank_m: float = 0.41):
        self._l1 = thigh_m   # femur
        self._l2 = shank_m   # tibia

    def solve(self, foot_xyz: np.ndarray) -> np.ndarray:
        """
        foot_xyz: foot position relative to hip joint [x, y, z]
        Returns: [hip_flex, hip_abd, hip_rot, knee_flex, ankle_df, subtalar]
        """
        x, y, z = float(foot_xyz[0]), float(foot_xyz[1]), float(foot_xyz[2])

        # Hip abduction/adduction from lateral offset
        hip_abd = float(np.arctan2(y, -z))

        # Project to sagittal plane for hip flex + knee
        r_sagittal = float(np.sqrt(x**2 + z**2))
        r_sagittal = max(r_sagittal, abs(self._l1 - self._l2) + 0.001)
        r_sagittal = min(r_sagittal, self._l1 + self._l2 - 0.001)

        # Cosine rule for knee angle
        cos_knee = float((self._l1**2 + self._l2**2 - r_sagittal**2) /
                         (2 * self._l1 * self._l2))
        cos_knee = float(np.clip(cos_knee, -1, 1))
        knee = float(-np.arccos(cos_knee))  # negative = flexion

        # Hip flexion angle
        alpha = float(np.arctan2(x, -z))
        beta  = float(np.arcsin(np.clip(
            self._l2 * np.sin(-knee) / r_sagittal, -1, 1)))
        hip_flex = alpha - beta

        # Ankle: maintain foot parallel to ground
        ankle = -(hip_flex + knee)

        # Apply biological ROM limits
        hip_flex  = float(np.clip(hip_flex,  np.radians(-20), np.radians(120)))
        knee      = float(np.clip(knee,       np.radians(-135), 0))
        ankle     = float(np.clip(ankle,      np.radians(-50), np.radians(20)))
        hip_abd   = float(np.clip(hip_abd,    np.radians(-45), np.radians(45)))

        return np.array([hip_flex, hip_abd, 0.0, knee, ankle, 0.0])


# ── Arm swing generator ───────────────────────────────────────────────────────

class ArmSwingGenerator:
    """
    Contralateral arm swing coupled to leg phase.
    Shoulder flexion/extension proportional to opposite hip angle.
    Elbow oscillates at half the shoulder amplitude (passive pendulum).

    Biological coupling constant: k ≈ 0.6 (arm amplitude / hip amplitude)
    """

    K_COUPLING = 0.60    # arm-leg phase coupling constant

    def angles(self, leg_phase_rad: float,
               contralateral: bool = True,
               amplitude_deg: float = 30.0) -> np.ndarray:
        """
        leg_phase_rad: phase of OPPOSITE leg (contralateral coupling)
        Returns arm joint angles [shoulder_flex, shoulder_abd, shoulder_rot,
                                   elbow, wrist_fe, wrist_ru, forearm_ps]
        """
        # Shoulder: sinusoidal, in phase with contralateral leg
        phase = leg_phase_rad if contralateral else leg_phase_rad + np.pi
        shoulder_flex = float(np.radians(amplitude_deg) * np.sin(phase))
        shoulder_abd  = float(np.radians(3.0) * np.sin(phase * 2))   # small ABD
        # Elbow: follows shoulder at half frequency (passive swinging motion)
        elbow = float(np.radians(amplitude_deg * 0.4) * (1 - np.cos(phase)) / 2)
        elbow = float(np.clip(elbow, 0, np.radians(90)))
        return np.array([shoulder_flex, shoulder_abd, 0.0, elbow, 0.0, 0.0, 0.0])


# ── Balance recovery ──────────────────────────────────────────────────────────

class BalanceController:
    """
    IMU-driven balance recovery.
    Trunk lean (roll/pitch) → corrective joint angles.

    Primary corrections:
      Roll  → ankle inversion/eversion (subtalar) + hip abduction
      Pitch → ankle dorsiflexion/plantarflexion + hip flexion
    """

    def __init__(self):
        self._kp_roll  = 0.8   # rad/rad proportional gain
        self._kp_pitch = 0.6
        self._kd_roll  = 0.15  # derivative gain (damping)
        self._kd_pitch = 0.12
        self._prev_roll  = 0.0
        self._prev_pitch = 0.0

    def correction(self, roll_rad: float, pitch_rad: float,
                   dt: float = 0.01) -> dict:
        """
        Returns corrective joint angle deltas for both legs.
        """
        droll  = (roll_rad  - self._prev_roll)  / dt
        dpitch = (pitch_rad - self._prev_pitch) / dt
        self._prev_roll  = roll_rad
        self._prev_pitch = pitch_rad

        ankle_corr = float(-self._kp_pitch * pitch_rad - self._kd_pitch * dpitch)
        hip_abd_corr = float(-self._kp_roll * roll_rad  - self._kd_roll  * droll)
        subtalar_corr = float(-0.4 * roll_rad)

        # Apply symmetrically to both legs
        return {
            "ankle_df_pf":   ankle_corr,
            "hip_abd_add":   hip_abd_corr,
            "subtalar":      subtalar_corr,
        }


# ── Gait Engine ───────────────────────────────────────────────────────────────

class GaitEngine:
    """
    4-limb bipedal gait engine.

    Publishes joint angle targets to spinal cord nodes via ZMQ.
    Receives:
      - MLR drive from reticular formation (speed command)
      - Vestibular data (roll/pitch) from A1 node
      - ZMP feedback from force plates or IMU-estimated

    Publishes:
      - T.SPINAL_CPG  → spinal_legs: CPG phase + joint targets
      - Arm swing targets embedded in T.EFF_M1_ARM_* via PFC efference
    """

    HZ = 10   # 10 Hz control loop

    def __init__(self, config: dict):
        self.name = "GaitEngine"
        self.cfg  = config
        self.geom = HumanoidGeometry()
        self.zmp_calc = ZMPCalculator(self.geom.com_height_m)
        self.foot_traj = FootTrajectory()
        self.leg_ik    = LegIKSolver(self.geom.thigh_m, self.geom.shank_m)
        self.arm_sw    = ArmSwingGenerator()
        self.balance   = BalanceController()

        # ZMQ
        import zmq
        self._ctx = zmq.Context()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(f"tcp://*:{config.get('pub_port', 5670)}")
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.RCVTIMEO, 10)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        for ep in config.get("sub_endpoints", []):
            self._sub.connect(ep)

        # State
        self._speed_ms     = 0.0
        self._roll_rad     = 0.0
        self._pitch_rad    = 0.0
        self._accel_ms2    = np.zeros(3)
        self._com_pos      = np.array([0.0, 0.0, self.geom.com_height_m])
        self._global_phase = 0.0   # 0–2π master oscillator
        self._running      = False
        self._lock         = threading.Lock()
        self._t_last       = time.time()

        # Hip origins (world frame, relative to pelvis)
        self._hip_l = np.array([ self.geom.hip_width_m/2, 0, 0])
        self._hip_r = np.array([-self.geom.hip_width_m/2, 0, 0])

    def _on_vestibular(self, payload: dict):
        """Update balance state from IMU."""
        with self._lock:
            self._roll_rad  = float(np.radians(payload.get("roll_deg",  0)))
            self._pitch_rad = float(np.radians(payload.get("pitch_deg", 0)))
            accel = payload.get("linear_accel_g", [0, 0, 0])
            self._accel_ms2 = np.array(accel) * self.geom.g_ms2

    def _on_mlr(self, payload: dict):
        """MLR drive from reticular formation → speed."""
        d = float(payload.get("drive", 0.0))
        with self._lock:
            self._speed_ms = float(np.clip(d * 1.5, 0.0, 2.0))

    def _recv_loop(self):
        """Background thread: receive vestibular + MLR signals."""
        while self._running:
            try:
                parts = self._sub.recv_multipart()
                if len(parts) != 2: continue
                topic, raw = parts
                payload = json.loads(raw.decode()).get("payload", {})
                if topic.startswith(b"AFF_VEST"):
                    self._on_vestibular(payload)
                elif topic.startswith(b"BS_MLR_LOCO"):
                    self._on_mlr(payload)
            except Exception:
                pass

    def _gait_cycle(self, phase: float, gp: GaitParameters) -> dict:
        """
        Compute all joint angles for current global phase.
        Returns dict with arm_l, arm_r, leg_l, leg_r arrays.
        """
        # Limb phases (trot coupling: diagonal pairs in phase)
        # RL + LA in phase, LL + RA offset by π
        ph_rl = phase
        ph_la = phase          # contralateral = in phase with RL
        ph_ll = phase + np.pi  # opposite
        ph_ra = phase + np.pi  # contralateral = in phase with LL

        # ── Right leg ────────────────────────────────────────────────────
        swing_frac = gp.swing_fraction
        t_rl = (ph_rl % (2*np.pi)) / (2*np.pi)   # 0–1 normalised phase
        is_swing_rl = t_rl < swing_frac

        if is_swing_rl:
            t_sw  = t_rl / swing_frac
            target = np.array([gp.step_length_m * 0.5, -self.geom.hip_width_m/2,
                                -self.geom.leg_length_m * 0.95])
            start  = np.array([-gp.step_length_m * 0.5, -self.geom.hip_width_m/2,
                                -self.geom.leg_length_m])
            foot_pos = self.foot_traj.position(t_sw, start[:2], target[:2])
            foot_pos[2] = start[2] + foot_pos[2]   # add height
        else:
            t_st = (t_rl - swing_frac) / (1 - swing_frac)
            foot_x = gp.step_length_m * 0.5 - t_st * gp.step_length_m
            foot_pos = np.array([foot_x, -self.geom.hip_width_m/2,
                                  -self.geom.leg_length_m])

        q_rl = self.leg_ik.solve(foot_pos - np.array([0, -self.geom.hip_width_m/2, 0]))

        # ── Left leg ─────────────────────────────────────────────────────
        t_ll = (ph_ll % (2*np.pi)) / (2*np.pi)
        is_swing_ll = t_ll < swing_frac

        if is_swing_ll:
            t_sw = t_ll / swing_frac
            target = np.array([gp.step_length_m * 0.5, self.geom.hip_width_m/2,
                                -self.geom.leg_length_m * 0.95])
            start  = np.array([-gp.step_length_m * 0.5, self.geom.hip_width_m/2,
                                -self.geom.leg_length_m])
            foot_pos_l = self.foot_traj.position(t_sw, start[:2], target[:2])
            foot_pos_l[2] = start[2] + foot_pos_l[2]
        else:
            t_st = (t_ll - swing_frac) / (1 - swing_frac)
            foot_x = gp.step_length_m * 0.5 - t_st * gp.step_length_m
            foot_pos_l = np.array([foot_x, self.geom.hip_width_m/2,
                                    -self.geom.leg_length_m])

        q_ll = self.leg_ik.solve(foot_pos_l - np.array([0, self.geom.hip_width_m/2, 0]))

        # ── Balance corrections ───────────────────────────────────────────
        with self._lock:
            roll = self._roll_rad; pitch = self._pitch_rad; dt = 0.01
        bal = self.balance.correction(roll, pitch, dt)
        q_rl[4] += bal["ankle_df_pf"]
        q_ll[4] += bal["ankle_df_pf"]
        q_rl[1] += bal["hip_abd_add"]
        q_ll[1] -= bal["hip_abd_add"]   # mirror
        q_rl[5] += bal["subtalar"]
        q_ll[5] -= bal["subtalar"]

        # ── Arm swing ─────────────────────────────────────────────────────
        q_ra = self.arm_sw.angles(ph_rl, contralateral=True,
                                   amplitude_deg=gp.arm_swing_deg)
        q_la = self.arm_sw.angles(ph_ll, contralateral=True,
                                   amplitude_deg=gp.arm_swing_deg)

        # ── ZMP check ─────────────────────────────────────────────────────
        with self._lock:
            accel = self._accel_ms2.copy()
            com   = self._com_pos.copy()
        zmp = self.zmp_calc.compute(com, accel)

        # Support polygon: stance feet only
        stance_feet = []
        if not is_swing_rl:
            stance_feet.append(np.array([float(q_rl[0]) * 0.1,
                                          -self.geom.hip_width_m/2]))
        if not is_swing_ll:
            stance_feet.append(np.array([float(q_ll[0]) * 0.1,
                                           self.geom.hip_width_m/2]))
        if len(stance_feet) >= 2:
            sp = np.array(stance_feet + [stance_feet[0]])
            stable = self.zmp_calc.is_stable(zmp, sp)
        else:
            stable = True  # single support always check separately

        return {
            "leg_r": q_rl.tolist(),
            "leg_l": q_ll.tolist(),
            "arm_r": q_ra.tolist(),
            "arm_l": q_la.tolist(),
            "zmp":   zmp.tolist(),
            "zmp_stable": stable,
            "phase": float(phase),
            "speed_ms": self._speed_ms,
            "swing_rl": is_swing_rl,
            "swing_ll": is_swing_ll,
        }

    def _control_loop(self):
        interval = 1.0 / self.HZ
        import zmq

        while self._running:
            t0 = time.time(); dt = t0 - self._t_last; self._t_last = t0

            with self._lock:
                speed = self._speed_ms

            if speed < 0.05:
                time.sleep(interval); continue

            # Advance master phase
            gp = GaitParameters.from_speed(speed)
            omega = 2 * np.pi * gp.cadence_hz
            self._global_phase = (self._global_phase + omega * dt) % (2 * np.pi)

            result = self._gait_cycle(self._global_phase, gp)

            # Publish to spinal nodes
            msg_bytes = json.dumps({
                "topic": "SPN_CPG",
                "timestamp_ms": t0 * 1000,
                "source": "GaitEngine",
                "target": "broadcast",
                "payload": result,
                "phase": self._global_phase,
                "neuromod": {"DA": 0.5, "NE": 0.2, "5HT": 0.5, "ACh": 0.5},
            }).encode()
            self._pub.send_multipart([b"SPN_CPG", msg_bytes])

            if not result["zmp_stable"]:
                logger.warning(f"ZMP outside support polygon! ZMP={result['zmp']}")

            elapsed = time.time() - t0
            time.sleep(max(0, interval - elapsed))

    def start(self):
        self._running = True
        threading.Thread(target=self._recv_loop,    daemon=True).start()
        threading.Thread(target=self._control_loop, daemon=True).start()
        logger.info(f"{self.name} v0.33 | ZMP | 4-limb CPG | {self.HZ}Hz")

    def stop(self):
        self._running = False
        self._pub.close(); self._sub.close(); self._ctx.term()


if __name__ == "__main__":
    with open("/etc/brain/config.json") as f:
        cfg = json.load(f).get("gait_engine", {
            "pub_port": 5670,
            "sub_endpoints": [
                "tcp://192.168.1.33:5632",  # vestibular
                "tcp://192.168.1.20:5660",  # reticular formation
            ]
        })
    g = GaitEngine(cfg); g.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: g.stop()
