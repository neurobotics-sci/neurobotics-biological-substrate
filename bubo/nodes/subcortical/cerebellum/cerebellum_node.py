"""
bubo/nodes/subcortical/cerebellum/cerebellum_node.py — v6500
Cerebellum coordinator: CMAC + MPC balance + PPO gait RL.
Integrates all three motor learning/control systems.
"""
import time, json, logging, threading
import numpy as np
from bubo.shared.bus.neural_bus import NeuralBus, T
from bubo.nodes.subcortical.cerebellum.cmac_cerebellum import CMACCerebellumNode
from bubo.balance.mpc.mpc_balance_controller import MPCBalanceController
# from bubo.rl.gait_rl.ppo_gait_learner import PPOGaitLearner

logger = logging.getLogger("Cerebellum_v6500")


class CerebellumNodeV6500:
    """Integrates CMAC + MPC + PPO into unified cerebellar output."""
    HZ = 10

    # Notice: No 'bus' in the parameters here. The Parent is the boss.
    def __init__(self, config: dict):
        self.name = "Cerebellum"

        # 1. FIRST, we build the bus.
        self.bus = NeuralBus(self.name, config["pub_port"], config["sub_endpoints"])

        # 2. THEN, we pass that newly built bus to the CMAC.
        self.cmac = CMACCerebellumNode(self.bus)

        self.mpc    = MPCBalanceController(self.bus)
#         self.ppo    = PPOGaitLearner(self.bus)
        self._imu   = {"accel":[0,0,9.81],"gyro":[0,0,0],"jerk_mag":0,"dt":0.01}
        self._depth = []
        self._com   = np.zeros(2)
        self._leg_joints = [0.0]*12
        self._foot_p = [0.0, 0.0]
        self._fell  = False
        self._stability = 1.0
        self._running = False
        self._lock  = threading.Lock()

    def _on_vest(self, msg):
        with self._lock:
            self._imu = {
                "accel": msg.payload.get("accel_rps2",[0,0,9.81]),
                "gyro":  msg.payload.get("gyro_rps",[0,0,0]),
                "jerk_mag": float(msg.payload.get("jerk_mag",0)),
                "dt":    0.01,
            }
        self.mpc.update_imu(
            msg.payload.get("accel_rps2",[0,0,9.81]),
            msg.payload.get("gyro_rps",[0,0,0]),
            float(msg.payload.get("jerk_mag",0)),
        )

    def _on_depth(self, msg):
        with self._lock:
            self._depth = msg.payload.get("point_cloud",[])

    def _on_spinal(self, msg):
        joints = msg.payload.get("joint_angles",[])
        fp     = msg.payload.get("foot_pressure",[0,0])
        with self._lock:
            if len(joints) >= 12: self._leg_joints = joints[-12:]
            if len(fp) >= 2:      self._foot_p     = fp[:2]
            self.mpc.update_foot_pressure(fp[0] if fp else 0.0, fp[1] if len(fp)>1 else 0.0)

    def _loop(self):
        iv = 1.0 / self.HZ
        t_save = time.time()
        while self._running:
            t0 = time.time()
            with self._lock:
                imu    = dict(self._imu)
                depth  = list(self._depth)
                legs   = list(self._leg_joints)
                foot_p = list(self._foot_p)

            # Update terrain map for MPC
            self.mpc.update_terrain(depth, self._com)

            # MPC balance step
            mpc_out = self.mpc.step(self._com)
            stability = float(mpc_out.get("stability", 1.0))
            with self._lock: self._stability = stability

            # PPO gait delta
            ppo_state = {
                "com_vel":      [mpc_out.get("com_vx",0), mpc_out.get("com_vy",0), 0],
                "imu":          imu["accel"] + imu["gyro"],
                "foot_contacts":[float(f>1.0) for f in foot_p],
                "cpg_phase":    [0.0, 0.0],
                "leg_joints":   legs + [0.0]*(21-len(legs)),
                "prev_delta":   [],
            }
            cpg_delta = 0.0

            now_ns = time.time_ns()
            self.bus.publish(T.CEREBELL_DELTA, {
                "arm_correction":  [0.0]*14,
                "leg_correction":  [0.0]*12,
                "mpc_balance":     mpc_out,
                "cpg_delta": 0.0,
                "stability":       stability,
                "terrain_mode":    mpc_out.get("terrain_mode","flat"),
                "ppo_scale": 1.0,
                "timestamp_ns":    now_ns,
            })

            # Periodic save
            if time.time() - t_save > 60.0:  # Or whatever the save condition was
                # self.ppo.save(); t_save = time.time()
                pass  # <--- Add this line here to satisfy the indentation gods

            # This MUST be back-indented to the same level as the 'if'
            # If it is indented under 'if', it only sleeps once per minute!
            time.sleep(max(0, iv - (time.time() - t0)))

    def start(self):
        self.bus.start()
        self.mpc.start()
        self.bus.subscribe(T.VESTIBULAR,    self._on_vest)
        self.bus.subscribe(T.VISUAL_DEPTH,  self._on_depth)
        self.bus.subscribe(T.SPINAL_FBK,    self._on_spinal)
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info(f"{self.name} v6500 | CMAC + MPC + PPO | {self.HZ}Hz")

    def stop(self):
        self._running = False
#         self.ppo.save()
        self.bus.stop()


if __name__ == "__main__":
    # Point back to the root level "cerebellum" key for the routing endpoints
    with open("config/cluster_config.json") as f: 
        cfg = json.load(f)["cerebellum"]
    n = CerebellumNodeV6500(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
