"""
bubo/nodes/cortex/insula/insula_node.py — v11.14
Insula: interoception, mechanical fatigue (Arrhenius), nod-off (safety-first).
Hosts NodOffController. PTP-timestamped outputs.
"""
import time, json, logging, threading
import numpy as np
from bubo.shared.bus.neural_bus import NeuralBus, T
from bubo.safety.nod_off import NodOffController

logger = logging.getLogger("Insula")

EA_J = 60000.0; R_GAS = 8.314; K0_WEAR = 1e8

SERVO_NAMES = [
    "shoulder_l_flex","shoulder_l_abd","shoulder_l_rot","elbow_l","wrist_l_flex","wrist_l_rot","forearm_l",
    "shoulder_r_flex","shoulder_r_abd","shoulder_r_rot","elbow_r","wrist_r_flex","wrist_r_rot","forearm_r",
    "hip_l_flex","hip_l_abd","hip_l_rot","knee_l","ankle_l","subtalar_l",
    "hip_r_flex","hip_r_abd","hip_r_rot","knee_r","ankle_r","subtalar_r",
]

REST_THRESH   = 0.70
REPAIR_THRESH = 0.90


def arrhenius_factor(temp_C: float) -> float:
    T_K = temp_C + 273.15; T0 = 298.15
    return float(np.clip(np.exp(-EA_J / R_GAS * (1/T_K - 1/T0)), 0.5, 200.0))


class FatigueModel:
    ALPHA=0.30; BETA=0.40; GAMMA=0.30

    def __init__(self):
        self._fat  = {n: 0.0 for n in SERVO_NAMES}
        self._duty = {n: 0.0 for n in SERVO_NAMES}
        self._osc  = {n: 0   for n in SERVO_NAMES}
        self._prev = {n: 0.0 for n in SERVO_NAMES}
        self._prev_dq = {n: 0.0 for n in SERVO_NAMES}

    def update(self, angles, temps, torques, dt):
        gf = af = lf = 0.0
        for i, name in enumerate(SERVO_NAMES):
            if i >= len(angles): break
            q = float(angles[i])
            temp = float(temps[i]) if i < len(temps) else 25.0
            torque = float(torques[i]) if i < len(torques) else 0.0
            self._duty[name] = 0.99*self._duty[name] + 0.01*float(abs(torque) > 0.5)
            dq = q - self._prev[name]
            if abs(dq) > 0.005 and np.sign(dq) != np.sign(self._prev_dq[name]):
                self._osc[name] += 1
            self._prev[name] = q; self._prev_dq[name] = dq
            arrh = arrhenius_factor(temp)
            inc = (self.ALPHA*self._duty[name]
                   + self.BETA*min(arrh/100.0, 1.0)
                   + self.GAMMA*min(self._osc[name]/10000.0, 1.0)) * dt * 0.0001
            self._fat[name] = float(np.clip(self._fat[name] + inc, 0, 1))
            gf = max(gf, self._fat[name])
            if any(k in name for k in ["hip","knee","ankle","subtalar"]): lf = max(lf, self._fat[name])
            else: af = max(af, self._fat[name])
        return {"global":gf,"arm":af,"leg":lf,"per":dict(self._fat),
                "worst":max(self._fat,key=self._fat.get),
                "needs_rest":gf>REST_THRESH,"needs_repair":gf>REPAIR_THRESH}

    def recovery_step(self, dt):
        for n in self._fat: self._fat[n] *= np.exp(-dt/300.0)


class InsulaNode:
    HZ = 10

    def __init__(self, config: dict):
        self.name  = "Insula"
        self.bus   = NeuralBus(self.name, config["pub_port"], config["sub_endpoints"])
        self.fat   = FatigueModel()
        self.nodoff = NodOffController(self.bus)

        self._angles  = [0.0]*len(SERVO_NAMES)
        self._temps   = [25.0]*len(SERVO_NAMES)
        self._torques = [0.0]*len(SERVO_NAMES)
        self._cpu_C   = 45.0; self._curr_A = 1.5; self._batt = 1.0
        self._adenosine = 0.0; self._fear = 0.0; self._da = 0.6
        self._zmp_ok = True; self._looming = False; self._gripping = False
        self._vest_jerk = 0.0; self._pain = 0.0
        self._resting = False; self._rest_t = 0.0
        self._running = False; self._lock = threading.Lock(); self._t_last = time.time()

    def _on_spinal_fbk(self, msg):
        a = msg.payload.get("joint_angles", [])
        with self._lock:
            for i, v in enumerate(a[:len(self._angles)]): self._angles[i] = float(v)

    def _on_hypo(self, msg):
        p = msg.payload
        with self._lock:
            self._batt  = float(p.get("battery_frac", self._batt))
            self._cpu_C = float(p.get("cpu_temp_C", self._cpu_C))
            self._curr_A = float(p.get("current_A", self._curr_A))

    def _on_circadian(self, msg):
        with self._lock:
            self._adenosine = float(msg.payload.get("adenosine", 0))

    def _on_da(self, msg):
        with self._lock: self._da = float(msg.payload.get("DA", 0.6))

    def _on_cea(self, msg):
        with self._lock: self._fear = float(msg.payload.get("cea_activation", 0))

    def _on_mt(self, msg):
        loom = any(e.get("looming") for e in msg.payload.get("motion_events", []))
        with self._lock: self._looming = loom

    def _on_vest(self, msg):
        jerk = float(msg.payload.get("jerk_mag", 0))
        with self._lock: self._vest_jerk = jerk
        if jerk > 0.2: self.nodoff.external_interrupt()

    def _on_noci(self, msg):
        p = float(msg.payload.get("intensity", 0))
        with self._lock: self._pain = p
        self.nodoff.update_safety(self._zmp_ok, self._looming, self._gripping,
                                   self._fear, self._vest_jerk, self._batt, p)

    def _on_rest(self, msg):
        with self._lock: self._resting = True

    def _on_freeze(self, msg):
        self.nodoff.external_interrupt()

    def _loop(self):
        iv = 1.0 / self.HZ
        while self._running:
            t0 = time.time(); dt = max(t0 - self._t_last, 0.001); self._t_last = t0
            with self._lock:
                ang = list(self._angles); tmp = list(self._temps)
                tor = list(self._torques); batt = self._batt
                cpu = self._cpu_C; aden = self._adenosine; da = self._da
                fear = self._fear; jerk = self._vest_jerk; pain = self._pain
                zmp  = self._zmp_ok; loom = self._looming; grip = self._gripping

            if self._resting:
                self.fat.recovery_step(dt)
                fat_state = {"global": 0.0, "arm": 0.0, "leg": 0.0, "per": {},
                             "worst": "", "needs_rest": False, "needs_repair": False}
            else:
                fat_state = self.fat.update(ang, tmp, tor, dt)

            # Rest/Repair trigger
            if fat_state["needs_repair"] and not self._resting:
                self._resting = True; self._rest_t = time.time()
                self.bus.publish(T.REST_REPAIR, {
                    "trigger": "mechanical_fatigue", "global_fatigue": fat_state["global"],
                    "worst_servo": fat_state["worst"], "duration_s": 60,
                    "timestamp_ns": time.time_ns(),
                })
                logger.warning(f"REPAIR: global_fatigue={fat_state['global']:.2f} worst={fat_state['worst']}")
            elif fat_state["needs_rest"] and not self._resting:
                self._resting = True; self._rest_t = time.time()
                self.bus.publish(T.REST_REPAIR, {
                    "trigger": "fatigue_rest", "global_fatigue": fat_state["global"], "duration_s": 30,
                    "timestamp_ns": time.time_ns(),
                })
            if self._resting and (time.time() - self._rest_t) > 30.0:
                if fat_state.get("global", 1.0) < 0.30:
                    self._resting = False

            # Nod-off evaluation (safety-first)
            self.nodoff.update_safety(zmp, loom, grip, fear, jerk, batt, pain)
            nod = self.nodoff.step(aden, da)

            # Interoception integration
            thermal_distress = float(np.clip(abs(cpu - 40.0)/30.0, 0, 1))
            cardiac_load     = float(np.clip(self._curr_A / 5.0, 0, 1))
            body_feeling     = float(np.clip(
                0.4*batt - 0.3*fat_state["global"] - 0.2*thermal_distress - 0.1*nod.get("adenosine",0), -1, 1))

            self.bus.publish(T.INSULA_STATE, {
                "body_feeling": body_feeling, "thermal_distress": thermal_distress,
                "cardiac_load": cardiac_load, "global_fatigue": fat_state["global"],
                "arm_fatigue": fat_state["arm"], "leg_fatigue": fat_state["leg"],
                "nod_off": nod, "is_resting": self._resting,
                "timestamp_ns": time.time_ns(),
            })
            if fat_state["needs_rest"] or fat_state["needs_repair"]:
                self.bus.publish(T.INSULA_FATIGUE, {
                    **fat_state, "cpu_temp_C": cpu, "is_resting": self._resting,
                    "timestamp_ns": time.time_ns(),
                })

            time.sleep(max(0, iv - (time.time() - t0)))

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.SPINAL_FBK,    self._on_spinal_fbk)
        self.bus.subscribe(T.HYPO_STATE,    self._on_hypo)
        self.bus.subscribe(T.SYS_CIRCADIAN, self._on_circadian)
        self.bus.subscribe(T.DA_VTA,        self._on_da)
        self.bus.subscribe(T.AMYG_CEA_OUT,  self._on_cea)
        self.bus.subscribe(T.VISUAL_MT,     self._on_mt)
        self.bus.subscribe(T.VESTIBULAR,    self._on_vest)
        self.bus.subscribe(T.NOCI_HEAT,     self._on_noci)
        self.bus.subscribe(T.NOCI_MECH,     self._on_noci)
        self.bus.subscribe(T.REST_REPAIR,   self._on_rest)
        self.bus.subscribe(T.SAFETY_FREEZE, self._on_freeze)
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info(f"{self.name} v11.14 | Arrhenius fatigue | nod-off safety-first | {self.HZ}Hz")

    def stop(self): self._running = False; self.bus.stop()


if __name__ == "__main__":
    with open("/etc/bubo/config.json") as f: cfg = json.load(f)["insula"]
    n = InsulaNode(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
