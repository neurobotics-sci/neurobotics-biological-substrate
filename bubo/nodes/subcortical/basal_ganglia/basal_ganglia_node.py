"""
bubo/nodes/subcortical/basal_ganglia/basal_ganglia_node.py — v5550
Basal Ganglia: DA-modulated action selection + ACC conflict monitoring (absorbed from cingulate).

v5550: Conflict monitoring (ACC function) absorbed here from removed cingulate Orin node.
  Cost: ~2ms added to 100Hz BG loop — within budget.
  The conflict signal (top-2 probabilities within 0.15) already lives in BG output;
  we formalise it and broadcast on T.ACC_CONFLICT.
"""
import time, json, logging, threading
import numpy as np
from collections import deque
from bubo.shared.bus.neural_bus import NeuralBus, T
from bubo.shared.watchdog.node_watchdog import NodeWatchdog

logger = logging.getLogger("BasalGanglia")

ACTIONS = [
    "reach_left","reach_right","step_forward","step_back","turn_head",
    "grasp","release","stand_still","look_left","look_right","speak",
    "withdraw","crouch","jump","balance_adj","explore","seek_charge","rest_posture",
]
N = len(ACTIONS)


class ACCConflictMonitor:
    """
    Anterior Cingulate Cortex conflict monitoring — absorbed from removed cingulate Orin.
    Fires when two competing actions have similar probability (difficult decision).
    Also detects sustained prediction error (outcome ≠ expectation).
    """
    CONFLICT_THRESH = 0.15   # difference between top-2 probs < this → conflict

    def __init__(self):
        self._err_hist = deque(maxlen=50)
        self._conflict_hist = deque(maxlen=50)

    def evaluate(self, probabilities: list, rpe: float) -> dict:
        if not probabilities:
            return {"conflict": 0.0, "error": 0.0, "sustained_conflict": 0.0}
        p = np.array(probabilities)
        sorted_p = np.sort(p)[::-1]
        conflict = float(1.0 - (sorted_p[0] - sorted_p[1])) if len(sorted_p) > 1 else 0.0
        error    = float(np.clip(abs(rpe), 0, 1))
        self._conflict_hist.append(conflict)
        self._err_hist.append(error)
        return {
            "conflict":           conflict,
            "error":              error,
            "sustained_conflict": float(np.mean(self._conflict_hist)),
            "sustained_error":    float(np.mean(self._err_hist)),
            "high_conflict":      conflict > (1.0 - self.CONFLICT_THRESH),
        }


class BasalGangliaNode:
    HZ = 100

    def __init__(self, config: dict):
        self.name = "BasalGanglia"
        self.bus  = NeuralBus(self.name, config["pub_port"], config["sub_endpoints"])
        self.wd   = NodeWatchdog(self.name, self.bus)
        self.acc  = ACCConflictMonitor()   # absorbed from cingulate

        self._d1  = np.ones(N)*0.5
        self._d2  = np.ones(N)*0.5
        self._da  = 0.6; self._ne = 0.2; self._sero = 0.5
        self._bg_temp    = 0.30
        self._hunger     = 0.0; self._thermal = 0.0; self._fatigue = 0.0
        self._salience   = np.ones(N)*0.1
        self._last_action = None
        self._rpe        = 0.0
        self._last_probs : list = []
        self._running    = False
        self._lock       = threading.Lock()
        self._lr         = 0.02

    def _on_da(self, msg):
        with self._lock:
            self._da       = float(msg.payload.get("DA", 0.6))
            self._bg_temp  = float(msg.payload.get("bg_temperature", 0.30))
            self._hunger   = float(msg.payload.get("hunger", 0))
            self._thermal  = float(msg.payload.get("thermal", 0))

    def _on_neuromod(self, msg):
        with self._lock:
            if "NE"  in msg.payload: self._ne   = float(msg.payload["NE"])
            if "5HT" in msg.payload: self._sero = float(msg.payload["5HT"])

        a = msg.payload.get("action","")
        if a in ACTIONS:
            sal = np.ones(N)*0.1
            sal[ACTIONS.index(a)] += 0.6
            with self._lock: self._salience = sal

    def _on_insula(self, msg):
        with self._lock: self._fatigue = float(msg.payload.get("global_fatigue", 0))

    def _on_reward(self, msg):
        rpe = float(msg.payload.get("rpe", 0))
        ai  = int(msg.payload.get("action_idx", 0))
        with self._lock: self._rpe = rpe
        m = abs(self._da - 0.5)*2 * self._lr * rpe
        self._d1[ai] = float(np.clip(self._d1[ai]+m, 0.05, 0.95))
        self._d2[ai] = float(np.clip(self._d2[ai]-m*0.8, 0.05, 0.95))

    def _select(self):
        with self._lock:
            da=self._da; ne=self._ne; sero=self._sero
            temp=max(self._bg_temp, 0.10)
            sal=self._salience.copy()
            hunger=self._hunger; thermal=self._thermal; fatigue=self._fatigue
            rpe=self._rpe

        sal[ACTIONS.index("seek_charge")]  += hunger*0.8
        sal[ACTIONS.index("rest_posture")] += fatigue*0.6
        for a in ["step_forward","step_back","reach_left","reach_right"]:
            sal[ACTIONS.index(a)] *= (1.0 - 0.6*thermal)

        direct = sal*self._d1*(0.4+1.2*da)
        ind    = sal*self._d2*(1.4-1.0*da)
        gpe    = np.clip(1.0-ind*0.8, 0.1, 1.0)
        stn    = np.clip(0.2/(gpe+0.1), 0.1, 1.0)*(0.5+ne*0.8)
        gpi    = np.clip(ind*0.6+stn-direct, 0, 2)
        thal   = np.clip(1.0-gpi, 0, 1)
        exp_t  = np.exp(thal/temp)
        prob   = exp_t/exp_t.sum()
        wi     = int(np.argmax(prob))
        self._last_probs = prob.tolist()

        # ACC conflict monitoring (absorbed from cingulate)
        acc = self.acc.evaluate(prob.tolist(), rpe)

        return {
            "action": ACTIONS[wi], "action_idx": wi,
            "certainty": float(prob[wi]),
            "probabilities": prob.tolist(),
            "bg_temp": temp, "da_mode": "da_driven",
            "acc_conflict": acc,
        }

    def _loop(self):
        iv = 1.0 / self.HZ
        while self._running:
            t0 = time.time()
            self.wd.heartbeat()
            result = self._select()
            now_ns = time.time_ns()

            if result["action"] != self._last_action:
                self.bus.publish(T.CTX_PFC_CMD, {
                    **result, "source":"basal_ganglia", "motor":{},
                    "timestamp_ns": now_ns,
                })
                self._last_action = result["action"]

            # Broadcast ACC conflict signal (absorbed from cingulate)
            acc = result["acc_conflict"]
            if acc["high_conflict"] or acc["sustained_conflict"] > 0.5:
                self.bus.publish(T.ACC_CONFLICT, {
                    **acc, "action": result["action"],
                    "source": "bg_acc_absorbed", "timestamp_ns": now_ns,
                })

            self.wd.record_publish()
            time.sleep(max(0, iv-(time.time()-t0)))

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.DA_VTA,      self._on_da)
        self.bus.subscribe(T.NE_LC,       self._on_neuromod)
        self.bus.subscribe(T.SERO_RAPHE,  self._on_neuromod)
        self.bus.subscribe(T.INSULA_STATE, self._on_insula)
        self.bus.subscribe(T.SYS_REWARD,  self._on_reward)
        self.wd.start()
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info(f"{self.name} v5550 | DA-temp | ACC-conflict(absorbed) | {self.HZ}Hz")

    def stop(self): self._running = False; self.bus.stop()


if __name__ == "__main__":
    with open("/etc/bubo/config.json") as f: cfg=json.load(f)["basal_ganglia"]
    n=BasalGangliaNode(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
