"""
bubo/nodes/cortex/cingulate/cingulate_node.py — v11.14
Anterior/Posterior Cingulate Cortex — Orin Nano 8GB (192.168.1.17)

ACC (BA24/32): error detection, conflict monitoring, pain affect, autonomic coupling.
PCC (BA23/31) + Precuneus: Default Mode Network hub, episodic future thinking.
"""
import time, json, logging, threading
import numpy as np
from collections import deque
from bubo.shared.bus.neural_bus import NeuralBus, T

logger = logging.getLogger("CingulateCortex")


class ACCErrorDetector:
    """
    Anterior Cingulate Cortex — conflict and error monitoring.
    Fires when:
      (a) BG action outcome ≠ PFC prediction (prediction error)
      (b) Two BG actions have similar probability (conflict)
      (c) Insula pain affect signal exceeds threshold (pain suffering)
    """
    CONFLICT_THRESH = 0.15   # max prob - second prob < this → conflict
    ERROR_THRESH    = 0.25

    def __init__(self):
        self._err_hist    = deque(maxlen=50)
        self._conflict_hist = deque(maxlen=50)
        self._pain_hist   = deque(maxlen=50)

    def evaluate_action(self, probabilities: list, outcome_rpe: float) -> dict:
        if not probabilities:
            return {"error": 0.0, "conflict": 0.0, "pain_affect": 0.0}
        probs = np.array(probabilities)
        sorted_p = np.sort(probs)[::-1]
        conflict = float(1.0 - (sorted_p[0] - sorted_p[1])) if len(sorted_p) > 1 else 0.0
        error    = float(np.clip(abs(outcome_rpe), 0, 1))
        self._conflict_hist.append(conflict); self._err_hist.append(error)
        return {"error": error, "conflict": conflict,
                "sustained_error":   float(np.mean(self._err_hist)),
                "sustained_conflict": float(np.mean(self._conflict_hist))}

    def pain_affect(self, noci_intensity: float, insula_body_feel: float) -> float:
        """Affective (emotional) pain = noci + negative body feeling."""
        affect = float(np.clip(noci_intensity * 0.6 + max(-insula_body_feel, 0) * 0.4, 0, 1))
        self._pain_hist.append(affect)
        return affect


class DefaultModeNetwork:
    """
    PCC/Precuneus: default mode network and episodic future thinking.
    Active when: arousal < 0.4 AND fear < 0.2 AND no active motor task.
    Generates prospective memory: pre-plays upcoming tasks through hippocampal context.
    """
    def __init__(self):
        self._dmn_active = False
        self._prospective = deque(maxlen=10)   # upcoming action plans
        self._t_idle = time.time()

    def step(self, arousal: float, fear: float, motor_active: bool,
             pfc_action: str, hippo_context: list) -> dict:
        idle = arousal < 0.4 and fear < 0.2 and not motor_active
        if idle and not self._dmn_active:
            self._dmn_active = True; self._t_idle = time.time()
        elif not idle:
            self._dmn_active = False

        # Episodic future thinking: pre-play next likely action
        epi_future = {}
        if self._dmn_active and hippo_context:
            epi_future = {
                "next_action": pfc_action,
                "context":     hippo_context[:4],
                "idle_s":      time.time() - self._t_idle,
            }
            self._prospective.append(epi_future)

        return {
            "dmn_active":     self._dmn_active,
            "idle_s":         time.time() - self._t_idle if self._dmn_active else 0.0,
            "episodic_future": epi_future,
        }


class SalienceNetwork:
    """
    Anterior insula + ACC salience detection.
    Monitors for unexpected events that warrant network switching:
    DMN → Task-Positive Network (attention, PFC-driven action).
    """
    def __init__(self):
        self._baseline = deque(maxlen=100)
        self._alert    = False

    def evaluate(self, insula_state: dict, parietal_threat: float,
                 noci: float, audio_rms: float) -> dict:
        signal = max(parietal_threat, noci, float(np.clip(audio_rms * 2, 0, 1)))
        self._baseline.append(signal)
        baseline_mu  = float(np.mean(self._baseline)) if self._baseline else 0.3
        baseline_sig = float(np.std(self._baseline))  if len(self._baseline) > 5 else 0.1
        z_score      = (signal - baseline_mu) / max(baseline_sig, 0.01)
        salience_alert = z_score > 2.0    # > 2 SD above baseline
        if salience_alert and not self._alert:
            self._alert = True
        elif z_score < 1.0:
            self._alert = False
        return {"salience_alert": salience_alert, "z_score": float(z_score),
                "signal": signal, "baseline_mu": baseline_mu}


class CingulateNode:
    HZ = 20

    def __init__(self, config: dict):
        self.name   = "CingulateCortex"
        self.bus    = NeuralBus(self.name, config["pub_port"], config["sub_endpoints"])
        self.acc    = ACCErrorDetector()
        self.dmn    = DefaultModeNetwork()
        self.sal    = SalienceNetwork()

        self._pfc_probs      = []
        self._pfc_action     = "stand_still"
        self._pfc_motor_on   = False
        self._rpe            = 0.0
        self._noci           = 0.0
        self._arousal        = 0.5
        self._fear           = 0.0
        self._insula_state   = {}
        self._hippo_context  = []
        self._parietal_threat = 0.0
        self._audio_rms      = 0.0
        self._running        = False
        self._lock           = threading.Lock()

    def _on_pfc(self, msg):
        with self._lock:
            self._pfc_probs   = msg.payload.get("probabilities", [])
            self._pfc_action  = msg.payload.get("action", "stand_still")
            self._pfc_motor_on = self._pfc_action not in ("stand_still", "rest_posture")

    def _on_reward(self, msg):
        with self._lock: self._rpe = float(msg.payload.get("rpe", 0))

    def _on_noci(self, msg):
        with self._lock: self._noci = float(msg.payload.get("intensity", 0))

    def _on_insula(self, msg):
        with self._lock: self._insula_state = msg.payload

    def _on_hippo(self, msg):
        p = msg.payload.get("context_vector") or msg.payload.get("slam_pose", [])
        with self._lock: self._hippo_context = p[:8] if p else []

    def _on_circadian(self, msg):
        with self._lock: self._arousal = float(msg.payload.get("arousal", 0.5))

    def _on_cea(self, msg):
        with self._lock: self._fear = float(msg.payload.get("cea_activation", 0))

    def _on_parietal(self, msg):
        with self._lock: self._parietal_threat = float(msg.payload.get("threat_level", 0))

    def _on_audio(self, msg):
        with self._lock: self._audio_rms = float(msg.payload.get("rms_db", -60) / -60)

    def _loop(self):
        iv = 1.0 / self.HZ
        while self._running:
            t0 = time.time()
            with self._lock:
                probs   = list(self._pfc_probs)
                action  = self._pfc_action
                motor   = self._pfc_motor_on
                rpe     = self._rpe
                noci    = self._noci
                arousal = self._arousal
                fear    = self._fear
                ins     = dict(self._insula_state)
                hctx    = list(self._hippo_context)
                par_thr = self._parietal_threat
                aud_rms = self._audio_rms

            # ACC evaluations
            acc_out  = self.acc.evaluate_action(probs, rpe)
            pain_aff = self.acc.pain_affect(noci, ins.get("body_feeling", 0))

            # DMN
            dmn_out  = self.dmn.step(arousal, fear, motor, action, hctx)

            # Salience network
            sal_out  = self.sal.evaluate(ins, par_thr, noci, aud_rms)

            # ACC → autonomic coupling: high error/conflict → HPA stressor boost
            acc_stress = float(np.clip(
                0.5 * acc_out["sustained_error"] + 0.5 * acc_out["sustained_conflict"], 0, 1))

            self.bus.publish(T.ACC_ERROR, {
                **acc_out, "acc_stress": acc_stress, "timestamp_ns": time.time_ns()})
            self.bus.publish(T.ACC_CONFLICT, {
                "conflict": acc_out["conflict"],
                "action":   action,
                "salience": sal_out,
                "timestamp_ns": time.time_ns(),
            })
            self.bus.publish(T.ACC_PAIN_AFF, {
                "pain_affect": pain_aff, "noci": noci,
                "suffering":   pain_aff > 0.5,
                "timestamp_ns": time.time_ns(),
            })
            self.bus.publish(T.PCC_DMN, {**dmn_out, "timestamp_ns": time.time_ns()})
            if dmn_out["episodic_future"]:
                self.bus.publish(T.PCC_EPISODIC, {
                    **dmn_out["episodic_future"], "timestamp_ns": time.time_ns()})

            # Salience alert → switch DMN off, boost PFC attention
            if sal_out["salience_alert"]:
                self.bus.publish(T.RF_AROUSAL, {
                    "arousal": min(1.0, arousal + 0.3),
                    "source":  "salience_network",
                    "z_score": sal_out["z_score"],
                })

            time.sleep(max(0, iv - (time.time() - t0)))

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.CTX_PFC_CMD,   self._on_pfc)
        self.bus.subscribe(T.SYS_REWARD,    self._on_reward)
        self.bus.subscribe(T.NOCI_HEAT,     self._on_noci)
        self.bus.subscribe(T.NOCI_MECH,     self._on_noci)
        self.bus.subscribe(T.INSULA_STATE,  self._on_insula)
        self.bus.subscribe(T.HIPPO_CONTEXT, self._on_hippo)
        self.bus.subscribe(T.SYS_CIRCADIAN, self._on_circadian)
        self.bus.subscribe(T.AMYG_CEA_OUT,  self._on_cea)
        self.bus.subscribe(T.PARIETAL_PERISP, self._on_parietal)
        self.bus.subscribe(T.AUDITORY_A1,   self._on_audio)
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info(f"{self.name} v11.14 | ACC error/conflict | PCC DMN | salience | {self.HZ}Hz")

    def stop(self): self._running = False; self.bus.stop()


if __name__ == "__main__":
    with open("/etc/bubo/config.json") as f: cfg = json.load(f)["cingulate"]
    n = CingulateNode(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
