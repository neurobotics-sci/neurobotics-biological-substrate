"""
bubo/nodes/thalamus/core_r/thalamus_r_node.py — v50.0
Thalamus-R: Motor/PFC relay — NEW Orin Nano 8GB (192.168.1.18)

VA (ventral anterior, BG→M1), MD (mediodorsal, PFC↔limbic),
Reuniens (hippocampus↔PFC communication).

Monitors thalamus-L heartbeat. On L failure → takes over sensory relay.
"""
import time, json, logging, threading
import numpy as np
from bubo.shared.bus.neural_bus import NeuralBus, T

logger = logging.getLogger("Thalamus-R")

THAL_L_TIMEOUT = 150.0  # ms


class ThalamicRNode:
    HZ = 100
    HB_HZ = 20

    def __init__(self, config: dict):
        self.name = "Thalamus-R"
        self.bus  = NeuralBus(self.name, config["pub_port"], config["sub_endpoints"])

        # BG → M1 relay (VA nucleus)
        self._bg_action     = "stand_still"
        self._bg_certainty  = 0.5
        self._bg_probs      = []

        # PFC ↔ Limbic relay (MD nucleus)
        self._hippo_context = []
        self._amyg_cea      = 0.0

        # Hippocampus → PFC (Reuniens nucleus)
        self._slam_pose     = [0, 0, 0]

        # Social modulation
        self._social_da_boost   = 0.0
        self._social_bond_level = 0.0

        # Thalamus-L failover
        self._thal_l_last_hb_ms = time.time() * 1000
        self._thal_l_failed     = False
        self._backup_sensory    = False

        # Sensory backup buffers (activated on thal-L failure)
        self._sens_bufs = {m: {"features":[],"strength":0.0,"t":0.0}
                           for m in ["visual","auditory","soma"]}
        self._running = False
        self._lock = threading.Lock()

    def _on_bg(self, msg):
        with self._lock:
            self._bg_action    = msg.payload.get("action", "stand_still")
            self._bg_certainty = float(msg.payload.get("certainty", 0.5))
            self._bg_probs     = msg.payload.get("probabilities", [])


    def _on_hippo(self, msg):
        pose = msg.payload.get("slam_pose", [])
        ctx  = msg.payload.get("context_vector", [])
        with self._lock:
            if pose: self._slam_pose = pose[:3]
            if ctx:  self._hippo_context = ctx[:8]

    def _on_cea(self, msg):
        with self._lock: self._amyg_cea = float(msg.payload.get("cea_activation",0))

    def _on_social_bond(self, msg):
        with self._lock:
            self._social_da_boost   = float(msg.payload.get("da_boost", 0))
            self._social_bond_level = float(msg.payload.get("bond_level", 0))

    def _on_thal_l_hb(self, msg):
        if msg.payload.get("node") == "thalamus-l":
            with self._lock: self._thal_l_last_hb_ms = time.time() * 1000

    def _on_vis_backup(self, msg):
        if self._backup_sensory:
            with self._lock:
                self._sens_bufs["visual"] = {"features": msg.payload.get("features",[]),
                                               "strength": float(msg.payload.get("mean_v1_energy",0.3)),
                                               "t": time.time()}

    def _relay_loop(self):
        iv = 1.0 / self.HZ; hb_iv = 1.0 / self.HB_HZ; t_last_hb = time.time()
        while self._running:
            t0 = time.time()
            with self._lock:
                bg_a = self._bg_action; bg_c = self._bg_certainty; bg_p = list(self._bg_probs)
                sp   = list(self._slam_pose); hctx = list(self._hippo_context)
                da_b = self._social_da_boost; bond = self._social_bond_level
                thal_l_gap = time.time()*1000 - self._thal_l_last_hb_ms

            # Thalamus-L failover monitoring
            if thal_l_gap > THAL_L_TIMEOUT and not self._thal_l_failed:
                self._thal_l_failed = True; self._backup_sensory = True
                logger.warning("Thalamus-L FAILOVER: taking over sensory relay")
                self.bus.publish(T.THAL_FAILOVER, {
                    "failed_node": "thalamus-l", "backup_node": "thalamus-r",
                    "timestamp_ns": time.time_ns()})
            elif thal_l_gap < 150 and self._thal_l_failed:
                self._thal_l_failed = False; self._backup_sensory = False
                logger.info("Thalamus-L recovered")

            # VA nucleus: BG → cortex motor relay
            # Apply social DA modulation here (VTA-thalamus interaction)
            effective_certainty = float(np.clip(bg_c + da_b * 0.1, 0, 1))
            self.bus.publish(T.THAL_MOTOR, {
                "action":          bg_a,
                "certainty":       effective_certainty,
                "social_da_boost": da_b,
                "social_bond":     bond,
                "cea_modulation":  float(1.0 - cea * 0.3),  # fear reduces motor drive
                "timestamp_ns":    time.time_ns(),
            })

            # MD nucleus: PFC ↔ limbic bidirectional relay
                self.bus.publish(T.THAL_SENSORY, {
                    "amyg_cea":       cea,
                    "social_context": {"bond": bond, "da_boost": da_b},
                    "timestamp_ns":   time.time_ns(),
                })

            # Reuniens nucleus: hippocampus context → PFC
            if sp or hctx:
                self.bus.publish(T.CTX_PREDICTION, {
                    "slam_pose":     sp,
                    "context":       hctx,
                    "source":        "reuniens_thal_r",
                    "timestamp_ns":  time.time_ns(),
                })

            # Heartbeat
            if time.time() - t_last_hb >= hb_iv:
                t_last_hb = time.time()
                self.bus.publish(T.THAL_HB, {
                    "node": "thalamus-r",
                    "backup_sensory": self._backup_sensory,
                    "timestamp_ns": time.time_ns(),
                })

            time.sleep(max(0, iv - (time.time() - t0)))

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.HIPPO_CONTEXT,   self._on_hippo)
        self.bus.subscribe(T.HIPPO_PLACE,     self._on_hippo)
        self.bus.subscribe(T.AMYG_CEA_OUT,   self._on_cea)
        self.bus.subscribe(T.SOCIAL_BOND,     self._on_social_bond)
        self.bus.subscribe(T.THAL_HB,         self._on_thal_l_hb)
        self.bus.subscribe(T.VISUAL_V1,       self._on_vis_backup)
        self._running = True
        threading.Thread(target=self._relay_loop, daemon=True).start()
        logger.info(f"{self.name} v50.0 | VA+MD+Reuniens | failover-ready | {self.HZ}Hz")

    def stop(self): self._running = False; self.bus.stop()


if __name__ == "__main__":
    with open("/etc/bubo/config.json") as f: cfg = json.load(f)["thalamus_r"]
    n = ThalamicRNode(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
