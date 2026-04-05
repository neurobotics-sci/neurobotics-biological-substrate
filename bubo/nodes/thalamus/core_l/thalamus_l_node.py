"""
bubo/nodes/thalamus/core_l/thalamus_l_node.py — v50.0
Thalamus-L: Sensory relay — Orin Nano 8GB (192.168.1.13)
VPL (somatosensory), LGN (visual), MGN (auditory), Pulvinar (multisensory/SC).

v50.0 NEW: distributed thalamocortical architecture.
  - Publishes T.THAL_HB heartbeat at 20Hz for failover detection
  - Monitors T.THAL_HB from thalamus-R
  - On thalamus-R failure: activates backup motor relay
  - Social threat modulation applied at thalamic gating level
"""
import time, json, logging, threading
import numpy as np
from collections import deque
from bubo.shared.bus.neural_bus import NeuralBus, T

logger = logging.getLogger("Thalamus-L")

THAL_R_IP   = "192.168.1.18"
THAL_HB_MS  = 50.0   # heartbeat period (20Hz)
THAL_TIMEOUT = 150.0  # 3 missed HBs → failover


class ThalamicRelay:
    """Alpha-oscillation-gated sensory relay nucleus."""
    def __init__(self, modality: str):
        self.modality   = modality
        self._alpha_gate = 0.5    # 0=open, 1=closed
        self._attended  = False

    def gate(self, signal: float, global_alpha: float, attended: bool) -> float:
        if attended:
            self._alpha_gate *= 0.92   # open gate when attended
        else:
            self._alpha_gate = min(1.0, self._alpha_gate * 1.08)  # close when not attended
        self._alpha_gate = float(np.clip(self._alpha_gate, 0.05, 0.95))
        return float(signal * (1.0 - self._alpha_gate))


class ThalamicLNode:
    HZ = 100
    HB_HZ = 20

    def __init__(self, config: dict):
        self.name  = "Thalamus-L"
        self.bus   = NeuralBus(self.name, config["pub_port"], config["sub_endpoints"])

        self._relays = {m: ThalamicRelay(m) for m in ["visual","auditory","soma","vestibular","sc"]}
        self._alpha       = 0.5
        self._attended    = "visual"
        self._social_threat_weight = 1.0  # modified by social node
        self._oxt_suppression = 0.0

        # Buffered sensory data
        self._bufs = {m: {"features":[],"strength":0.0,"t":0.0} for m in self._relays}

        # Thalamus-R failover monitoring
        self._thal_r_last_hb_ms = time.time() * 1000
        self._thal_r_failed     = False
        self._backup_mode       = False  # True when covering thal-R duties

        self._running = False
        self._lock    = threading.Lock()

    # ── Sensory handlers ─────────────────────────────────────────────
    def _on_vis(self, msg):
        with self._lock:
            self._bufs["visual"] = {"features": msg.payload.get("features",[]),
                                     "strength": float(msg.payload.get("mean_v1_energy",0.3)),
                                     "t": time.time()}
    def _on_aud(self, msg):
        with self._lock:
            self._bufs["auditory"] = {"features": msg.payload.get("features",[]),
                                       "strength": float(msg.payload.get("band_energy_mean",0.3)),
                                       "t": time.time()}
    def _on_som(self, msg):
        with self._lock:
            self._bufs["soma"] = {"features": msg.payload.get("features",[]),
                                   "strength": float(msg.payload.get("area_3b",0.2)),
                                   "t": time.time()}
    def _on_vest(self, msg):
        jerk = float(msg.payload.get("jerk_mag",0))
        with self._lock:
            self._bufs["vestibular"] = {"features":[msg.payload.get("roll_deg",0),
                                                     msg.payload.get("pitch_deg",0)],
                                         "strength": min(jerk+0.1,1.0), "t": time.time()}
    def _on_sc(self, msg):
        sal = float(msg.payload.get("max_salience",0.1))
        with self._lock:
            self._bufs["sc"] = {"features": msg.payload.get("motion_events",[]),
                                  "strength": sal, "t": time.time()}
    def _on_attention(self, msg):
        with self._lock: self._attended = msg.payload.get("dominant","visual")
    def _on_circadian(self, msg):
        with self._lock: self._alpha = float(1.0 - msg.payload.get("arousal",0.7))
    def _on_social_threat(self, msg):
        with self._lock:
            self._social_threat_weight = float(msg.payload.get("threat_weight", 1.0))
            self._oxt_suppression      = float(msg.payload.get("oxt_suppression", 0.0))
    def _on_thal_r_hb(self, msg):
        with self._lock: self._thal_r_last_hb_ms = time.time() * 1000

    def _relay_loop(self):
        iv = 1.0 / self.HZ; hb_iv = 1.0 / self.HB_HZ; t_last_hb = time.time()
        while self._running:
            t0 = time.time()
            with self._lock:
                alpha=self._alpha; attended=self._attended; bufs=dict(self._bufs)
                thal_r_gap = time.time()*1000 - self._thal_r_last_hb_ms

            # Failover detection
            if thal_r_gap > THAL_TIMEOUT and not self._thal_r_failed:
                self._thal_r_failed = True; self._backup_mode = True
                logger.warning("Thalamus-R FAILOVER: activating backup motor relay")
                self.bus.publish(T.THAL_FAILOVER, {
                    "failed_node": "thalamus-r",
                    "backup_node": "thalamus-l",
                    "timestamp_ns": time.time_ns()})
            elif thal_r_gap < THAL_HB_MS * 3 and self._thal_r_failed:
                self._thal_r_failed = False; self._backup_mode = False
                logger.info("Thalamus-R recovered — exiting backup mode")

            # Relay each sensory modality
            for mod, relay_topic in [
                ("visual", T.VISUAL_V1), ("auditory", T.AUDITORY_A1),
                ("soma", T.TOUCH_SA1), ("vestibular", T.VESTIBULAR)
            ]:
                buf = bufs[mod]
                age = time.time() - buf["t"]
                if age > 0.5: continue
                gated = self._relays[mod].gate(buf["strength"], alpha, mod == attended)
                # Apply social OXT modulation to amygdala-relevant signals
                if mod in ("visual","auditory") and self._oxt_suppression > 0:
                    gated *= (1.0 - 0.3 * self._oxt_suppression)
                if gated > 0.02:
                    self.bus.publish(T.THAL_SENSORY, {
                        "modality": mod, "features": buf["features"],
                        "gated_strength": gated,
                        "alpha_gate": self._relays[mod]._alpha_gate,
                        "social_mod": float(1.0 - self._social_threat_weight),
                        "timestamp_ns": time.time_ns(),
                    })

            # Heartbeat
            if time.time() - t_last_hb >= hb_iv:
                t_last_hb = time.time()
                self.bus.publish(T.THAL_HB, {
                    "node": "thalamus-l",
                    "backup_mode": self._backup_mode,
                    "timestamp_ns": time.time_ns(),
                })

            time.sleep(max(0, iv - (time.time() - t0)))

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.VISUAL_V1,          self._on_vis)
        self.bus.subscribe(T.AUDITORY_A1,        self._on_aud)
        self.bus.subscribe(T.TOUCH_SA1,          self._on_som)
        self.bus.subscribe(T.VESTIBULAR,         self._on_vest)
        self.bus.subscribe(T.SC_SACCADE,         self._on_sc)
        self.bus.subscribe(T.CTX_ATTENTION,      self._on_attention)
        self.bus.subscribe(T.SYS_CIRCADIAN,      self._on_circadian)
        self.bus.subscribe(T.SOCIAL_THREAT_MOD,  self._on_social_threat)
        self.bus.subscribe(T.THAL_HB,            self._on_thal_r_hb)
        self._running = True
        threading.Thread(target=self._relay_loop, daemon=True).start()
        logger.info(f"{self.name} v50.0 | VPL+LGN+MGN+Pulvinar | failover-ready | {self.HZ}Hz")

    def stop(self): self._running = False; self.bus.stop()


if __name__ == "__main__":
    with open("/etc/bubo/config.json") as f: cfg = json.load(f)["thalamus_l"]
    n = ThalamicLNode(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
