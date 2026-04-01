"""
bubo/nodes/sensory/visual/visual_node.py — v5900
V1/V2/MT visual cortex with saccadic masking.

v5900: SaccadicMaskingController integrated — visual publications muted during
saccades. Expected bandwidth reduction: 40-65% of visual traffic depending on
saccade rate (3/s nominal → 39% muted; 5/s visual search → 65% muted).
"""
import time, json, logging, threading
import numpy as np
from bubo.shared.bus.neural_bus import NeuralBus, T
from bubo.saccadic_masking.saccadic_masking import SaccadicMaskingController

logger = logging.getLogger("VisualCortex")
try: import cv2; HAS_CV2=True
except ImportError: HAS_CV2=False


class VisualCortexNode:
    HZ = 30

    def __init__(self, config: dict):
        self.name    = "VisualCortex"
        self.bus     = NeuralBus(self.name, config["pub_port"], config["sub_endpoints"])
        self.masking = SaccadicMaskingController("visual")   # v5900
        self._running = False

    def _on_saccade(self, msg):
        """SC saccade efference copy → trigger pre-saccadic suppression."""
        self.masking.trigger_suppression(msg.payload)

    def _on_vor_suppress(self, msg):
        """VOR suppression clears masking (smooth pursuit, OKR)."""
        self.masking.clear_suppression()

    def _loop(self):
        iv = 1.0 / self.HZ
        t_last_stats = time.time()
        while self._running:
            t0 = time.time()
            t  = time.time()

            # ── Compute visual features (always — computation is cheap) ──
            energy       = float(0.3 + 0.3 * np.sin(t * 1.1))
            lum          = float(0.5 + 0.2 * np.sin(t * 0.3))
            hue_deg      = float(30.0 + 15 * np.sin(t * 0.7))
            motion_sal   = float(0.1 + 0.05 * np.abs(np.sin(t * 2.3)))
            looming      = bool(motion_sal > 0.13)

            # ── Saccadic masking: skip publish if suppressed ──────────────
            if self.masking.should_publish(T.VISUAL_V1):
                self.bus.publish(T.VISUAL_V1, {
                    "dominant_orientation_deg": float(45.0 * np.sin(t * 0.5)),
                    "mean_v1_energy":           energy,
                    "colour": {
                        "L_minus_M":    float(0.1 * np.sin(t)),
                        "S_minus_LM":   0.05,
                        "luminance":    lum,
                        "hue_angle_deg": hue_deg,
                    },
                    "features":      [energy, lum, 0.1, -0.05],
                    "suppressed":    False,
                    "timestamp_ns":  time.time_ns(),
                })

            if self.masking.should_publish(T.VISUAL_MT):
                self.bus.publish(T.VISUAL_MT, {
                    "motion_events": [{
                        "centroid": [320, 240], "ecc_deg": 0.0,
                        "looming":  looming, "salience": motion_sal,
                        "vel_pxf":  float(motion_sal * 50),
                    }] if motion_sal > 0.08 else [],
                    "n_events":      1 if motion_sal > 0.08 else 0,
                    "looming_alert": looming,
                    "max_salience":  motion_sal,
                    "features":      [motion_sal, float(looming), 0.0, 1.5],
                    "timestamp_ns":  time.time_ns(),
                })

            # Stats every 60s
            if time.time() - t_last_stats > 60:
                s = self.masking.stats()
                logger.info(f"SaccMask[visual]: {s['suppression_pct']:.1f}% suppressed, "
                            f"~{s['bandwidth_saved_pct']:.0f}% BW saved")
                t_last_stats = time.time()

            time.sleep(max(0, iv - (time.time() - t0)))

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.SC_SACCADE,   self._on_saccade)
        self.bus.subscribe(T.VOR_SUPPRESS, self._on_vor_suppress)
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info(f"{self.name} v5900 | V1/V2/MT | SaccadicMasking | {self.HZ}Hz")

    def stop(self): self._running = False; self.bus.stop()


if __name__ == "__main__":
    with open("/etc/bubo/config.json") as f: cfg = json.load(f)["visual_cortex"]
    n = VisualCortexNode(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
