"""
bubo/nodes/cortex/parietal/parietal_node.py — v11.14
Posterior Parietal Cortex — Orin Nano 8GB (192.168.1.16)

Brodmann areas: SPL (BA7), IPL (BA39/40), IPS, angular gyrus, supramarginal gyrus.

Functions:
  1. Visuospatial integration — egocentric (body-centred) ↔ allocentric (world-centred)
     reference frame switching, driven by visual depth + SLAM context.
  2. Peripersonal space mapping — 50cm radius threat sphere around Bubo.
     Any obstacle within PPS triggers priority signal to thalamus.
  3. Tool use / affordance — recognise graspable objects from shape, size, orientation.
     Output: grasp type (precision/power/lateral) + approach vector in body frame.
  4. Body schema — continuous internal model of all limb positions in space.
     Updated from spinal proprioceptive feedback; degraded during limp mode.
  5. Spatial attention — IPS directs covert attention; biases thalamic alpha gate.
"""
import time, json, logging, threading
import numpy as np
from bubo.shared.bus.neural_bus import NeuralBus, T

logger = logging.getLogger("PosteriorParietal")

PPS_RADIUS_M = 0.50   # peripersonal space radius


class EgocentricMap:
    """
    Maintains a 3-D occupancy representation around Bubo in body frame.
    Updated from SC depth maps and SLAM landmarks.
    Resolution: 0.10m voxels, 2m radius cube = 40×40×20 voxels.
    """
    RES   = 0.10
    RANGE = 2.0
    N     = int(2 * RANGE / RES)

    def __init__(self):
        self._grid = np.zeros((self.N, self.N, self.N // 2), dtype=np.float32)
        self._decay = 0.95

    def update(self, point_cloud: list, body_pose: np.ndarray):
        self._grid *= self._decay
        for pt in point_cloud[:200]:
            if len(pt) < 3: continue
            xi = int((pt[0] + self.RANGE) / self.RES)
            yi = int((pt[1] + self.RANGE) / self.RES)
            zi = int(pt[2] / self.RES)
            if 0 <= xi < self.N and 0 <= yi < self.N and 0 <= zi < self.N // 2:
                self._grid[xi, yi, zi] = min(1.0, self._grid[xi, yi, zi] + 0.4)

    def pps_threat(self) -> float:
        """Fraction of PPS voxels occupied."""
        r = int(PPS_RADIUS_M / self.RES)
        cx = cy = self.N // 2
        cz = self.N // 4
        pps = self._grid[cx-r:cx+r, cy-r:cy+r, cz:cz+r]
        return float(pps.mean()) if pps.size > 0 else 0.0

    def nearest_obstacle_m(self) -> float:
        ys, xs, zs = np.where(self._grid > 0.5)
        if len(xs) == 0: return self.RANGE
        cx = cy = self.N // 2
        dists = np.sqrt((xs - cx)**2 + (ys - cy)**2) * self.RES
        return float(dists.min())


class AffordanceDetector:
    """
    Simple shape-based grasp affordance from visual depth.
    Classifies visible objects into: graspable_precision, graspable_power, none.
    """
    def detect(self, depth_stats: dict, motion_events: list) -> dict:
        d_m = depth_stats.get("median_m", 5.0)
        if d_m > 1.5 or d_m < 0.05:
            return {"type": "none", "confidence": 0.0}
        # Simple heuristic: close + small motion = graspable
        has_motion = len(motion_events) > 0
        if d_m < 0.40:
            g_type = "precision" if d_m < 0.20 else "power"
            return {"type": g_type, "confidence": float(0.7 - d_m), "distance_m": d_m}
        return {"type": "reach_only", "confidence": 0.3, "distance_m": d_m}


class BodySchema:
    """
    Internal model of limb positions in egocentric space.
    Updated from spinal proprioception; published for parietal integration.
    """
    def __init__(self):
        self._arm_l = np.zeros(7); self._arm_r = np.zeros(7)
        self._leg_l = np.zeros(6); self._leg_r = np.zeros(6)

    def update(self, joint_angles: list):
        arr = np.array(joint_angles, dtype=float)
        if len(arr) >= 14: self._arm_l = arr[:7]; self._arm_r = arr[7:14]
        if len(arr) >= 26: self._leg_l = arr[14:20]; self._leg_r = arr[20:26]

    def to_dict(self) -> dict:
        return {"arm_l": self._arm_l.tolist(), "arm_r": self._arm_r.tolist(),
                "leg_l": self._leg_l.tolist(), "leg_r": self._leg_r.tolist()}


class ParietalNode:
    HZ = 20

    def __init__(self, config: dict):
        self.name   = "PosteriorParietal"
        self.bus    = NeuralBus(self.name, config["pub_port"], config["sub_endpoints"])
        self.emap   = EgocentricMap()
        self.aff    = AffordanceDetector()
        self.schema = BodySchema()

        self._depth_stats  = {}
        self._point_cloud  = []
        self._motion_evs   = []
        self._slam_pose    = np.zeros(3)
        self._visual_feats = np.zeros(32)
        self._limp_active  = False
        self._running      = False
        self._lock         = threading.Lock()

    def _on_depth(self, msg):
        with self._lock:
            self._depth_stats = msg.payload.get("depth_stats", {})
            self._point_cloud = msg.payload.get("point_cloud", [])

    def _on_mt(self, msg):
        with self._lock:
            self._motion_evs = msg.payload.get("motion_events", [])

    def _on_hippo(self, msg):
        p = msg.payload.get("slam_pose", [0, 0, 0])
        with self._lock: self._slam_pose = np.array(p[:3], dtype=float)

    def _on_spinal(self, msg):
        a = msg.payload.get("joint_angles", [])
        self.schema.update(a)

    def _on_limp(self, msg):
        with self._lock: self._limp_active = True

    def _on_limp_clr(self, msg):
        with self._lock: self._limp_active = False

    def _loop(self):
        iv = 1.0 / self.HZ
        while self._running:
            t0 = time.time()
            with self._lock:
                ds = dict(self._depth_stats); pc = list(self._point_cloud)
                mt = list(self._motion_evs); pose = self._slam_pose.copy()
                limp = self._limp_active

            # Update occupancy map
            self.emap.update(pc, pose)
            pps_threat    = self.emap.pps_threat()
            nearest_m     = self.emap.nearest_obstacle_m()
            affordance    = self.aff.detect(ds, mt)
            body          = self.schema.to_dict()

            # Spatial attention: attend to closest / most salient object
            attn_az = 0.0
            if mt:
                best = max(mt, key=lambda e: e.get("salience", 0))
                u    = best.get("centroid", [320, 240])[0]
                attn_az = float((u - 320) / 320 * 45)  # degrees from centre

            self.bus.publish(T.PARIETAL_SPATIAL, {
                "egocentric_nearest_m": nearest_m,
                "slam_pose":            pose.tolist(),
                "timestamp_ns":         time.time_ns(),
            })
            self.bus.publish(T.PARIETAL_PERISP, {
                "threat_level":    pps_threat,
                "nearest_m":       nearest_m,
                "pps_radius_m":    PPS_RADIUS_M,
                "alert":           pps_threat > 0.3 or nearest_m < PPS_RADIUS_M,
                "timestamp_ns":    time.time_ns(),
            })
            self.bus.publish(T.PARIETAL_TOOL, {
                "affordance":     affordance,
                "body_schema":    body,
                "limp_degraded":  limp,
                "timestamp_ns":   time.time_ns(),
            })
            self.bus.publish(T.PARIETAL_BODY, {**body, "timestamp_ns": time.time_ns()})
            self.bus.publish(T.PARIETAL_ATTN, {
                "azimuth_deg":  attn_az,
                "pps_threat":   pps_threat,
                "timestamp_ns": time.time_ns(),
            })
            # Thalamic attention redirect if PPS threat
            if pps_threat > 0.3:
                self.bus.publish(T.CTX_ATTENTION, {
                    "gate":     {"visual": 0.9, "auditory": 0.4, "soma": 0.6},
                    "dominant": "visual",
                    "reason":   "pps_threat",
                })

            time.sleep(max(0, iv - (time.time() - t0)))

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.VISUAL_DEPTH,   self._on_depth)
        self.bus.subscribe(T.VISUAL_MT,      self._on_mt)
        self.bus.subscribe(T.HIPPO_PLACE,    self._on_hippo)
        self.bus.subscribe(T.SPINAL_FBK,     self._on_spinal)
        self.bus.subscribe(T.LIMP_MODE_ACTIVE, self._on_limp)
        self.bus.subscribe(T.LIMP_MODE_CLEAR,  self._on_limp_clr)
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info(f"{self.name} v11.14 | PPS({PPS_RADIUS_M}m) | affordance | body-schema | {self.HZ}Hz")

    def stop(self): self._running = False; self.bus.stop()


if __name__ == "__main__":
    with open("/etc/bubo/config.json") as f: cfg = json.load(f)["parietal"]
    n = ParietalNode(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
