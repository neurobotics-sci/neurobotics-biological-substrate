"""
bubo/nodes/cortex/social/social_node.py — v6500
Social bonding node with 200-dim latent emotion model.
"""
import time, json, logging, threading
import numpy as np
from bubo.shared.bus.neural_bus import NeuralBus, T
from bubo.social.latent_emotion.latent_emotion_model import MultiPersonSocialMemory

logger = logging.getLogger("SocialNode")


class SocialNode:
    HZ = 10
    def __init__(self, config: dict):
        self.name   = "SocialNode"
        self.bus    = NeuralBus(self.name, config["pub_port"], config["sub_endpoints"])
        self.memory = MultiPersonSocialMemory()
        self._face_detected = False
        self._face_emb  = None
        self._face_bbox = [0,0,1,1]
        self._fear      = 0.0
        self._da        = 0.5
        self._body_emb  = np.zeros(20)
        self._running   = False
        self._lock      = threading.Lock()

    def _on_visual(self, msg):
        faces = msg.payload.get("faces",[])
        if faces:
            f = faces[0]
            with self._lock:
                self._face_detected = True
                self._face_bbox = f.get("bbox",[0,0,100,100])
                emb = f.get("embedding")
                self._face_emb = np.array(emb, dtype=float) if emb else np.zeros(128)
        else:
            with self._lock:
                self._face_detected = False

    def _on_cea(self, msg):
        with self._lock: self._fear = float(msg.payload.get("cea_activation",0))

    def _on_da(self, msg):
        with self._lock: self._da = float(msg.payload.get("DA",0.5))

    def _on_spinal(self, msg):
        joints = msg.payload.get("joint_angles",[])
        with self._lock:
            self._body_emb = np.resize(np.array(joints[:20],dtype=float), 20)

    def _loop(self):
        iv = 1.0 / self.HZ
        while self._running:
            t0 = time.time()
            with self._lock:
                det  = self._face_detected
                emb  = self._face_emb.copy() if self._face_emb is not None else None
                bbox = list(self._face_bbox)
                body = self._body_emb.copy()
                fear = self._fear
                da   = self._da

            if det and emb is not None:
                result = self.memory.process_face(
                    face_id=None, name="person",
                    face_emb=emb, body_emb=body,
                    bond_level=0.0)
                now_ns = time.time_ns()

                self.bus.publish(T.SOCIAL_FACE, {**result, "timestamp_ns": now_ns})
                self.bus.publish(T.SOCIAL_THREAT_MOD, {
                    "threat_weight": result["threat_weight"],
                    "oxt_suppression": float(1.0 - result["threat_weight"]),
                    "timestamp_ns": now_ns,
                })
                self.bus.publish(T.SOCIAL_BOND, {
                    "bond_level":  result.get("bond_level",0),
                    "da_boost":    result.get("da_boost",0),
                    "sero_boost":  0.0,
                    "valence":     result.get("valence",0),
                    "arousal":     result.get("arousal",0),
                    "timestamp_ns": now_ns,
                })
                if result["da_boost"] > 0.05 and fear < 0.4 and da > 0.4:
                    self.bus.publish(T.SOCIAL_APPROACH, {
                        "approach_drive": float(result.get("bond_level",0) * (1.0 - fear)),
                        "name": result.get("name","person"),
                        "timestamp_ns": now_ns,
                    })
            time.sleep(max(0, iv - (time.time() - t0)))

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.VISUAL_V1,    self._on_visual)
        self.bus.subscribe(T.AMYG_CEA_OUT, self._on_cea)
        self.bus.subscribe(T.DA_VTA,       self._on_da)
        self.bus.subscribe(T.SPINAL_FBK,   self._on_spinal)
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info(f"{self.name} v6500 | 200-dim latent emotion | {self.HZ}Hz")

    def stop(self):
        self.memory.save()
        self._running = False
        self.bus.stop()


if __name__ == "__main__":
    with open("/etc/bubo/config.json") as f: cfg=json.load(f)["social"]
    n = SocialNode(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
