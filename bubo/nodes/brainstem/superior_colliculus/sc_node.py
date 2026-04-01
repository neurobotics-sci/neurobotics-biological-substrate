"""
bubo/nodes/brainstem/superior_colliculus/sc_node.py — v5550
Superior Colliculus: SGBM stereo + foveation + PPS proximity (absorbed from removed parietal node).

v5550: PPS (peripersonal space) proximity check absorbed here from parietal Orin node.
  Simple radial distance check on depth map: nearest_m < 0.5 → PPS_ALERT
  Adds ~0.8ms to SC loop — within budget at 30fps (33ms period).
"""
import time, json, logging, threading
import numpy as np
from collections import deque
from bubo.shared.bus.neural_bus import NeuralBus, T
from bubo.shared.kinematics.biomimetic_ik import make_eye, make_cervical_spine
from bubo.vor.vor_controller import VORController

logger = logging.getLogger("SuperiorColliculus")
try: import cv2; HAS_CV2=True
except ImportError: HAS_CV2=False

PPS_RADIUS_M = 0.50  # peripersonal space: 50cm radius


class StereoProc:
    def __init__(self, baseline=0.065, focal=580, cx=320, cy=240):
        self.B=baseline; self.f=focal; self.cx=cx; self.cy=cy
        if HAS_CV2:
            self._sgbm=cv2.StereoSGBM_create(minDisparity=0, numDisparities=64,
                blockSize=5, P1=200, P2=800, uniquenessRatio=10,
                speckleWindowSize=100, speckleRange=32)
    def compute(self, gl, gr):
        if HAS_CV2:
            d=self._sgbm.compute(gl,gr).astype(np.float32)/16.0
            with np.errstate(divide='ignore',invalid='ignore'):
                z=np.where(d>1, self.f*self.B/d, 0.0)
        else:
            diff=np.abs(gl.astype(float)-np.roll(gr.astype(float),4,axis=1))
            d=np.clip(64-diff*0.5,0,64).astype(np.float32)
            z=np.where(d>1, self.f*self.B/d, 0.0)
        return d, z.astype(np.float32)
    def deproject(self, u, v, z):
        return np.array([(u-self.cx)*z/self.f, (v-self.cy)*z/self.f, z])


class MotionDet:
    def __init__(self, h, w):
        self._acc=np.zeros((h,w),np.float32); self._prev=None; self._t=time.time()
        self._cx=w/2; self._cy=h/2
    def detect(self, gray):
        dt=max(time.time()-self._t,0.001); self._t=time.time()
        if self._prev is None or self._prev.shape!=gray.shape:
            self._prev=gray.copy(); return []
        diff=np.abs(gray.astype(np.float32)-self._prev.astype(np.float32))
        self._prev=gray.copy()
        a=1-np.exp(-dt/0.6); self._acc=(1-a)*self._acc+a*diff
        binary=(self._acc>12).astype(np.uint8)
        ys,xs=np.where(binary)
        if len(xs)<80: return []
        u,v=float(np.mean(xs)),float(np.mean(ys))
        ecc=float(np.hypot(u-self._cx,v-self._cy)*0.058)
        return [{"centroid":(u,v),"ecc_deg":ecc,"looming":float(np.sum(binary))>1200,
                 "salience":float(np.clip(ecc/20+0.1,0,1)),"vel_pxf":float(np.mean(self._acc))}]


class PPSChecker:
    """
    Peripersonal space check absorbed from removed parietal node.
    Simple: find nearest non-zero depth pixel in ±30° from body centre.
    Cost: ~0.8ms on BeagleBoard vs 8ms for full voxel occupancy map.
    """
    def check(self, depth_m: np.ndarray, baseline_m: float = 0.065) -> dict:
        if depth_m is None or depth_m.size == 0:
            return {"nearest_m": 99.0, "pps_alert": False, "threat_level": 0.0}
        h, w = depth_m.shape
        # Only look in central 60% of frame (body peripersonal space)
        roi = depth_m[h//4:3*h//4, w//5:4*w//5]
        valid = roi[roi > 0.05]
        nearest_m = float(np.min(valid)) if valid.size > 0 else 99.0
        pps_alert = nearest_m < PPS_RADIUS_M
        threat = float(np.clip(1.0 - nearest_m / PPS_RADIUS_M, 0, 1)) if pps_alert else 0.0
        return {"nearest_m": nearest_m, "pps_alert": pps_alert, "threat_level": threat}


class SCNode:
    HZ = 30

    def __init__(self, config: dict):
        self.name = "SuperiorColliculus"
        self.bus  = NeuralBus(self.name, config["pub_port"], config["sub_endpoints"])
        self.sp   = StereoProc(baseline=config.get("baseline_m", 0.065))
        self.mot  = MotionDet(480, 640)
        self.pps  = PPSChecker()   # absorbed from parietal
        self.eye_l = make_eye("L")
        self.eye_r = make_eye("R")
        self.neck  = make_cervical_spine()
        self.vor   = VORController(self.bus)

        self._audio_az  = 0.0
        self._gyro_rps  = [0.0, 0.0, 0.0]
        self._running   = False
        self._cap_l = self._cap_r = None

    def _on_audio(self, msg):
        az = msg.payload.get("azimuth_deg", 0)
        if az: self._audio_az = float(az)

    def _on_vest(self, msg):
        g = msg.payload.get("gyro_rps", [0,0,0])
        self._gyro_rps = g
        self.vor.update_vestibular(g)

    def _grab(self):
        def cap(c):
            if c and HAS_CV2 and c.isOpened():
                ok, f = c.read()
                if ok: return cv2.resize(f, (640,480))
            f=np.random.randint(40,120,(480,640,3),dtype=np.uint8)
            t=time.time()
            bx=int((np.sin(t*0.7)*0.3+0.5)*640); by=240
            f[max(0,by-30):by+30, max(0,bx-30):bx+30]=[200,160,60]
            return f
        bl=cap(self._cap_l); br=cap(self._cap_r)
        def gray(b):
            if HAS_CV2: return cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            return (0.299*b[:,:,2]+0.587*b[:,:,1]+0.114*b[:,:,0]).astype(np.uint8)
        return bl, br, gray(bl), gray(br)

    def _loop(self):
        if HAS_CV2:
            self._cap_l = cv2.VideoCapture(0)
            self._cap_r = cv2.VideoCapture(2)
        iv = 1.0 / self.HZ
        while self._running:
            t0 = time.time()
            bl, br, gl, gr = self._grab()
            disp, depth = self.sp.compute(gl, gr)

            # PPS check (absorbed from parietal, ~0.8ms)
            pps = self.pps.check(depth)

            vd = depth[depth > 0]
            d_stats = {
                "min_m":    float(vd.min())    if len(vd) else 0.0,
                "mean_m":   float(vd.mean())   if len(vd) else 0.0,
                "median_m": float(np.median(vd)) if len(vd) else 0.0,
            }
            step=16; ys,xs=np.mgrid[0:480:step,0:640:step]; zs=depth[ys,xs]; mask=zs>0.05
            pts=[self.sp.deproject(float(xs[mask].flat[i]),float(ys[mask].flat[i]),
                                   float(zs[mask].flat[i])).tolist() for i in range(min(int(mask.sum()),150))]

            evs = self.mot.detect(gl)
            now_ns = time.time_ns()

            self.bus.publish(T.VISUAL_DEPTH, {
                "depth_stats": d_stats, "point_cloud": pts,
                "timestamp_ns": now_ns,
            })
            self.bus.publish(T.VISUAL_MT, {
                "motion_events": evs[:4], "n_events": len(evs),
                "looming_alert": any(e.get("looming") for e in evs),
                "max_salience":  float(max((e.get("salience",0) for e in evs), default=0)),
                "timestamp_ns":  now_ns,
            })

            # Publish PPS threat (absorbed from parietal)
            self.bus.publish(T.PARIETAL_PERISP, {
                **pps, "pps_radius_m": PPS_RADIUS_M,
                "source": "sc_pps_check", "timestamp_ns": now_ns,
            })
            if pps["pps_alert"]:
                self.bus.publish(T.CTX_ATTENTION, {
                    "gate": {"visual":0.9,"auditory":0.4,"soma":0.6},
                    "dominant": "visual", "reason": "pps_threat",
                    "timestamp_ns": now_ns,
                })

            # Saccade to most salient motion event
            if evs:
                best = max(evs, key=lambda e: e.get("salience",0))
                u, v = best["centroid"]
                z = float(depth[max(0,int(v)-2):int(v)+2, max(0,int(u)-2):int(u)+2].mean()) or 1.5
                tgt = self.sp.deproject(u, v, z)
                ql  = self.eye_l.solve(tgt - np.array([-self.sp.B/2,0,0]))
                qr  = self.eye_r.solve(tgt - np.array([ self.sp.B/2,0,0]))
                qn  = self.neck.solve(tgt)
                self.bus.publish(T.SC_SACCADE, {
                    "target_3d": tgt.tolist(), "target_px": [u,v],
                    "vergence_deg": float(2*np.degrees(np.arctan2(self.sp.B/2, max(z,0.05)))),
                    "eye_l_rad": ql.tolist(), "eye_r_rad": qr.tolist(),
                    "neck_rad":  qn.tolist(), "timestamp_ns": now_ns,
                })
                # Suppress VOR during intentional saccade
                self.vor.suppress(duration_ms=120.0)

            time.sleep(max(0, iv-(time.time()-t0)))

    def start(self):
        self.bus.start()
        self.vor.start()
        self.bus.subscribe(T.AUDITORY_SPAT, self._on_audio)
        self.bus.subscribe(T.VESTIBULAR,    self._on_vest)
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info(f"{self.name} v5550 | SGBM+PPS(absorbed) | VOR | cv2={HAS_CV2} | {self.HZ}Hz")

    def stop(self):
        if HAS_CV2:
            if self._cap_l: self._cap_l.release()
            if self._cap_r: self._cap_r.release()
        self.vor.stop()
        self._running = False
        self.bus.stop()


if __name__ == "__main__":
    with open("/etc/bubo/config.json") as f: cfg=json.load(f)["superior_colliculus"]
    n=SCNode(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
