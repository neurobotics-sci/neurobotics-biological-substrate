from bubo.slam.rtabmap_bridge import RTABMapBridge
"""bubo/nodes/limbic/hippocampus/hippocampus_node.py — v50.0
Hippocampus: EKF-SLAM, DG/CA3/CA1, theta, place cells, social memory tags.
Nano 4GB (192.168.1.30). v50: social encounter locations tagged in SLAM map.
"""
import time,json,logging,threading,numpy as np
from collections import deque
from bubo.shared.bus.neural_bus import NeuralBus,T
from bubo.shared.oscillators.neural_oscillators import HippocampalTheta
from bubo.shared.plasticity.synaptic_plasticity import SynapticTagging
logger=logging.getLogger("Hippocampus")

class EKFSlam:
    MAX_LM=200;INIT_COV=2.0;MAHA_THRESH=9.21;PROC_NOISE=0.01;OBS_NOISE=0.10
    def __init__(self):
        self._x=np.zeros(3);self._P=np.eye(3)*0.001
        self._lms={};self._n=0;self._next_id=0
        self._path=deque(maxlen=2000);self._loop_closures=0
        self._Q=np.diag([0.01**2,0.01**2,0.005**2])
        self._social_markers={}  # landmark_id → social encounter info
    def predict(self,v,omega,dt):
        th=self._x[2];F=np.eye(3+2*self._n)
        F[0,2]=-v*dt*np.sin(th);F[1,2]=v*dt*np.cos(th)
        self._x[:3]+=np.array([v*dt*np.cos(th),v*dt*np.sin(th),omega*dt])
        self._x[2]=float((self._x[2]+np.pi)%(2*np.pi)-np.pi)
        Q=np.zeros_like(F);Q[:3,:3]=self._Q;self._P=F@self._P@F.T+Q
        self._path.append({"x":float(self._x[0]),"y":float(self._x[1]),"yaw":float(self._x[2])})
    def update(self,obs_xy):
        obs=np.array(obs_xy);best_id=None;best_d2=self.MAHA_THRESH
        for lid,lm in self._lms.items():
            innov=obs-lm["pos"];S=lm["cov"]+np.eye(2)*self.OBS_NOISE**2
            try: d2=float(innov@np.linalg.inv(S)@innov)
            except: continue
            if d2<best_d2: best_d2=d2;best_id=lid
        if best_id is not None:
            lm=self._lms[best_id];innov=obs-lm["pos"]
            K=lm["cov"]@np.linalg.inv(lm["cov"]+np.eye(2)*self.OBS_NOISE**2)
            self._x[:2]+=K@innov;lm["pos"]+=K@innov
            lm["cov"]=(np.eye(2)-K)@lm["cov"];lm["count"]=lm.get("count",0)+1
            if np.linalg.norm(innov)<0.25 and lm["count"]>5: self._loop_closures+=1;return "loop"
            return "updated"
        elif self._n<self.MAX_LM:
            lid=self._next_id;self._next_id+=1
            self._lms[lid]={"pos":obs.copy(),"cov":np.eye(2)*self.INIT_COV,"count":1}
            self._n+=1;return "new"
        return "none"
    def tag_social(self,face_id,name,bond_level):
        """Tag current position with a social encounter."""
        pose=self._x[:3].copy()
        self._social_markers[face_id]={"pose":pose.tolist(),"name":name,
                                         "bond":bond_level,"ts":time.time()}
    @property
    def pose(self): return {"x":float(self._x[0]),"y":float(self._x[1]),"yaw":float(self._x[2])}
    @property
    def n_landmarks(self): return self._n
    @property
    def social_locations(self): return dict(self._social_markers)

class HippocampusNode:
    HZ=10
    def __init__(self,config):
        self.name="Hippocampus"
        self.bus=NeuralBus(self.name,config["pub_port"],config["sub_endpoints"])
        self.slam=EKFSlam();self.theta=HippocampalTheta()
        self.tagging=SynapticTagging()
        self._v_ms=0.0;self._omega=0.0
        self._nm={"DA":0.5,"NE":0.2,"ACh":0.5}
        self._episodes=deque(maxlen=5000)
        self._running=False
        self._slam=RTABMapBridge();self._lock=threading.Lock();self._t_last=time.time()
    def _on_depth(self,msg):
        pts=msg.payload.get("point_cloud",[]);pose=self.slam.pose
        for pt in pts[:30]:
            if len(pt)<3 or pt[2]<0.1 or pt[2]>8: continue
            th=pose["yaw"];cos_th,sin_th=np.cos(th),np.sin(th)
            wx=pose["x"]+cos_th*pt[2]-sin_th*pt[0]
            wy=pose["y"]+sin_th*pt[2]+cos_th*pt[0]
            self.slam.update([wx,wy])
    def _on_vest(self,msg):
        accel=msg.payload.get("linear_accel_g",[0,0,0])
        gyro=msg.payload.get("gyro_rps",[0,0,0])
        self._v_ms=float(np.hypot(accel[0],accel[1]))*9.81*0.05;self._omega=float(gyro[2])
    def _on_noci(self,msg):
        feats=np.array(msg.payload.get("features",[0.5]*8))
        intensity=float(msg.payload.get("intensity",0.5))
        ne=self._nm.get("NE",0.2);importance=float(np.clip(ne*0.4+intensity*0.6,0,1))
        tid=f"ep_{time.time_ns()}"
        if self.theta.encoding_gate:
            self._episodes.append({"trace_id":tid,"timestamp":time.time(),
                                    "importance":importance,"ne":ne,
                                    "emotion_tag":{"valence":-intensity,"fear":intensity}})
            self.tagging.set_tag(tid,importance)
            self.bus.publish(T.HIPPO_ENCODE,{"trace_id":tid,"importance":importance,
                                              "timestamp_ns":time.time_ns()})
    def _on_social_face(self,msg):
        """Tag SLAM location with recognised face encounter."""
        fid=msg.payload.get("face_id");name=msg.payload.get("name","?")
        bond=float(msg.payload.get("bond_level",0))
        if fid: self.slam.tag_social(fid,name,bond)
        # Encode positive social episode
        if bond>0.3 and self.theta.encoding_gate:
            tid=f"soc_{time.time_ns()}"
            self._episodes.append({"trace_id":tid,"timestamp":time.time(),
                "importance":float(0.4+bond*0.4),"ne":self._nm.get("NE",0.2),
                "emotion_tag":{"valence":bond,"fear":0.0,"social":True}})
    def _on_neuromod(self,msg):
        self._nm.update({k:float(v) for k,v in msg.payload.items() if k in self._nm})
    def _slam_loop(self):
        iv=1.0/self.HZ
        while self._running:
            t0=time.time();dt=max(t0-self._t_last,0.001);self._t_last=t0
            self.slam.predict(self._v_ms,self._omega,dt)
            pose=self.slam.pose
            theta_st=self.theta.step(dt,self._v_ms,self._nm.get("ACh",0.5))
            self.bus.set_phase(theta_st.phase)
            self.bus.publish(T.HIPPO_PLACE,{**pose,"n_landmarks":self.slam.n_landmarks,
                "loop_closures":self.slam._loop_closures,"timestamp_ns":time.time_ns()})
            self.bus.publish(T.HIPPO_THETA,{"phase":theta_st.phase,"amp":theta_st.amplitude,
                "encoding_gate":self.theta.encoding_gate,"timestamp_ns":time.time_ns()})
            self.bus.publish(T.HIPPO_CONTEXT,{"context_vector":[pose["x"],pose["y"],pose["yaw"]],
                "slam_pose":[pose["x"],pose["y"],pose["yaw"]],
                "social_locations":self.slam.social_locations,"timestamp_ns":time.time_ns()})
            time.sleep(max(0,iv-(time.time()-t0)))
    def start(self):
        self.bus.start()
        self.bus.subscribe(T.VISUAL_DEPTH,self._on_depth)
        self.bus.subscribe(T.VESTIBULAR,self._on_vest)
        self.bus.subscribe(T.NOCI_HEAT,self._on_noci)
        self.bus.subscribe(T.NOCI_MECH,self._on_noci)
        self.bus.subscribe(T.DA_VTA,self._on_neuromod)
        self.bus.subscribe(T.NE_LC,self._on_neuromod)
        self.bus.subscribe(T.ACH_NBM,self._on_neuromod)
        self.bus.subscribe(T.SOCIAL_FACE,self._on_social_face)
        self._running=True
        threading.Thread(target=self._slam_loop,daemon=True).start()
        logger.info(f"{self.name} v50.0 | EKF-SLAM+social-tags | theta | {self.HZ}Hz")
    def stop(self): self._running=False;self.bus.stop()

if __name__=="__main__":
    with open("/etc/bubo/config.json") as f: cfg=json.load(f)["hippocampus"]
    n=HippocampusNode(cfg);n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
