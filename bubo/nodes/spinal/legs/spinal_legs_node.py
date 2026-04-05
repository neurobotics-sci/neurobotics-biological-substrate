"""bubo/nodes/spinal/legs/spinal_legs_node.py — v11.14
Spinal Legs: CPG, ZMP, 2×6-DOF IK, SPINAL_HB heartbeat, limp-mode aware.
"""
import time,json,logging,threading,numpy as np
from bubo.shared.bus.neural_bus import NeuralBus,T
from bubo.shared.kinematics.biomimetic_ik import make_human_leg
from bubo.shared.reflexes.spinal_reflexes import LocomotorCPG,FlexorWithdrawalReflex
logger=logging.getLogger("SpinalLegs")

class SpinalLegsNode:
    HZ=100
    def __init__(self,config):
        self.name="SpinalLegs"
        self.bus=NeuralBus(self.name,config["pub_port"],config["sub_endpoints"])
        self.ik_l=make_human_leg("L",np.array([-0.10,-0.10,0.0]))
        self.ik_r=make_human_leg("R",np.array([0.10,-0.10,0.0]))
        self.cpg=LocomotorCPG();self.wd=FlexorWithdrawalReflex()
        self._cmd_l=np.zeros(6);self._cmd_r=np.zeros(6)
        self._cerb_l=np.zeros(6);self._cerb_r=np.zeros(6)
        self._limp_l=None;self._limp_r=None
        self._mlr_drive=0.0;self._resting=False
        self._running=False;self._lock=threading.Lock()

        m=msg.payload.get("motor",{})
        gait=m.get("gait_mode","stand")
        with self._lock:
            self._mlr_drive=m.get("speed_ms",0.0)*0.5 if gait=="walk" else 0.0
            if msg.payload.get("resting"): self._resting=True

    def _on_cerb(self,msg):
        lc=msg.payload.get("leg_correction",[0]*12)
        limp=msg.payload.get("limp_mode",{})
        with self._lock:
            self._cerb_l=np.resize(np.array(lc[:6]),6)
            self._cerb_r=np.resize(np.array(lc[6:12]),6)
            if limp.get("active") and limp.get("limp_leg_cmd"):
                ll=limp["limp_leg_cmd"]
                self._limp_l=np.resize(np.array(ll[:6]),6)
                self._limp_r=np.resize(np.array(ll[6:]),6)
            else:
                self._limp_l=None;self._limp_r=None

    def _on_noci(self,msg):
        zone=msg.payload.get("zone_id","");intensity=float(msg.payload.get("intensity",0))
        if any(k in zone for k in ["foot","leg"]):
            for r in self.wd.compute(zone,intensity):
                with self._lock:
                    if "_L" in r.target_muscle:
                        self._cmd_l=np.clip(self._cmd_l+np.random.uniform(-0.05,0.2,6)*r.command,-np.pi,np.pi)
                    else:
                        self._cmd_r=np.clip(self._cmd_r+np.random.uniform(-0.05,0.2,6)*r.command,-np.pi,np.pi)

    def _on_rest(self,msg):
        with self._lock: self._resting=True;self._mlr_drive=0.0
    def _on_limp_clr(self,msg):
        with self._lock: self._limp_l=None;self._limp_r=None

    def _loop(self):
        iv=1.0/self.HZ
        while self._running:
            t0=time.time()
            if not self._resting:
                cpg_out=self.cpg.step(self._mlr_drive)
            else:
                cpg_out={"leg_l":{},"leg_r":{},"gait_phase":{}}
            with self._lock:
                limp_l=self._limp_l.copy() if self._limp_l is not None else None
                limp_r=self._limp_r.copy() if self._limp_r is not None else None
                cl=np.zeros(6);cr=np.zeros(6)
                for d,arr in [(cpg_out["leg_l"],cl),(cpg_out["leg_r"],cr)]:
                    arr[0]+=d.get("hip_flex_ext",0);arr[3]+=d.get("knee_flex_ext",0);arr[4]+=d.get("ankle_df_pf",0)
                cl=np.clip(cl+self._cmd_l+self._cerb_l,-np.pi,np.pi)
                cr=np.clip(cr+self._cmd_r+self._cerb_r,-np.pi,np.pi)
            if limp_l is not None: cl=limp_l
            if limp_r is not None: cr=limp_r
            for i,j in enumerate(self.ik_l.joints): j.q=float(np.clip(cl[i],j.q_min,j.q_max))
            for i,j in enumerate(self.ik_r.joints): j.q=float(np.clip(cr[i],j.q_min,j.q_max))
            ee_l,_=self.ik_l.forward_kinematics()
            ee_r,_=self.ik_r.forward_kinematics()
            now_ns=time.time_ns()
            self.bus.publish(T.SPINAL_FBK,{"limb":"legs","joint_angles":cl.tolist()+cr.tolist(),
                "ee_l":ee_l.tolist(),"ee_r":ee_r.tolist(),"timestamp_ns":now_ns})
            self.bus.publish(T.SPINAL_HB,{"limb":"legs","t_ns":now_ns,"seq":int(t0*100)%65535})
            self.bus.publish(T.SPINAL_CPG,{"mlr_drive":self._mlr_drive,**cpg_out})
            time.sleep(max(0,iv-(time.time()-t0)))

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.CEREBELL_DELTA,self._on_cerb)
        self.bus.subscribe(T.NOCI_HEAT,self._on_noci)
        self.bus.subscribe(T.NOCI_MECH,self._on_noci)
        self.bus.subscribe(T.MLR_LOCO,lambda m: setattr(self,"_mlr_drive",float(m.payload.get("drive",0))))
        self.bus.subscribe(T.REST_REPAIR,self._on_rest)
        self.bus.subscribe(T.LIMP_MODE_CLEAR,self._on_limp_clr)
        self._running=True
        threading.Thread(target=self._loop,daemon=True).start()
        logger.info(f"{self.name} v11.14 | HB@100Hz | limp-aware | CPG | 2×6-DOF IK")
    def stop(self): self._running=False;self.bus.stop()

if __name__=="__main__":
    with open("/etc/bubo/config.json") as f: cfg=json.load(f)["spinal_legs"]
    n=SpinalLegsNode(cfg);n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
