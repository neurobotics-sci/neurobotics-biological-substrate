"""bubo/nodes/spinal/arms/spinal_arms_node.py — v11.14
Spinal Arms: 2×7-DOF IK, reflexes, SPINAL_HB heartbeat (100Hz), limp-mode aware.
"""
import time,json,logging,threading,numpy as np
from bubo.shared.bus.neural_bus import NeuralBus,T
from bubo.shared.kinematics.biomimetic_ik import make_human_arm
from bubo.shared.reflexes.spinal_reflexes import FlexorWithdrawalReflex
logger=logging.getLogger("SpinalArms")

class SpinalArmsNode:
    HZ=100; HB_INTERVAL=1.0/100  # publish heartbeat at 100Hz

    def __init__(self,config):
        self.name="SpinalArms"
        self.bus=NeuralBus(self.name,config["pub_port"],config["sub_endpoints"])
        self.ik_l=make_human_arm("L",np.array([-0.20,0.14,0.0]))
        self.ik_r=make_human_arm("R",np.array([0.20,0.14,0.0]))
        self.wd=FlexorWithdrawalReflex()
        self._cmd_l=np.zeros(7);self._cmd_r=np.zeros(7)
        self._cerb_l=np.zeros(7);self._cerb_r=np.zeros(7)
        self._limp_l=None;self._limp_r=None
        self._resting=False;self._running=False;self._lock=threading.Lock()

    def _on_pfc(self,msg):
        m=msg.payload.get("motor",{})
        with self._lock:
            self._cmd_l=np.resize(np.array(m.get("arm_l",self._cmd_l.tolist())),7)
            self._cmd_r=np.resize(np.array(m.get("arm_r",self._cmd_r.tolist())),7)

    def _on_cerb(self,msg):
        ac=msg.payload.get("arm_correction",[0]*14)
        limp=msg.payload.get("limp_mode",{})
        with self._lock:
            self._cerb_l=np.resize(np.array(ac[:7]),7)
            self._cerb_r=np.resize(np.array(ac[7:14]),7)
            if limp.get("active") and limp.get("limp_arm_cmd"):
                la=limp["limp_arm_cmd"]
                self._limp_l=np.resize(np.array(la[:7]),7)
                self._limp_r=np.resize(np.array(la[7:]),7)
            else:
                self._limp_l=None;self._limp_r=None

    def _on_noci(self,msg):
        zone=msg.payload.get("zone_id","");intensity=float(msg.payload.get("intensity",0))
        for r in self.wd.compute(zone,intensity):
            with self._lock:
                if "_L" in r.target_muscle:
                    self._cmd_l=np.clip(self._cmd_l+np.random.uniform(-0.05,0.2,7)*r.command,-np.pi,np.pi)
                else:
                    self._cmd_r=np.clip(self._cmd_r+np.random.uniform(-0.05,0.2,7)*r.command,-np.pi,np.pi)

    def _on_rest(self,msg): self._resting=True
    def _on_limp_clr(self,msg):
        with self._lock: self._limp_l=None;self._limp_r=None

    def _loop(self):
        iv=1.0/self.HZ
        while self._running:
            t0=time.time()
            with self._lock:
                limp_l=self._limp_l.copy() if self._limp_l is not None else None
                limp_r=self._limp_r.copy() if self._limp_r is not None else None
                cl=np.clip(self._cmd_l+self._cerb_l,-np.pi,np.pi)
                cr=np.clip(self._cmd_r+self._cerb_r,-np.pi,np.pi)

            # Limp mode: use cerebellar direct drive
            if limp_l is not None: cl=limp_l
            if limp_r is not None: cr=limp_r

            for i,j in enumerate(self.ik_l.joints): j.q=float(np.clip(cl[i],j.q_min,j.q_max))
            for i,j in enumerate(self.ik_r.joints): j.q=float(np.clip(cr[i],j.q_min,j.q_max))
            ee_l,_=self.ik_l.forward_kinematics()
            ee_r,_=self.ik_r.forward_kinematics()
            now_ns=time.time_ns()
            self.bus.publish(T.SPINAL_FBK,{"limb":"arms","joint_angles":cl.tolist()+cr.tolist(),
                "ee_l":ee_l.tolist(),"ee_r":ee_r.tolist(),"timestamp_ns":now_ns})
            # v11.14: heartbeat (100Hz) — cerebellum watchdog
            self.bus.publish(T.SPINAL_HB,{"limb":"arms","t_ns":now_ns,"seq":int(t0*100)%65535})
            time.sleep(max(0,iv-(time.time()-t0)))

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.CTX_PFC_CMD,self._on_pfc)
        self.bus.subscribe(T.CEREBELL_DELTA,self._on_cerb)
        self.bus.subscribe(T.NOCI_HEAT,self._on_noci)
        self.bus.subscribe(T.NOCI_MECH,self._on_noci)
        self.bus.subscribe(T.REST_REPAIR,self._on_rest)
        self.bus.subscribe(T.LIMP_MODE_CLEAR,self._on_limp_clr)
        self._running=True
        threading.Thread(target=self._loop,daemon=True).start()
        logger.info(f"{self.name} v11.14 | HB@100Hz | limp-aware | 2×7-DOF IK")
    def stop(self): self._running=False;self.bus.stop()

if __name__=="__main__":
    with open("/etc/bubo/config.json") as f: cfg=json.load(f)["spinal_arms"]
    n=SpinalArmsNode(cfg);n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
