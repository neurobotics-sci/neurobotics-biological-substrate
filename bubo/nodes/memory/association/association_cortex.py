"""
bubo/nodes/memory/association/association_cortex.py — v11.14
Superior Temporal Sulcus + Mirror Neuron System + Temporal Binding Window.
Nano 4GB (192.168.1.34). Binds coincident cross-modal events within 80ms.
"""
import time, json, logging, threading
import numpy as np
from collections import deque
from typing import Optional, Tuple
from bubo.shared.bus.neural_bus import NeuralBus, T

logger = logging.getLogger("AssociationCortex")

TBW_MS = 80.0   # temporal binding window


class TemporalBinder:
    def __init__(self):
        self._vis_t = self._aud_t = self._som_t = 0.0
        self._vis_f = self._aud_f = self._som_f = []
    def update_vis(self, f): self._vis_t=time.time()*1000; self._vis_f=f
    def update_aud(self, f): self._aud_t=time.time()*1000; self._aud_f=f
    def update_som(self, f): self._som_t=time.time()*1000; self._som_f=f
    def check(self) -> Tuple[bool, dict]:
        now=time.time()*1000; bound={}
        if now-self._vis_t<TBW_MS*3: bound["visual"]=self._vis_f
        if now-self._aud_t<TBW_MS*3: bound["audio"]=self._aud_f
        if now-self._som_t<TBW_MS*3: bound["soma"]=self._som_f
        if "visual" in bound and "audio" in bound and abs(self._vis_t-self._aud_t)<TBW_MS:
            return True, bound
        return False, bound


class CrossModalAttention:
    N_HEADS=4; DIM=64
    def __init__(self):
        rng=np.random.default_rng(1)
        self._Wq=rng.standard_normal((self.N_HEADS,self.DIM,32))*0.1
        self._Wk=rng.standard_normal((self.N_HEADS,self.DIM,32))*0.1
        self._Wv=rng.standard_normal((self.N_HEADS,self.DIM,32))*0.1
    def attend(self, qs, ks, vs):
        if not qs or not ks: return np.zeros(self.DIM), np.zeros((1,1))
        q=np.array([np.resize(x,32) for x in qs]); k=np.array([np.resize(x,32) for x in ks]); v=np.array([np.resize(x,32) for x in vs])
        outs=[]; sc=1.0/np.sqrt(self.DIM/self.N_HEADS)
        for h in range(self.N_HEADS):
            Q=q@self._Wq[h].T; K=k@self._Wk[h].T; V=v@self._Wv[h].T
            s=Q@K.T*sc; s-=s.max(axis=1,keepdims=True)
            a=np.exp(s)/(np.exp(s).sum(axis=1,keepdims=True)+1e-8)
            outs.append(np.mean(a@V,axis=0))
        out=np.mean(outs,axis=0); out/=(np.linalg.norm(out)+1e-8)
        return out, np.mean([np.exp(q@k.T*sc) for _ in range(1)],axis=0)


class MirrorNeuronSystem:
    THRESH=0.40
    TEMPLATES={"reach_right":np.array([0.7,0.3,0.1,0.0,0.0]),"reach_left":np.array([0.7,0.0,0.3,0.0,0.0]),
               "grasp":np.array([0.3,0.2,0.2,0.8,0.0]),"step_forward":np.array([0.2,0.0,0.0,0.0,0.9]),
               "wave":np.array([0.6,0.4,0.0,0.0,0.0])}
    def match(self, v) -> Tuple[Optional[str],float]:
        if not len(v): return None,0.0
        vn=np.resize(v,5); vn/=(np.linalg.norm(vn)+1e-8)
        best=None; bs=0.0
        for a,t in self.TEMPLATES.items():
            tn=t/(np.linalg.norm(t)+1e-8); s=float(np.dot(vn,tn))
            if s>bs: bs=s; best=a
        return (best,bs) if bs>=self.THRESH else (None,bs)


class AssociationCortexNode:
    HZ=20
    def __init__(self, config):
        self.name="AssociationCortex"
        self.bus=NeuralBus(self.name,config["pub_port"],config["sub_endpoints"])
        self.tbw=TemporalBinder(); self.attn=CrossModalAttention(); self.mns=MirrorNeuronSystem()
        self._vf=np.zeros(32); self._af=np.zeros(16); self._sf=np.zeros(8)
        self._vm=np.zeros(5); self._da=0.6; self._running=False; self._lock=threading.Lock()

    def _on_vis(self,msg):
        f=msg.payload.get("features",[]); mt=msg.payload.get("motion_events",[])
        with self._lock:
            if f: self._vf=np.resize(np.array(f,dtype=float),32); self.tbw.update_vis(f)
            if mt and len(mt)>0:
                e=mt[0]; self._vm=np.array([e.get("vel_pxf",0)/50,e.get("ecc_deg",0)/45,
                    float(e.get("looming",False)),e.get("salience",0),0])
    def _on_aud(self,msg):
        f=msg.payload.get("features",[])
        with self._lock:
            if f: self._af=np.resize(np.array(f,dtype=float),16); self.tbw.update_aud(f)
    def _on_som(self,msg):
        f=msg.payload.get("features",[])
        with self._lock:
            if f: self._sf=np.resize(np.array(f,dtype=float),8); self.tbw.update_som(f)
    def _on_da(self,msg):
        with self._lock: self._da=float(msg.payload.get("DA",0.6))

    def _loop(self):
        iv=1.0/self.HZ
        while self._running:
            t0=time.time()
            with self._lock: vf=self._vf.copy(); af=self._af.copy(); sf=self._sf.copy(); vm=self._vm.copy(); da=self._da
            bound,mods=self.tbw.check()
            qs=[v for v in [vf,af,sf] if v.any()]
            assoc,_=self.attn.attend(qs,qs,qs)
            mirror,mconf=self.mns.match(vm)
            energies={"visual":float(np.mean(np.abs(vf))),"audio":float(np.mean(np.abs(af))),"soma":float(np.mean(np.abs(sf)))}
            dom=max(energies,key=energies.get)
            gate={m:(0.9 if m==dom else 0.3) for m in energies}
            self.bus.publish(T.CTX_ASSOC,{"assoc_vector":assoc[:16].tolist(),"binding_occurred":bound,
                "bound_modalities":list(mods.keys()),"dominant_modality":dom,"attention_gate":gate,
                "mirror_action":mirror,"mirror_confidence":mconf,"features":assoc[:8].tolist(),
                "timestamp_ns":time.time_ns()})
            self.bus.publish(T.CTX_ATTENTION,{"gate":gate,"dominant":dom,"timestamp_ns":time.time_ns()})
            if mirror and mconf>0.5:
                self.bus.publish(T.CTX_PFC_CMD,{"action":mirror,"source":"mirror_neuron","confidence":mconf,"motor":{}})
            time.sleep(max(0,iv-(time.time()-t0)))

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.VISUAL_V1,self._on_vis); self.bus.subscribe(T.VISUAL_MT,self._on_vis)
        self.bus.subscribe(T.AUDITORY_A1,self._on_aud); self.bus.subscribe(T.TOUCH_SA1,self._on_som)
        self.bus.subscribe(T.DA_VTA,self._on_da)
        self._running=True; threading.Thread(target=self._loop,daemon=True).start()
        logger.info(f"{self.name} v11.14 | TBW={TBW_MS}ms | CrossAttn | MNS | {self.HZ}Hz")
    def stop(self): self._running=False; self.bus.stop()

if __name__=="__main__":
    with open("/etc/bubo/config.json") as f: cfg=json.load(f)["association"]
    n=AssociationCortexNode(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
