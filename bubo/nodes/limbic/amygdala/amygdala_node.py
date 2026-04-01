"""bubo/nodes/limbic/amygdala/amygdala_node.py — v50.0
Amygdala: LA/BA/CeA fear learning. v50: social threat_weight from Social node.
OXT suppresses BA→CeA output for bonded individuals (familiar = safe signal).
Nano 4GB (192.168.1.31), VLAN 20.
"""
import time,json,logging,threading,numpy as np
from bubo.shared.bus.neural_bus import NeuralBus,T
from bubo.shared.neuromodulators.neuromod_system import NeuromodState
logger=logging.getLogger("Amygdala")

class LateralAmygdala:
    def __init__(self,n=512):
        self._w=np.ones(n)*0.1;self._lr=0.05
        self._idx={};self._ptr=0;self._n=n
    def high_road(self,features,nm:NeuromodState,aversive=False):
        h=self._hash(features);w=self._get_w(h)
        act=float(np.tanh(w*np.linalg.norm(features)))
        if aversive:
            ne_b=1.0+3.0*nm.NE;delta=self._lr*ne_b*np.linalg.norm(features)
            idx=self._get_or_create(h);self._w[idx]=float(np.clip(self._w[idx]+delta,0,1))
        return {"activation":act,"la_weight":w}
    def _hash(self,f): return int(np.sum(np.round(np.array(f[:8])*10))*1000)%(2**31)
    def _get_w(self,h): return float(self._w[self._idx[h]]) if h in self._idx else 0.1
    def _get_or_create(self,h):
        if h not in self._idx: self._idx[h]=self._ptr%self._n;self._ptr+=1
        return self._idx[h]

class BasalAmygdala:
    def __init__(self): self._ext=0.0;self._lr=0.02
    def vmPFC_regulate(self,s): self._ext=float(np.clip(self._ext+self._lr*s,0,1))
    def net(self,la_act,social_suppress=0.0):
        # Social suppression (OXT→BA): reduces fear expression for bonded
        effective_ext=float(np.clip(self._ext+social_suppress*0.6,0,1))
        return {"net_fear":float(np.clip(la_act-effective_ext,0,1)),"extinction":effective_ext}

class CentralAmygdala:
    def output(self,ba,nm,social_threat_weight=1.0):
        fear=ba["net_fear"];sero_gate=float(1.0-0.5*nm.sero)
        # Social threat_weight scales CeA output: friend=0.15, stranger=1.0
        cea=float(np.clip(fear*sero_gate*1.5*social_threat_weight,0,1))
        return {"cea_activation":cea,"freeze":cea if cea<0.5 else 0.0,"flee":cea if cea>=0.5 else 0.0,
                "hypo_drive":cea*0.8,"lc_drive":cea*0.6,"fear_expressed":cea>0.3,
                "social_modulated":social_threat_weight<1.0}

class AmygdalaNode:
    def __init__(self,config):
        self.name="Amygdala"
        self.bus=NeuralBus(self.name,config["pub_port"],config["sub_endpoints"])
        self.LA=LateralAmygdala();self.BA=BasalAmygdala();self.CeA=CentralAmygdala()
        self._nm=NeuromodState()
        self._social_threat_weight=1.0   # 1.0=stranger, 0.15=bonded
        self._social_oxt_suppress=0.0   # OXT suppression of BA
        self._running=False;self._lock=threading.Lock()

    def _on_noci(self,msg):
        feats=np.resize(np.array(msg.payload.get("features",[0.5]*8)),32)
        with self._lock: nm=self._nm;tw=self._social_threat_weight;oxt=self._social_oxt_suppress
        la=self.LA.high_road(feats,nm,aversive=True)
        ba=self.BA.net(la["activation"],social_suppress=oxt)
        cea=self.CeA.output(ba,nm,social_threat_weight=tw)
        self.bus.publish(T.AMYG_CEA_OUT,{**cea,"la_activation":la["activation"],
                                          "social_threat_weight":tw,"timestamp_ns":time.time_ns()})
        self.bus.publish(T.AMYG_LA_OUT,{"la_activation":la["activation"],"timestamp_ns":time.time_ns()})
        self.bus.publish(T.AMYG_BA_OUT,ba)

    def _on_sensory(self,msg):
        feats=np.resize(np.array(msg.payload.get("features",[])),32)
        if not feats.any(): return
        with self._lock: nm=self._nm;tw=self._social_threat_weight;oxt=self._social_oxt_suppress
        la=self.LA.high_road(feats,nm,aversive=False)
        ba=self.BA.net(la["activation"],social_suppress=oxt)
        cea=self.CeA.output(ba,nm,social_threat_weight=tw)
        if cea["fear_expressed"]:
            self.bus.publish(T.AMYG_CEA_OUT,{**cea,"la_activation":la["activation"],
                                              "timestamp_ns":time.time_ns()})

    def _on_vmPFC(self,msg): self.BA.vmPFC_regulate(float(msg.payload.get("vmPFC_signal",0)))

    def _on_social_threat(self,msg):
        """Social node → update threat weight (bond level modulates amygdala)."""
        with self._lock:
            self._social_threat_weight=float(msg.payload.get("threat_weight",1.0))
            self._social_oxt_suppress=float(msg.payload.get("oxt_suppression",0.0))
        logger.debug(f"Social mod: threat_w={self._social_threat_weight:.2f} oxt_sup={self._social_oxt_suppress:.2f}")

    def _on_neuromod(self,msg):
        p=msg.payload
        with self._lock:
            if "DA" in p: self._nm.DA=float(p["DA"])
            if "NE" in p: self._nm.NE=float(p["NE"])
            if "5HT" in p: self._nm.sero=float(p["5HT"])

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.NOCI_HEAT,self._on_noci)
        self.bus.subscribe(T.NOCI_COLD,self._on_noci)
        self.bus.subscribe(T.NOCI_MECH,self._on_noci)
        self.bus.subscribe(T.TOUCH_SA1,self._on_sensory)
        self.bus.subscribe(T.VISUAL_V1,self._on_sensory)
        self.bus.subscribe(T.VMFPC_REG,self._on_vmPFC)
        self.bus.subscribe(T.SOCIAL_THREAT_MOD,self._on_social_threat)  # v50 NEW
        self.bus.subscribe(T.DA_VTA,self._on_neuromod)
        self.bus.subscribe(T.NE_LC,self._on_neuromod)
        self.bus.subscribe(T.SERO_RAPHE,self._on_neuromod)
        self._running=True
        logger.info(f"{self.name} v50.0 | LA/BA/CeA | social-threat-mod | OXT-suppression")
    def stop(self): self._running=False;self.bus.stop()

if __name__=="__main__":
    with open("/etc/bubo/config.json") as f: cfg=json.load(f)["amygdala"]
    n=AmygdalaNode(cfg);n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
