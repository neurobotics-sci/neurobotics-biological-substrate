"""bubo/shared/bus/neural_bus.py — v50.0 VLAN-aware + PTP timestamps"""
import zmq,json,time,threading,logging,numpy as np
from dataclasses import dataclass,asdict
from typing import Callable,Dict,List,Optional
logger=logging.getLogger("NeuralBus")

class T:
    # VLAN 20 — Low-Road (reflex)
    TOUCH_SA1=b"AFF_TOUCH_SA1";TOUCH_SA2=b"AFF_TOUCH_SA2"
    NOCI_HEAT=b"AFF_NOCI_HEAT";NOCI_COLD=b"AFF_NOCI_COLD";NOCI_MECH=b"AFF_NOCI_MECH"
    PROPRIO_IA=b"AFF_PROP_IA";PROPRIO_IB=b"AFF_PROP_IB"
    VISUAL_V1=b"AFF_VIS_V1";VISUAL_MT=b"AFF_VIS_MT";VISUAL_DEPTH=b"AFF_VIS_DEPTH"
    VISUAL_FACE=b"AFF_VIS_FACE"
    AUDITORY_A1=b"AFF_AUD_A1";AUDITORY_SPAT=b"AFF_AUD_SPAT";VESTIBULAR=b"AFF_VEST"
    THERMAL_WARM=b"AFF_THERM_WARM";THERMAL_COOL=b"AFF_THERM_COOL"
    SPINAL_REFLEX=b"SPN_REFLEX";SPINAL_CPG=b"SPN_CPG";SPINAL_FBK=b"SPN_FBK";SPINAL_HB=b"SPN_HB"
    CEREBELL_DELTA=b"CRB_DELTA";CLIMBING_FIBRE=b"CRB_CLIMB";EFFERENCE_COPY=b"CRB_EFF_CP"
    SAFETY_FREEZE=b"SFY_FREEZE";SAFETY_ZMP_FAIL=b"SFY_ZMP";NOD_OFF=b"SFY_NODOFF"
    LIMP_MODE_ACTIVE=b"SFY_LIMP";LIMP_MODE_CLEAR=b"SFY_LIMP_CLR"
    VOR_CMD=b"VOR_CMD";VOR_EYE_FB=b"VOR_EYE_FB";VOR_SUPPRESS=b"VOR_SUPP"
    REFLEX_ASR=b"RFX_ASR";REFLEX_BLINK=b"RFX_BLINK";REFLEX_TLR=b"RFX_TLR"
    REFLEX_ATNR=b"RFX_ATNR";REFLEX_GRASP=b"RFX_GRASP";REFLEX_MORO=b"RFX_MORO"
    REFLEX_PLR=b"RFX_PLR";REFLEX_OKR=b"RFX_OKR"
    # VLAN 10 — High-Road (logic)
    AMYG_LA_OUT=b"LMB_LA_OUT";AMYG_CEA_OUT=b"LMB_CEA_OUT";AMYG_BA_OUT=b"LMB_BA_OUT"
    HIPPO_THETA=b"LMB_HTHETA";HIPPO_ENCODE=b"LMB_HENCODE";HIPPO_RECALL=b"LMB_HRECALL"
    HIPPO_PLACE=b"LMB_HPLACE";HIPPO_CONTEXT=b"LMB_HCONTEXT";HYPO_STATE=b"LMB_HYPO";VMFPC_REG=b"LMB_VMFPC"
    CTX_PFC_CMD=b"CTX_PFC_CMD";CTX_ASSOC=b"CTX_ASSOC";CTX_ATTENTION=b"CTX_ATTN";CTX_PREDICTION=b"CTX_PRED"
    INSULA_STATE=b"INS_STATE";INSULA_FATIGUE=b"INS_FATIGUE";INSULA_PAIN_AFF=b"INS_PAIN";REST_REPAIR=b"SYS_REST"
    PARIETAL_SPATIAL=b"PAR_SPATIAL";PARIETAL_TOOL=b"PAR_TOOL";PARIETAL_BODY=b"PAR_BODY"
    PARIETAL_PERISP=b"PAR_PERISP";PARIETAL_ATTN=b"PAR_ATTN"
    ACC_ERROR=b"CNG_ERROR";ACC_CONFLICT=b"CNG_CONFLICT";ACC_PAIN_AFF=b"CNG_PAIN"
    PCC_DMN=b"CNG_DMN";PCC_EPISODIC=b"CNG_EPISODIC"
    # LLM (v5900)
    CTX_LLM_RESP=b"CTX_LLM_RESP"   # LLM reasoning response
    CTX_LLM_STATS=b"CTX_LLM_STATS" # LLM performance stats
    CTX_LLM_MODE=b"CTX_LLM_MODE"   # current quantization mode
    THAL_SENSORY=b"THL_SENS";THAL_MOTOR=b"THL_MOTOR";THAL_HB=b"THL_HB";THAL_FAILOVER=b"THL_FAIL"
    SC_SACCADE=b"BS_SC_SACC";SC_PURSUIT=b"BS_SC_PURS";RF_AROUSAL=b"BS_RF_AROUS";MLR_LOCO=b"BS_MLR_LOCO"
    EFF_M1_ARM_L=b"EFF_M1_AL";EFF_M1_ARM_R=b"EFF_M1_AR";EFF_M1_LEG_L=b"EFF_M1_LL";EFF_M1_LEG_R=b"EFF_M1_LR"
    EFF_M1_NECK=b"EFF_M1_NK";EFF_SPEECH=b"EFF_SPEECH";EFF_EYE_L=b"EFF_EYE_L";EFF_EYE_R=b"EFF_EYE_R"
    # VLAN 30 — Neuromodulators
    DA_VTA=b"NM_DA_VTA";NE_LC=b"NM_NE_LC";SERO_RAPHE=b"NM_5HT_RAP";ACH_NBM=b"NM_ACH_NBM"
    # VLAN 40 — System
    SYS_EMERGENCY=b"SYS_EMERGENCY";SYS_CIRCADIAN=b"SYS_CIRCADIAN";SYS_REWARD=b"SYS_REWARD"
    SYS_PTP_SYNC=b"SYS_PTP";LTM_CONSOLIDATE=b"LTM_CONSOLIDATE";LTM_STATS=b"LTM_STATS";LTM_PRUNE=b"LTM_PRUNE"
    ANS_SYMPATH=b"ANS_SYMP";ANS_PARASYMPATH=b"ANS_PARA"


    # v6000 motor cortex split
    PM_MOTOR_PLAN = b"CTX_PM_PLAN"   # premotor plan → M1
    EFF_HAND_L    = b"EFF_HAND_L"    # M1 → left Omnihand
    EFF_HAND_R    = b"EFF_HAND_R"    # M1 → right Omnihand
    # v6000 homunculus
    S1_BODY_MAP   = b"S1_BODY_MAP"   # homunculus body activity map
    VLAN_MAP={20:[b"AFF_",b"SPN_",b"CRB_",b"SFY_",b"VOR_",b"RFX_"],
              10:[b"LMB_",b"CTX_",b"BRC_",b"INS_",b"PAR_",b"CNG_",b"SOC_",b"THL_",b"BS_",b"EFF_"],
              30:[b"NM_"],40:[b"SYS_",b"LTM_",b"ANS_"]}
    @classmethod
    def vlan(cls,topic):
        for vid,pfxs in cls.VLAN_MAP.items():
            for p in pfxs:
                if topic.startswith(p): return vid
        return 40

@dataclass
class NeuralMessage:
    topic:str;timestamp_ms:float;timestamp_ns:int;source:str;target:str
    payload:dict;phase:float=0.0;neuromod:dict=None;vlan:int=40
    def __post_init__(self):
        if self.neuromod is None: self.neuromod={"DA":0.5,"NE":0.2,"5HT":0.5,"ACh":0.5}
        if self.timestamp_ns==0: self.timestamp_ns=time.time_ns()
    def serialize(self): return json.dumps(asdict(self)).encode()
    @classmethod
    def deserialize(cls,data): return cls(**json.loads(data.decode()))
    @property
    def age_ms(self): return (time.time_ns()-self.timestamp_ns)/1e6

class NeuralBus:
    def __init__(self,source,pub_port,sub_endpoints,hwm=1000):
        self.source=source;self._ctx=zmq.Context()
        self._handlers={};self._running=False
        self._nm={"DA":0.5,"NE":0.2,"5HT":0.5,"ACh":0.5};self._phase=0.0
        self.pub=self._ctx.socket(zmq.PUB);self.pub.setsockopt(zmq.SNDHWM,hwm)
        self.pub.setsockopt(zmq.SNDBUF,8*1024*1024);self.pub.bind(f"tcp://*:{pub_port}")
        self.sub=self._ctx.socket(zmq.SUB);self.sub.setsockopt(zmq.RCVHWM,hwm)
        self.sub.setsockopt(zmq.RCVBUF,8*1024*1024);self.sub.setsockopt(zmq.RCVTIMEO,50)
        for ep in sub_endpoints: self.sub.connect(ep)
    def subscribe(self,topic,handler):
        self.sub.setsockopt(zmq.SUBSCRIBE,topic);self._handlers.setdefault(topic,[]).append(handler)
    def publish(self,topic,payload,target="broadcast",phase=None):
        now=time.time_ns()
        msg=NeuralMessage(topic=topic.decode(),timestamp_ms=now/1e6,timestamp_ns=now,
            source=self.source,target=target,payload=payload,
            phase=phase if phase is not None else self._phase,
            neuromod=dict(self._nm),vlan=T.vlan(topic))
        self.pub.send_multipart([topic,msg.serialize()])
    def set_neuromod(self,**kw):
        for k,v in kw.items():
            if k in self._nm: self._nm[k]=float(np.clip(v,0,1))
    def set_phase(self,p): self._phase=float(p%(2*np.pi))
    def start(self):
        self._running=True;threading.Thread(target=self._recv_loop,daemon=True).start()
    def stop(self):
        self._running=False;self.pub.close();self.sub.close();self._ctx.term()
    def _recv_loop(self):
        while self._running:
            try: raw=self.sub.recv_multipart()
            except zmq.Again: continue
            if len(raw)!=2: continue
            tb,data=raw
            try: msg=NeuralMessage.deserialize(data)
            except: continue
            for reg,handlers in self._handlers.items():
                if reg==b"" or tb.startswith(reg):
                    for h in handlers:
                        try: h(msg)
                        except Exception as e: logger.error(f"Handler [{tb}]: {e}")
