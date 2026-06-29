"""brain/shared/bus/neural_bus.py — VLAN-aware + PTP timestamps"""
import sys,zmq,json,time,threading,logging,numpy as np
from dataclasses import dataclass,asdict
from typing import Callable,Dict,List,Optional

logger=logging.getLogger("NeuralBus")

# MOD 1: Route logging strictly to stdout so it prints cleanly to your console
logging.basicConfig(
    level=logging.DEBUG, 
    stream=sys.stdout, 
    format="%(asctime)s [%(levelname)s] %(message)s"
)

from enum import Enum, unique

@unique
class T(bytes, Enum):
    """
    SBALF Master Message Fabric and Port blueprints.
    Enforces absolute byte-string type safety across all ZeroMQ partitions.
    """
    def __new__(cls, value):
        obj = bytes.__new__(cls, value)
        obj._value_ = value
        return obj

    # 🌐 VLAN 20 — Low-Road (Reflex / Sensorimotor Tier)
    TOUCH_SA1        = b"AFF_TOUCH_SA1"        # Somatosensory tactile
    TOUCH_SA2        = b"AFF_TOUCH_SA2"
    NOCI_HEAT        = b"AFF_NOCI_HEAT"        # Thermal pain
    NOCI_COLD        = b"AFF_NOCI_COLD"
    NOCI_MECH        = b"AFF_NOCI_MECH"        # Mechanical pressure pain
    PROPRIO_IA       = b"AFF_PROP_IA"          # Joint spindle feedback
    PROPRIO_IB       = b"AFF_PROP_IB"
    VISUAL_V1        = b"AFF_VIS_V1"           # V1 early vision features
    VISUAL_MT        = b"AFF_VIS_MT"           # MT optic flow events
    VISUAL_DEPTH     = b"AFF_VIS_DEPTH"        # SGBM stereo point clouds
    VISUAL_FACE      = b"AFF_VIS_FACE"
    AUDITORY_A1      = b"AFF_AUD_A1"           # A1 raw microphone energy
    AUDITORY_SPAT    = b"AFF_AUD_SPAT"         # Spatial audio azimuth
    VESTIBULAR       = b"AFF_VEST"             # 6-DOF IMU fusion
    THERMAL_WARM     = b"AFF_THERM_WARM"
    THERMAL_COOL     = b"AFF_THERM_COOL"
    SPINAL_REFLEX    = b"SPN_REFLEX"
    SPINAL_CPG       = b"SPN_CPG"              # Locomotor pattern generation
    SPINAL_FBK       = b"SPN_FBK"              # Joint telemetry feedback
    SPINAL_HB        = b"SPN_HB"               # 100Hz hardware heartbeat watchdog
    CEREBELL_DELTA   = b"CRB_DELTA"            # CMAC online motor correction
    CLIMBING_FIBRE   = b"CRB_CLIMB"            # Cerebellar error signal
    EFFERENCE_COPY   = b"CRB_EFF_CP"           # PM/SMA copy to M1
    SAFETY_FREEZE    = b"SFY_FREEZE"           # Emergency motor lock
    SAFETY_ZMP_FAIL  = b"SFY_ZMP"              # Zero Moment Point balance failure
    NOD_OFF          = b"SFY_NODOFF"           # Microsleep / sedation onset
    LIMP_MODE_ACTIVE = b"SFY_LIMP"             # Spinal failure fallback drive
    LIMP_MODE_CLEAR  = b"SFY_LIMP_CLR"
    VOR_CMD          = b"VOR_CMD"              # Vestibulo-ocular reflex adjustments
    VOR_EYE_FB       = b"VOR_EYE_FB"
    VOR_SUPPRESS     = b"VOR_SUPP"
    REFLEX_ASR       = b"RFX_ASR"              # Acoustic startle reflex
    REFLEX_BLINK     = b"RFX_BLINK"            # Blinking reflex
    REFLEX_TLR       = b"RFX_TLR"              # Tonic Labyrinthine reflex
    REFLEX_ATNR      = b"RFX_ATNR"             # Asymmetric Tonic Neck reflex
    REFLEX_GRASP     = b"RFX_GRASP"            # Palmar grasp reflex
    REFLEX_MORO      = b"RFX_MORO"             # Vestibular fall detection
    REFLEX_PLR       = b"RFX_PLR"              # Pupillary light reflex
    REFLEX_OKR       = b"RFX_OKR"              # Optokinetic reflex

    # 🧠 VLAN 10 — High-Road (Cognitive / Executive Tier)
    AMYG_LA_OUT      = b"LMB_LA_OUT"           # Lateral Amygdala high-road
    AMYG_CEA_OUT     = b"LMB_CEA_OUT"          # Central Amygdala output
    AMYG_BA_OUT      = b"LMB_BA_OUT"
    HIPPO_THETA      = b"LMB_HTHETA"           # 7Hz hippocampal theta clock
    HIPPO_ENCODE     = b"LMB_HENCODE"          # Epistemic encoding event
    HIPPO_RECALL     = b"LMB_HRECALL"          # Episodic lookup query
    HIPPO_PLACE      = b"LMB_HPLACE"           # 6-DOF RTABMap position
    HIPPO_CONTEXT    = b"LMB_HCONTEXT"         # Reuniens spatial context
    HYPO_STATE       = b"LMB_HYPO"             # Cortisol / metabolic profiles
    VMFPC_REG        = b"LMB_VMFPC"            # Amygdala emotional extinction
    CTX_PFC_CMD      = b"CTX_PFC_CMD"          # Executive goal selection
    CTX_ASSOC        = b"CTX_ASSOC"            # Multi-modal attention mapping
    CTX_ATTENTION    = b"CTX_ATTN"             # Alpha-oscillation gate override
    CTX_PREDICTION   = b"CTX_PRED"
    BROCA_SPEECH_ACT = b"BRC_SPEECH"           # Syntactic speech request
    BROCA_SYNTAX     = b"BRC_SYNTAX"
    BROCA_MOTSEQ     = b"BRC_MOTSEQ"           # Speech motor orchestration
    INSULA_STATE     = b"INS_STATE"            # Interoceptive awareness map
    INSULA_FATIGUE   = b"INS_FATIGUE"          # Arrhenius mechanical tracking
    INSULA_PAIN_AFF  = b"INS_PAIN"             # Somatosensory pain suffering
    REST_REPAIR      = b"SYS_REST"             # DMN sedation state
    PARIETAL_SPATIAL = b"PAR_SPATIAL"          # Egocentric/allocentric grid maps
    PARIETAL_TOOL    = b"PAR_TOOL"             # Grasp affordance allocations
    PARIETAL_BODY    = b"PAR_BODY"             # Proprioceptive body schema
    PARIETAL_PERISP  = b"PAR_PERISP"           # 50cm peripersonal space alert
    PARIETAL_ATTN    = b"PAR_ATTN"
    ACC_ERROR        = b"CNG_ERROR"            # ACC error prediction monitoring
    ACC_CONFLICT     = b"CNG_CONFLICT"         # ACC competing action monitor
    ACC_PAIN_AFF     = b"CNG_PAIN"
    PCC_DMN          = b"CNG_DMN"              # Default Mode Network activation
    PCC_EPISODIC     = b"CNG_EPISODIC"         # Future thinking simulations
    SOCIAL_FACE      = b"SOC_FACE"             # Face ArcFace embedding identification
    SOCIAL_BOND      = b"SOC_BOND"             # Oxytocin attachment level tracking
    SOCIAL_THREAT_MOD= b"SOC_THRMOD"           # Amygdala suppression modifier
    SOCIAL_APPROACH  = b"SOC_APPROACH"         # Incentive approach drive
    SOCIAL_OXY       = b"SOC_OXY"
    THAL_SENSORY     = b"THL_SENS"             # LGN/MGN/VPL sensory routing
    THAL_MOTOR       = b"THL_MOTOR"            # VA/VL motor routing gate
    THAL_HB          = b"THL_HB"               # Thalamic supervisor heartbeat
    THAL_FAILOVER    = b"THL_FAIL"             # Cross-thalamic failover alert
    SC_SACCADE       = b"BS_SC_SACC"           # Balistic foveation trigger
    SC_PURSUIT       = b"BS_SC_PURS"
    RF_AROUSAL       = b"BS_RF_AROUS"          # Reticular formation vigilance drive
    MLR_LOCO         = b"BS_MLR_LOCO"          # Mesencephalic locomotor drive
    EFF_M1_ARM_L     = b"EFF_M1_AL"            # M1 output → Left arm
    EFF_M1_ARM_R     = b"EFF_M1_AR"            # M1 output → Right arm
    EFF_M1_LEG_L     = b"EFF_M1_LL"            # M1 output → Left leg
    EFF_M1_LEG_R     = b"EFF_M1_LR"            # M1 output → Right leg
    EFF_M1_NECK      = b"EFF_M1_NK"
    EFF_SPEECH       = b"EFF_SPEECH"
    EFF_EYE_L        = b"EFF_EYE_L"
    EFF_EYE_R        = b"EFF_EYE_R"
    PM_MOTOR_PLAN    = b"CTX_PM_PLAN"          # Premotor plan to M1
    EFF_HAND_L       = b"EFF_HAND_L"           # M1 to Left Omnihand
    EFF_HAND_R       = b"EFF_HAND_R"           # M1 to Right Omnihand
    S1_BODY_MAP      = b"S1_BODY_MAP"

    # 🤖 VLAN 10 — Large Language Model Interfaces
    CTX_LLM_RESP     = b"CTX_LLM_RESP"         # PFC Language output response
    CTX_LLM_STATS    = b"CTX_LLM_STATS"        # LLM token/latency performance metrics
    CTX_LLM_MODE     = b"CTX_LLM_MODE"         # Active adaptive quantization flag

    # 🧪 VLAN 30 — Neuromodulators (Chemical Substrate Layer)
    DA_VTA           = b"NM_DA_VTA"            # Mesolimbic Tonic/Phasic Dopamine
    NE_LC            = b"NM_NE_LC"             # Locus Coeruleus Noradrenaline
    SERO_RAPHE       = b"NM_5HT_RAP"           # Dorsal Raphe Serotonin
    ACH_NBM          = b"NM_ACH_NBM"           # Nucleus Basalis Acetylcholine

    # ⚙️ VLAN 40 — System Infrastructure Layer
    SYS_EMERGENCY    = b"SYS_EMERGENCY"        # Low-battery / thermal override
    SYS_CIRCADIAN    = b"SYS_CIRCADIAN"        # BMAL1/CLOCK ODE state tracking
    SYS_REWARD       = b"SYS_REWARD"           # Global reinforcement signal
    SYS_PTP_SYNC     = b"SYS_PTP"              # PTP nanosecond synchronization telemetry
    LTM_CONSOLIDATE  = b"LTM_CONSOLIDATE"      # NREM3 memory transfer execution
    LTM_STATS        = b"LTM_STATS"
    LTM_PRUNE        = b"LTM_PRUNE"            # Glial database cleanup execution
    ANS_SYMPATH      = b"ANS_SYMP"             # Autonomic sympathetic metrics
    ANS_PARASYMPATH  = b"ANS_PARA"

    VLAN_MAP = {
        20: [b"AFF_", b"SPN_", b"CRB_", b"SFY_", b"VOR_", b"RFX_"],
        10: [b"LMB_", b"CTX_", b"BRC_", b"INS_", b"PAR_", b"CNG_", b"SOC_", b"THL_", b"BS_", b"EFF_"],
        30: [b"NM_"],
        40: [b"SYS_", b"LTM_", b"ANS_"]
    }

    @classmethod
    def vlan(cls, topic: bytes) -> int:
        for vid, pfxs in cls.VLAN_MAP.value.items():
            for p in pfxs:
                if topic.startswith(p): 
                    return vid
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
