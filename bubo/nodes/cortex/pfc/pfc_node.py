"""
bubo/nodes/cortex/pfc/pfc_node.py — v10.17
Prefrontal Cortex — Orin Nano 8GB (L: 192.168.1.10, R: 192.168.1.11)

v10.17: Integrated with DA personality system (exploration_bonus, bg_temperature),
Insula body-feeling modulation (fatigue → motor inhibit), Broca coordination.
"""
import time, json, logging, threading
import numpy as np
from collections import deque
from bubo.shared.bus.neural_bus import NeuralBus, NeuralMessage, T
from bubo.shared.neuromodulators.neuromod_system import NeuromodState
from bubo.shared.oscillators.neural_oscillators import NeuralOscillatorBank
from bubo.shared.plasticity.synaptic_plasticity import HebbianOja, BCMRule

logger = logging.getLogger("PFC")

try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False


ACTIONS = [
    "reach_left","reach_right","step_forward","step_back",
    "turn_head","grasp","release","stand_still",
    "look_left","look_right","speak","withdraw",
    "crouch","jump","balance_adj","explore","seek_charge","rest_posture",
]
N = len(ACTIONS)


class WorkingMemory:
    def __init__(self, dim=64, capacity=7):
        rng=np.random.default_rng(1)
        self._W=rng.standard_normal((dim,dim))*0.1; np.fill_diagonal(self._W,0.3)
        self._act=np.zeros(dim); self._dim=dim; self._cap=capacity; self._tau=0.1
    def update(self, inp, dt, ach, cortisol=0.15):
        gain=0.5+ach*(1.0-0.5*min(cortisol,0.8))
        drive=gain*(self._W@self._act+np.resize(inp,self._dim))
        self._act=np.clip(self._act+(-self._act+np.tanh(drive))*dt/self._tau,-1,1)
        eff_cap=max(2,int(self._cap*(1-0.4*max(cortisol-0.4,0))))
        return self._act.copy(), eff_cap


class BasalGangliaInterface:
    """PFC→BG interface: sends salience + personality temperature."""
    def __init__(self):
        self._d1=np.ones(N)*0.5; self._d2=np.ones(N)*0.5; self._lr=0.02

    def select(self, salience, nm: NeuromodState, bg_temp=0.30,
               hunger=0.0, thermal=0.0, fatigue=0.0) -> dict:
        da=nm.DA
        sal=salience.copy()
        sal[ACTIONS.index("seek_charge")]+=hunger*0.8
        sal[ACTIONS.index("rest_posture")]+=fatigue*0.6
        for a in ["step_forward","step_back","reach_left","reach_right"]:
            sal[ACTIONS.index(a)]*=(1.0-0.6*thermal)
        direct=sal*self._d1*(0.4+1.2*da)
        ind=sal*self._d2*(1.4-1.0*da)
        gpe=np.clip(1.0-ind*0.8,0.1,1.0)
        stn=np.clip(0.2/(gpe+0.1),0.1,1.0)*( 0.5+nm.NE*0.8)
        gpi=np.clip(ind*0.6+stn-direct,0,2)
        thal=np.clip(1.0-gpi,0,1)
        T_use=max(bg_temp, 0.10)
        exp_t=np.exp(thal/T_use); prob=exp_t/exp_t.sum()
        wi=int(np.argmax(prob))
        return {"action":ACTIONS[wi],"action_idx":wi,"certainty":float(prob[wi]),"probabilities":prob.tolist()}

    def td_update(self, ai, rpe, nm):
        m=abs(nm.DA-0.5)*2*self._lr*rpe
        self._d1[ai]=float(np.clip(self._d1[ai]+m,0.05,0.95))
        self._d2[ai]=float(np.clip(self._d2[ai]-m*0.8,0.05,0.95))


class vmPFCRegulation:
    def __init__(self): self._ext=0.0; self._safety=0.0; self._lr=0.015
    def step(self, fear, sero, slam_ctx=None, fear_rep=False):
        if fear_rep and fear<0.3: self._ext=float(np.clip(self._ext+self._lr*(0.5+sero),0,1))
        if fear<0.1: self._safety=float(np.clip(self._safety+0.005*sero,0,0.8))
        regulation=float(np.clip(self._ext*(0.6+sero*0.4)+self._safety*0.5,0,1))
        return {"vmPFC_signal":regulation,"extinction":self._ext,"safety":self._safety}


# ── v5550: Absorbed parietal PPS + cingulate conflict (Orin nodes removed) ─
class PeakPeriPersonalSpace:
    """Minimal PPS from depth stats — replaces parietal node."""
    THRESHOLD_M = 0.50
    def evaluate(self, depth_stats: dict) -> dict:
        nearest = depth_stats.get("min_m", 5.0)
        threat  = float(max(0.0, 1.0 - nearest / self.THRESHOLD_M)) if nearest < self.THRESHOLD_M else 0.0
        return {"nearest_m": nearest, "threat": threat, "alert": threat > 0.3}

class ConflictMonitor:
    """Minimal ACC conflict detection — replaces cingulate node."""
    CONFLICT_THRESH = 0.15
    def evaluate(self, probs: list, rpe: float) -> dict:
        if len(probs) < 2: return {"conflict": 0.0, "error": abs(rpe)}
        top2 = sorted(probs, reverse=True)[:2]
        conflict = float(1.0 - (top2[0] - top2[1]))
        return {"conflict": conflict, "error": float(abs(rpe)),
                "high_conflict": (top2[0] - top2[1]) < self.CONFLICT_THRESH}


class PFCNode:
    HZ=50
    def __init__(self, side, config):
        assert side in ("L","R"); self.side=side; self.name=f"PFC_{side}"
        self.bus=NeuralBus(self.name,config["pub_port"],config["sub_endpoints"])
        self.wm=WorkingMemory(dim=64); self.bg=BasalGangliaInterface()
        self.vmPFC=vmPFCRegulation(); self.osc=NeuralOscillatorBank()
        self._nm=NeuromodState()
        self._fear=0.0; self._attention=0.5; self._sensory=np.zeros(64)
        self._slam_ctx=None; self._hunger=0.0; self._thermal=0.0
        self._cortisol=0.15; self._fatigue=0.0; self._motor_inhibit=0.0
        self._bg_temp=0.30; self._explore_bonus=0.30; self._da_mode="normal"
        self._insula_body_feel=0.0; self._resting=False
        self._lock=threading.Lock(); self._running=False; self._t_last=time.time()

    def _on_sensory(self, msg):
        f=msg.payload.get("features",[]); arr=np.array(f if isinstance(f,list) else [f])
        with self._lock: self._sensory=0.8*self._sensory+0.2*np.resize(arr,64)

    def _on_cea(self, msg):
        with self._lock: self._fear=float(msg.payload.get("cea_activation",0))*self._cortisol*1.5

    def _on_da(self, msg):
        p=msg.payload
        with self._lock:
            self._nm.DA=float(p.get("DA",0.6))
            self._bg_temp=float(p.get("bg_temperature",0.30))
            self._explore_bonus=float(p.get("exploration_bonus",0.30))
            self._da_mode=p.get("mode","normal")
            self._hunger=float(p.get("hunger",0)); self._thermal=float(p.get("thermal",0))

    def _on_neuromod(self, msg):
        p=msg.payload
        with self._lock:
            if "NE"  in p: self._nm.NE  =float(p["NE"])
            if "5HT" in p: self._nm.sero=float(p["5HT"])
            if "ACh" in p: self._nm.ACh =float(p["ACh"])

    def _on_hypo(self, msg):
        p=msg.payload
        with self._lock:
            self._cortisol=float(p.get("cortisol",0.15))
            self._hunger=float(p.get("hunger",0)); self._thermal=float(p.get("thermal",0))
            self._motor_inhibit=float(p.get("motor_inhibit",0))

    def _on_insula(self, msg):
        with self._lock:
            self._fatigue=float(msg.payload.get("global_fatigue",0))
            self._insula_body_feel=float(msg.payload.get("body_feeling",0))
            self._resting=bool(msg.payload.get("is_resting",False))

    def _on_rest_repair(self, msg):
        with self._lock: self._resting=True

    def _on_hippo(self, msg):
        slam=msg.payload.get("slam_pose") or msg.payload.get("context_vector")
        if slam: self._slam_ctx=np.array(slam[:8],dtype=float)

    def _on_reward(self, msg):
        rpe=float(msg.payload.get("rpe",0)); ai=int(msg.payload.get("action_idx",0))
        self.bg.td_update(ai,rpe,self._nm)

    def _salience(self) -> np.ndarray:
        s=np.ones(N)*0.1; f=self._fear; nm=self._nm
        if f>0.4: s[ACTIONS.index("withdraw")]+=0.7*f; s[ACTIONS.index("stand_still")]+=0.3*f
        if nm.DA>0.6 and f<0.2: s[ACTIONS.index("explore")]+=nm.DA*0.5*self._explore_bonus
        if self._resting: s[ACTIONS.index("rest_posture")]+=0.8
        return np.clip(s,0,1)

    def _motor(self, action, cert):
        s=float(np.clip(cert*(1-self._motor_inhibit),0,1))
        m={"reach_left":{"arm_l":[0.4*s,0,0,0.3*s,0,0,0]},"reach_right":{"arm_r":[0.4*s,0,0,0.3*s,0,0,0]},
           "step_forward":{"gait_mode":"walk","speed_ms":0.5*s},"step_back":{"gait_mode":"walk","speed_ms":-0.3*s},
           "withdraw":{"arm_l":[-0.3*s,0.2*s,0,0.5*s,0,0,0],"arm_r":[-0.3*s,0.2*s,0,0.5*s,0,0,0]},
           "look_left":{"gaze":[-0.3*s,0,1.0]},"look_right":{"gaze":[0.3*s,0,1.0]},
           "rest_posture":{"gait_mode":"stand","motor_scale":0.2},
           "seek_charge":{"gait_mode":"walk","speed_ms":0.3*s,"goal":"charger"},
        }
        return m.get(action,{"gait_mode":"stand"})

    def _loop(self):
        iv=1.0/self.HZ
        while self._running:
            t0=time.time(); dt=t0-self._t_last; self._t_last=t0
            with self._lock:
                nm=self._nm; sensory=self._sensory.copy(); fear=self._fear
                hunger=self._hunger; thermal=self._thermal; cortisol=self._cortisol
                fatigue=self._fatigue; resting=self._resting; bg_temp=self._bg_temp
                slam=self._slam_ctx.copy() if self._slam_ctx is not None else None
            osc=self.osc.step(0.0,nm.ACh,self._attention,0.0)
            self.bus.set_phase(osc["theta"]["phase"])
            wm_act,eff_cap=self.wm.update(sensory,dt,nm.ACh,cortisol)
            fear_rep=fear<0.3 and self._attention>0.3
            vmPFC_out=self.vmPFC.step(fear,nm.sero,slam,fear_rep)
            sal=self._salience()
            bg_out=self.bg.select(sal,nm,bg_temp,hunger,thermal,fatigue)
            motor=self._motor(bg_out["action"],bg_out["certainty"])
            self.bus.publish(T.CTX_PFC_CMD,{
                "action":bg_out["action"],"certainty":bg_out["certainty"],
                "motor":motor,"oscillators":osc,"wm_items":eff_cap,
                "da_mode":self._da_mode,"resting":resting,
                "body_feeling":self._insula_body_feel,
            })
            self.bus.publish(T.VMFPC_REG,{**vmPFC_out,"fear_level":fear})
            self.bus.publish(T.EFFERENCE_COPY,{"motor_command":motor,"timestamp_ms":t0*1000})
            time.sleep(max(0,iv-(time.time()-t0)))

    def start(self):
        self.bus.start()
        for topic,handler in [(T.TOUCH_SA1,self._on_sensory),(T.VISUAL_V1,self._on_sensory),
                               (T.AUDITORY_A1,self._on_sensory),(T.AMYG_CEA_OUT,self._on_cea),
                               (T.DA_VTA,self._on_da),(T.NE_LC,self._on_neuromod),
                               (T.SERO_RAPHE,self._on_neuromod),(T.ACH_NBM,self._on_neuromod),
                               (T.HYPO_STATE,self._on_hypo),(T.INSULA_STATE,self._on_insula),
                               (T.REST_REPAIR,self._on_rest_repair),
                               (T.HIPPO_CONTEXT,self._on_hippo),(T.HIPPO_PLACE,self._on_hippo)]:
            self.bus.subscribe(topic,handler)
        self._running=True
        threading.Thread(target=self._loop,daemon=True).start()
        logger.info(f"{self.name} v10.17 | BG(temp-modulated) | vmPFC | insula-coupled | GPU={HAS_GPU}")
    def stop(self): self._running=False; self.bus.stop()


if __name__=="__main__":
    import sys; side=sys.argv[1] if len(sys.argv)>1 else "L"
    with open("/etc/bubo/config.json") as f: cfg=json.load(f)[f"pfc_{side.lower()}"]
    n=PFCNode(side,cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
