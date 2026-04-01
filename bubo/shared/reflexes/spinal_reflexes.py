"""bubo/shared/reflexes/spinal_reflexes.py — v10.17
Rexed laminar organisation: WDR (V), myotatic, GTO, withdrawal, CPG.
"""
import numpy as np, time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


@dataclass
class ReflexCommand:
    target_muscle: str; joint_id: str; command: float; latency_ms: float; reflex_type: str


class WDRNeuron:
    def __init__(self, windup_tau=2.0): self._windup=0.0; self._tau=windup_tau; self._t=time.time()
    def fire(self, noci, touch):
        dt=time.time()-self._t; self._t=time.time()
        self._windup=np.exp(-dt/self._tau)*self._windup+0.3*noci
        return float(np.clip(touch*0.4+noci*0.8+self._windup, 0, 1))
    @property
    def windup(self): return self._windup


class MyotaticReflex:
    LATENCY_MS=22.0
    def __init__(self, kp=1.2, kd=0.4): self.kp=kp; self.kd=kd
    def compute(self, muscle_id, joint_id, stretch_m, velocity_ms, desired_length_m, gamma=0.5):
        ia_rate=self.kp*(stretch_m-desired_length_m)+self.kd*velocity_ms
        ia_rate*=(0.5+gamma)
        return ReflexCommand(muscle_id,joint_id,float(np.clip(ia_rate,0,1)),self.LATENCY_MS,"myotatic")


class GTOReflex:
    LATENCY_MS=30.0; THRESHOLD_N=50.0
    def compute(self, muscle_id, joint_id, tension_N):
        if tension_N<self.THRESHOLD_N: return None
        inh=float(np.clip((tension_N-self.THRESHOLD_N)/60.0, 0, 1))
        return ReflexCommand(muscle_id,joint_id,-inh,self.LATENCY_MS,"GTO_inhibition")


class FlexorWithdrawalReflex:
    LATENCY_IPSI_MS=65.0; LATENCY_CONTRA_MS=80.0
    IPSI={
        "hand_L":["bicep_L","deltoid_ant_L"],"hand_R":["bicep_R","deltoid_ant_R"],
        "foot_L":["iliopsoas_L","hamstrings_L"],"foot_R":["iliopsoas_R","hamstrings_R"],
        "arm_L":["bicep_L"],"arm_R":["bicep_R"],"leg_L":["iliopsoas_L"],"leg_R":["iliopsoas_R"],
    }
    CONTRA={
        "hand_L":["tricep_R"],"hand_R":["tricep_L"],
        "foot_L":["gluteus_max_R","quadriceps_R"],"foot_R":["gluteus_max_L","quadriceps_L"],
    }
    def compute(self, zone_id, intensity) -> List[ReflexCommand]:
        cmds=[]; region=self._region(zone_id)
        if region is None: return cmds
        for m in self.IPSI.get(region,[]):
            cmds.append(ReflexCommand(m,self._j(m),float(np.clip(intensity,0,1)),self.LATENCY_IPSI_MS,"flexor_withdrawal"))
        for m in self.CONTRA.get(region,[]):
            cmds.append(ReflexCommand(m,self._j(m),float(np.clip(intensity*0.7,0,1)),self.LATENCY_CONTRA_MS,"crossed_extension"))
        return cmds
    def _region(self, z):
        for k in self.IPSI:
            if k.replace("_","").lower() in z.replace("_","").lower(): return k
        return None
    def _j(self, m):
        for k,v in {"bicep":"elbow","tricep":"elbow","deltoid":"shoulder","iliopsoas":"hip",
                    "hamstrings":"knee","quadriceps":"knee","gluteus":"hip"}.items():
            if k in m.lower(): return v
        return "unknown"


class HalfCentreOscillator:
    def __init__(self, tau=0.1):
        self.tau=tau; self._f=0.5; self._e=0.5; self._f_pic=0.2; self._e_pic=0.2; self._tau_pic=2.0
    def step(self, dt, mlr_drive=0.5, sensory_mod=0.0):
        w=1.5+mlr_drive*0.5
        self._f_pic+=dt/self._tau_pic*(mlr_drive*0.4-self._f_pic)
        self._e_pic+=dt/self._tau_pic*(mlr_drive*0.4-self._e_pic)
        self._f+=dt/self.tau*(np.tanh(mlr_drive+self._f_pic-w*self._e+sensory_mod)-self._f)
        self._e+=dt/self.tau*(np.tanh(mlr_drive+self._e_pic-w*self._f-sensory_mod)-self._e)
        return {"flexor":float(np.clip(self._f,0,1)),"extensor":float(np.clip(self._e,0,1))}


class LocomotorCPG:
    def __init__(self):
        self._fl=HalfCentreOscillator(0.08); self._fr=HalfCentreOscillator(0.08)
        self._hl=HalfCentreOscillator(0.10); self._hr=HalfCentreOscillator(0.10)
        self._t=time.time()
    def step(self, mlr_drive=0.5, load_feedback=None):
        dt=max(time.time()-self._t,0.001); self._t=time.time()
        lf=load_feedback or {}
        fl=self._fl.step(dt,mlr_drive,lf.get("fl_load",0.0))
        fr=self._fr.step(dt,mlr_drive,lf.get("fr_load",0.0))
        hl=self._hl.step(dt,mlr_drive,lf.get("hl_load",0.0))
        hr=self._hr.step(dt,mlr_drive,lf.get("hr_load",0.0))
        def to_angles(cpg,limb):
            f,e=cpg["flexor"],cpg["extensor"]
            if limb=="leg": return {"hip_flex_ext":np.radians(30)*(f-e),"knee_flex_ext":np.radians(40)*f,"ankle_df_pf":np.radians(20)*(e-0.3)}
            return {"shoulder_flex_ext":np.radians(20)*(f-e),"elbow_flex_ext":np.radians(15)*f}
        return {"arm_l":to_angles(fl,"arm"),"arm_r":to_angles(fr,"arm"),
                "leg_l":to_angles(hl,"leg"),"leg_r":to_angles(hr,"leg"),
                "mlr_drive":mlr_drive,
                "gait_phase":{"fl_flex":fl["flexor"],"fr_flex":fr["flexor"],"hl_flex":hl["flexor"],"hr_flex":hr["flexor"]}}
