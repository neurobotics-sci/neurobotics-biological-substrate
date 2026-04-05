"""bubo/shared/neuromodulators/neuromod_system.py — v10.17"""
import numpy as np, time
from dataclasses import dataclass


@dataclass
class NeuromodState:
    DA: float = 0.50; NE: float = 0.20; sero: float = 0.50; ACh: float = 0.50
    def as_dict(self): return {"DA":self.DA,"NE":self.NE,"5HT":self.sero,"ACh":self.ACh}


class DopamineVTA:
    def __init__(self, gamma=0.95, alpha=0.05):
        self.gamma=gamma; self.alpha=alpha; self._V=0.0; self._tonic=0.5
    def prediction_error(self, reward, next_value):
        delta = reward + self.gamma*next_value - self._V
        self._V += self.alpha*delta
        return float(np.clip(self._tonic + 0.3*delta, 0.0, 1.0))
    def update_tonic(self, avg_reward):
        self._tonic = float(np.clip(0.99*self._tonic + 0.01*(0.3+0.4*avg_reward), 0.1, 0.9))
    @property
    def level(self): return self._tonic


class NoradrenalineLC:
    def __init__(self): self._tonic=0.20; self._phasic=0.0; self._t=time.time()
    def novelty_response(self, novelty):
        self._phasic = float(np.clip(novelty*0.8, 0, 0.6)); return self.level
    def threat_response(self, fear):
        self._tonic = float(np.clip(self._tonic+0.1*fear, 0.1, 0.9)); return self.level
    def update(self, dt):
        self._phasic = max(0.0, self._phasic-0.5*dt)
        self._tonic = 0.998*self._tonic + 0.002*0.2
    @property
    def level(self): return float(np.clip(self._tonic+self._phasic, 0, 1))


class SerotoninRaphe:
    def __init__(self): self._level=0.5
    def punishment_signal(self, p): self._level=float(np.clip(self._level-0.05*p, 0.1, 0.9))
    def reward_signal(self, r):    self._level=float(np.clip(self._level+0.02*r, 0.1, 0.9))
    def homeostatic_restore(self, dt): self._level=float(0.999*self._level+0.001*0.5)
    @property
    def level(self): return self._level


class AcetylcholineNBM:
    def __init__(self): self._level=0.5; self._phase=0.0
    def attention_demand(self, d): self._level=float(np.clip(self._level+0.1*d, 0.1, 0.95))
    def idle(self, dt):
        circ=0.4+0.3*np.cos(self._phase); self._level=float(0.99*self._level+0.01*circ)
    @property
    def level(self): return self._level


class NeuromodulatorSystem:
    def __init__(self):
        self.DA=DopamineVTA(); self.NE=NoradrenalineLC()
        self.sero=SerotoninRaphe(); self.ACh=AcetylcholineNBM()
        self._t=time.time()
    def step(self, reward=0.0, novelty=0.0, fear=0.0,
             attention=0.5, punishment=0.0) -> NeuromodState:
        dt=max(time.time()-self._t,0.001); self._t=time.time()
        da=self.DA.prediction_error(reward, self.DA.level)
        ne=self.NE.novelty_response(novelty)
        if fear>0.3: ne=self.NE.threat_response(fear)
        self.NE.update(dt)
        if punishment>0: self.sero.punishment_signal(punishment)
        if reward>0: self.sero.reward_signal(reward)
        self.sero.homeostatic_restore(dt)
        if attention>0.6: self.ACh.attention_demand(attention-0.6)
        else: self.ACh.idle(dt)
        return NeuromodState(DA=da,NE=ne,sero=self.sero.level,ACh=self.ACh.level)
    @property
    def state(self): return NeuromodState(DA=self.DA.level,NE=self.NE.level,sero=self.sero.level,ACh=self.ACh.level)
