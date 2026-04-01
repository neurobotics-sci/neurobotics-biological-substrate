"""bubo/shared/plasticity/synaptic_plasticity.py — v10.17"""
import numpy as np, time
from dataclasses import dataclass, field


class HebbianOja:
    def __init__(self, eta=0.01): self.eta=eta
    def update(self, w, pre, post, ach=0.5, da_pe=0.0):
        eff=self.eta*ach*(1.0+da_pe)
        dw=eff*np.outer(post, pre-post@w.T if w.ndim==2 else pre)
        return np.clip(w+dw, 0, 1)


class STDP:
    def __init__(self, A_plus=0.01, A_minus=0.012, tau_plus=0.020, tau_minus=0.020):
        self.A_plus=A_plus; self.A_minus=A_minus; self.tau_p=tau_plus; self.tau_m=tau_minus
    def update(self, w, delta_t, ne=0.2, da=0.5):
        ne_gain=1.0+1.5*ne
        if delta_t>0: dw= self.A_plus *np.exp(-delta_t/self.tau_p)*ne_gain*(0.5+da)
        else:         dw=-self.A_minus*np.exp( delta_t/self.tau_m)*ne_gain
        return float(np.clip(w+dw, 0, 1))


class BCMRule:
    def __init__(self, tau_theta=1.0, eta=0.005):
        self._theta=0.5; self.tau=tau_theta; self.eta=eta
    def update(self, w, pre, post, dt, ach=0.5):
        eta_eff=self.eta*ach
        phi=post*(post-self._theta)
        dw=eta_eff*np.outer(phi,pre)
        self._theta+=dt/self.tau*(np.mean(post**2)-self._theta)
        self._theta=float(np.clip(self._theta,0.05,0.95))
        return np.clip(w+dw, 0, 1)
    @property
    def theta(self): return self._theta


class CerebellarLTD:
    def __init__(self, eta_ltd=0.02, eta_ltp=0.002):
        self.eta_ltd=eta_ltd; self.eta_ltp=eta_ltp
    def update(self, w, granule_activity, climbing_fibre_active):
        if climbing_fibre_active: dw=-self.eta_ltd*granule_activity
        else:                     dw= self.eta_ltp*(1.0-w)
        return np.clip(w+dw, 0.01, 1.0)


class SynapticTagging:
    def __init__(self, tag_decay=3600.0):
        self._tags={}; self.tag_decay=tag_decay
    def set_tag(self, mid, strength): self._tags[mid]=(float(strength),time.time())
    def capture(self, ne_level, da_level):
        now=time.time(); threshold=0.6; enhanced=[]
        if ne_level+da_level<threshold: return enhanced
        boost=(ne_level+da_level-threshold)*2.0
        for mid,(s,ts) in list(self._tags.items()):
            decay=np.exp(-(now-ts)/self.tag_decay)
            if s*decay>0.1: enhanced.append({"memory_id":mid,"boost":float(boost*s*decay)})
            else: del self._tags[mid]
        return enhanced
