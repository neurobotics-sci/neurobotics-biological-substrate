"""bubo/shared/oscillators/neural_oscillators.py — v10.17"""
import numpy as np, time
from dataclasses import dataclass

@dataclass
class OscillatorState:
    phase: float; frequency: float; amplitude: float; band: str


class HippocampalTheta:
    def __init__(self, base_freq=7.0):
        self.base_freq=base_freq; self._phase=0.0; self._amplitude=0.6; self._t=time.time()
    def step(self, dt, speed_ms=0.0, ach_level=0.5) -> OscillatorState:
        freq=self.base_freq+2.0*min(speed_ms/2.0,1.0)
        self._amplitude=float(np.clip(0.3+0.7*ach_level,0.1,1.0))
        self._phase=(self._phase+2*np.pi*freq*dt)%(2*np.pi)
        return OscillatorState(self._phase,freq,self._amplitude,"theta")
    @property
    def encoding_gate(self): return 0<self._phase<np.pi
    @property
    def phase(self): return self._phase


class GammaOscillator:
    def __init__(self, freq=40.0): self.freq=freq; self._phase=0.0; self._amplitude=0.5
    def step(self, dt, theta_phase=0.0, attention=0.5) -> OscillatorState:
        theta_mod=0.5+0.5*np.cos(theta_phase-3*np.pi/2)
        self._amplitude=float(np.clip(attention*theta_mod,0.05,1.0))
        self._phase=(self._phase+2*np.pi*self.freq*dt)%(2*np.pi)
        return OscillatorState(self._phase,self.freq,self._amplitude,"gamma")


class BetaOscillator:
    def __init__(self, freq=20.0): self.freq=freq; self._phase=0.0; self._amplitude=0.7
    def step(self, dt, movement_level=0.0) -> OscillatorState:
        self._amplitude=float(np.clip(0.8-0.7*movement_level,0.05,0.85))
        self._phase=(self._phase+2*np.pi*self.freq*dt)%(2*np.pi)
        return OscillatorState(self._phase,self.freq,self._amplitude,"beta")
    @property
    def movement_gated(self): return self._amplitude<0.3


class AlphaThalamic:
    def __init__(self, freq=10.0): self.freq=freq; self._phase=0.0; self._amplitude=0.8
    def step(self, dt, attention_load=0.0) -> OscillatorState:
        self._amplitude=float(np.clip(0.9-0.8*attention_load,0.05,0.95))
        self._phase=(self._phase+2*np.pi*self.freq*dt)%(2*np.pi)
        return OscillatorState(self._phase,self.freq,self._amplitude,"alpha")
    @property
    def sensory_gate(self): return float(1.0-self._amplitude)


class NeuralOscillatorBank:
    def __init__(self):
        self.theta=HippocampalTheta(); self.gamma=GammaOscillator()
        self.beta=BetaOscillator();   self.alpha=AlphaThalamic()
        self._t=time.time()
    def step(self, speed_ms=0.0, ach_level=0.5, attention=0.5, movement=0.0) -> dict:
        dt=max(time.time()-self._t,0.001); self._t=time.time()
        ts=self.theta.step(dt,speed_ms,ach_level)
        gs=self.gamma.step(dt,ts.phase,attention)
        bs=self.beta.step(dt,movement)
        als=self.alpha.step(dt,attention)
        return {
            "theta":{"phase":ts.phase,"amp":ts.amplitude,"freq":ts.frequency,"encoding_gate":self.theta.encoding_gate},
            "gamma":{"phase":gs.phase,"amp":gs.amplitude},
            "beta": {"phase":bs.phase,"amp":bs.amplitude,"movement_gated":self.beta.movement_gated},
            "alpha":{"phase":als.phase,"amp":als.amplitude,"sensory_gate":self.alpha.sensory_gate},
        }
