"""
bubo/nodes/subcortical/hypothalamus/hypothalamus_node.py — v5550
Hypothalamus: HPA axis ODE, DA+personality, thermal regulation, vagus nerve.

v5550 additions:
  - ThermalRegulationLoop: polls cluster temps, throttles PFC via cpufreq
  - VagusNerve: monitors cluster health, triggers Category 1 stop
  - DDS Partition "Neuro": broadcasts DA/NE/5HT/ACh to ALL partitions
  - Absorbed thermal monitoring from removed nodes
"""
import time, json, logging, threading
import numpy as np
from pathlib import Path
from bubo.shared.bus.neural_bus import NeuralBus, T
from bubo.shared.neuromodulators.neuromod_system import NeuromodulatorSystem
from bubo.shared.watchdog.node_watchdog import NodeWatchdog
from bubo.thermal.thermal_regulation import ThermalRegulationLoop
from bubo.vagus.vagus_nerve import VagusNerve

logger = logging.getLogger("Hypothalamus")


class HPAAxis:
    """Hypothalamus-Pituitary-Adrenal axis ODE — CRH→ACTH→Cortisol feedback loop."""
    K_CRH=0.8; K_CRH_D=0.15; K_ACTH=0.6; K_ACTH_D=0.25
    K_CORT=0.5; K_CORT_D=0.10; K_FB=0.40
    def __init__(self): self._CRH=0.1; self._ACTH=0.05; self._cort=0.15
    def step(self, stressor, dt):
        fb=self.K_FB*self._cort**2
        self._CRH  = float(np.clip(self._CRH  + (self.K_CRH*stressor - self.K_CRH_D*self._CRH  - fb)*dt, 0,1))
        self._ACTH = float(np.clip(self._ACTH + (self.K_ACTH*self._CRH - self.K_ACTH_D*self._ACTH)*dt, 0,1))
        self._cort = float(np.clip(self._cort + (self.K_CORT*self._ACTH - self.K_CORT_D*self._cort - fb)*dt, 0,1))
        return {"CRH":self._CRH, "ACTH":self._ACTH, "cortisol":self._cort}


def read_battery_frac() -> float:
    for p in ["/sys/class/power_supply/BAT0/charge_now",
              "/sys/class/power_supply/battery/charge_now"]:
        try:
            now  = float(Path(p).read_text())
            full = float(Path(p.replace("charge_now","charge_full")).read_text())
            return float(np.clip(now/max(full,1), 0, 1))
        except: pass
    return 1.0   # simulation: assume full


def read_cpu_temp() -> float:
    try: return float(Path("/sys/class/thermal/thermal_zone0/temp").read_text())/1000.0
    except: return 45.0


def battery_to_personality(bf: float) -> tuple:
    """Map battery fraction → (tonic_DA, explore_bonus, mode_label)."""
    if bf > 0.80: return 0.85, 0.50, "explorer"
    if bf > 0.60: return 0.68, 0.30, "normal"
    if bf > 0.40: return 0.50, 0.10, "conservative"
    if bf > 0.20: return 0.32, 0.00, "cautious"
    if bf > 0.05: return 0.12, 0.00, "survival"
    return 0.02, 0.00, "emergency"


class HypothalamusNode:
    HZ = 10

    def __init__(self, config: dict):
        self.name = "Hypothalamus"
        self.bus  = NeuralBus(self.name, config["pub_port"], config["sub_endpoints"])
        self.hpa  = HPAAxis()
        self.nm   = NeuromodulatorSystem()
        self.wd   = NodeWatchdog(self.name, self.bus)

        # v5550: thermal regulation loop + vagus nerve
        self.thermal = ThermalRegulationLoop(self.bus)
        self.vagus   = VagusNerve(self.bus, sim_mode=not Path("/sys/class/gpio").exists())

        self._fear    = 0.0
        self._circ_da = 0.0
        self._running = False
        self._lock    = threading.Lock()

    def _on_cea(self, msg):
        with self._lock: self._fear = float(msg.payload.get("cea_activation", 0))

    def _on_circadian(self, msg):
        with self._lock: self._circ_da = float(msg.payload.get("da_modulation", 0))

    def _on_emergency(self, msg):
        """Escalate severe emergencies to vagus nerve."""
        etype = msg.payload.get("type","")
        if etype in ("thermal_shutdown","critical_failure","hallucination_detected"):
            self.vagus.fire(f"emergency_{etype}")

    def _loop(self):
        interval = 1.0 / self.HZ
        while self._running:
            t0 = time.time()
            self.wd.heartbeat()
            self.vagus.heartbeat()

            bf      = read_battery_frac()
            cpu_C   = read_cpu_temp()
            thermal = self.thermal

            with self._lock:
                fear    = self._fear
                circ_da = self._circ_da

            tonic_DA, explore_bonus, mode = battery_to_personality(bf)
            tonic_DA = float(np.clip(tonic_DA + circ_da - 0.3*fear, 0.01, 1.0))

            # Use thermal loop's motor_inhibit
            motor_inhibit  = thermal.motor_inhibit
            pub_rate_scale = thermal.pub_scale
            thermal_stress = float(np.clip((cpu_C - 55)/30, 0, 1))
            hunger         = float(np.clip(1.0 - bf*1.2, 0, 1))
            stressor       = float(np.clip(0.3*thermal_stress + 0.25*hunger + 0.3*fear, 0, 1))
            hpa            = self.hpa.step(stressor, dt=1.0/self.HZ)
            nm             = self.nm.step(fear=fear, attention=0.5)

            bg_temp      = 0.30 * (1.0 + explore_bonus)
            wm_scale     = float(np.clip(1.0 - 0.5*max(hpa["cortisol"]-0.4,0), 0.3, 1.0))

            # Broadcast to ALL DDS partitions (Neuro partition)
            now_ns = time.time_ns()
            self.bus.publish(T.DA_VTA, {
                "DA": tonic_DA, "tonic_DA": tonic_DA, "mode": mode,
                "bg_temperature": bg_temp, "exploration_bonus": explore_bonus,
                "hunger": hunger, "thermal": thermal_stress,
                "battery_frac": bf, "timestamp_ns": now_ns,
            })
            self.bus.publish(T.NE_LC,      {"NE": nm.NE, "timestamp_ns": now_ns})
            self.bus.publish(T.SERO_RAPHE, {"5HT": nm.sero, "timestamp_ns": now_ns})
            self.bus.publish(T.ACH_NBM,    {"ACh": nm.ACh, "timestamp_ns": now_ns})
            self.bus.publish(T.HYPO_STATE, {
                "cortisol":      hpa["cortisol"], "CRH": hpa["CRH"], "ACTH": hpa["ACTH"],
                "hunger":        hunger, "thermal": thermal_stress,
                "motor_inhibit": motor_inhibit, "pub_rate_scale": pub_rate_scale,
                "battery_frac":  bf, "cpu_temp_C": cpu_C, "mode": mode,
                "thermal_level": thermal.level,
                "wm_capacity_scale": wm_scale,
                "vagus_state":   self.vagus.state.value,
                "exploration_bias": float(0.3*hunger),
                "timestamp_ns":  now_ns,
            })
            if mode == "emergency":
                self.bus.publish(T.SYS_EMERGENCY, {
                    "type": "low_battery", "battery_frac": bf, "timestamp_ns": now_ns})

            self.wd.record_publish()
            time.sleep(max(0, interval - (time.time() - t0)))

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.AMYG_CEA_OUT,  self._on_cea)
        self.bus.subscribe(T.SYS_CIRCADIAN, self._on_circadian)
        self.bus.subscribe(T.SYS_EMERGENCY, self._on_emergency)
        self.thermal.start()
        self.vagus.start()
        self.wd.start()
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info(f"{self.name} v5550 | HPA+DA | thermal loop | vagus nerve | {self.HZ}Hz")

    def stop(self):
        self._running = False
        self.thermal.stop()
        self.vagus.stop()
        self.bus.stop()


if __name__ == "__main__":
    with open("/etc/bubo/config.json") as f: cfg = json.load(f)["hypothalamus"]
    n = HypothalamusNode(cfg)
    n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        n.stop()
