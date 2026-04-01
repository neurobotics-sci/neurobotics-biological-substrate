"""
bubo/thermal/thermal_regulation.py — Bubo v5550
Hypothalamus Thermal Regulation Loop

═══════════════════════════════════════════════════════════════════════════════
BIOLOGICAL ANALOGY
═══════════════════════════════════════════════════════════════════════════════

The hypothalamus is the thermoregulatory centre of the mammalian brain:
- Preoptic area (POA): warm-sensitive neurons fire when brain temp rises
- Dorsomedial nucleus (DMH): coordinates thermoeffectors
- Responses: vasodilation, sweating, panting (heat dissipation)
  OR vasoconstriction, shivering, huddling (heat conservation)

The key feature: the brain will SACRIFICE COGNITIVE PERFORMANCE to maintain
safe temperature. High fever → confusion → delirium → coma (in that order).
This is not a bug — it's adaptive: a hot brain that stops thinking correctly
is better than a brain that burns out.

Bubo's hypothalamus implements this same priority: thermal safety > cognition.

═══════════════════════════════════════════════════════════════════════════════
THERMAL HIERARCHY
═══════════════════════════════════════════════════════════════════════════════

Temperature thresholds (Jetson hardware):
  ≤55°C  NOMINAL:   Normal operation, no throttling
  56-65°C WARM:     Begin throttling PFC nodes (reduce pub rate)
  66-74°C HOT:      Throttle PFC + Association + Social nodes
                    Switch cpufreq governor to powersave
  75-82°C HIGH:     Motor inhibit signal — reduce servo activity
                    Disable GPU inference on Social node
  83-87°C CRITICAL: Emergency stop all motor activity
                    Begin shutdown of non-essential nodes
  >87°C  SHUTDOWN:  Graceful cluster shutdown, vagus nerve trigger

THROTTLING MECHANISM:
  1. cpufreq scaling_governor → "powersave" on target node via SSH
  2. T.HYPO_STATE motor_inhibit → spinal nodes reduce torque
  3. T.HYPO_STATE pub_rate_scale → nodes reduce their publication frequency
  4. Direct SSH: `cpufreq-set -g powersave` on remote node

WHICH NODES TO THROTTLE AND WHY:
  PFC-L/R:       Highest GPU users, non-time-critical computation
                 BG action selection can run at 10Hz instead of 50Hz safely
  Social:        Face recognition is GPU-intensive and discretionary
  Association:   Cross-modal binding can tolerate higher latency
  LTM:           Background consolidation can pause entirely
  Cerebellum:    CANNOT throttle — 100Hz motor control is safety-critical
  Spinal:        CANNOT throttle — real-time servo loop
  SC/Thalamus:   CANNOT throttle — sensory gating is safety-critical
"""

import time, subprocess, logging, threading
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger("ThermalRegulation")

SSH_KEY = Path.home() / ".ssh/bubo_id_ed25519"
SSH_OPTS = f"-o StrictHostKeyChecking=no -o ConnectTimeout=3 -i {SSH_KEY}"

# Thermal thresholds (°C)
T_NOMINAL   = 55.0
T_WARM      = 60.0
T_HOT       = 70.0
T_HIGH      = 78.0
T_CRITICAL  = 83.0
T_SHUTDOWN  = 87.0

# Nodes eligible for thermal throttling (safety-critical nodes excluded)
THROTTLEABLE_NODES = {
    "pfc-l":       {"ip": "192.168.1.10", "priority": 1},
    "pfc-r":       {"ip": "192.168.1.11", "priority": 1},
    "social":      {"ip": "192.168.1.19", "priority": 2},
    "association": {"ip": "192.168.1.34", "priority": 3},
    "ltm-store":   {"ip": "192.168.1.35", "priority": 4},
    "broca":       {"ip": "192.168.1.14", "priority": 5},
    "hippocampus": {"ip": "192.168.1.30", "priority": 6},
}

# Safety-critical — never throttle
CRITICAL_NODES = ["cerebellum", "spinal-arms", "spinal-legs",
                  "sup-colliculus", "thalamus-l", "thalamus-r"]


@dataclass
class ThermalState:
    node:        str
    temp_C:      float
    level:       str   # nominal/warm/hot/high/critical/shutdown
    throttled:   bool
    governor:    str   # performance/powersave
    pub_scale:   float  # 0.0-1.0 publication rate scale
    motor_inhibit: float  # 0.0-1.0


def classify_temp(temp_C: float) -> str:
    if temp_C <= T_NOMINAL:  return "nominal"
    if temp_C <= T_WARM:     return "warm"
    if temp_C <= T_HOT:      return "hot"
    if temp_C <= T_HIGH:     return "high"
    if temp_C <= T_CRITICAL: return "critical"
    return "shutdown"


def ssh_cmd(ip: str, cmd: str, timeout: int = 5) -> tuple:
    try:
        r = subprocess.run(
            f"ssh {SSH_OPTS} brain@{ip} '{cmd}'",
            shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip()
    except Exception as e:
        return 1, str(e)


def read_node_temp(ip: str) -> float:
    """Read CPU temperature from a remote Jetson node."""
    rc, out = ssh_cmd(ip,
        "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 40000")
    try:
        return float(out.strip()) / 1000.0
    except Exception:
        return 40.0


def set_cpu_governor(ip: str, governor: str) -> bool:
    """Set cpufreq governor on a remote node."""
    cmd = f"for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo {governor} > $f 2>/dev/null; done"
    rc, _ = ssh_cmd(ip, cmd)
    return rc == 0


class ThermalRegulationLoop:
    """
    Hypothalamus thermal regulation loop.
    Polls cluster temperatures, applies throttling, publishes motor_inhibit.
    Runs as a background thread inside the Hypothalamus node.
    """

    POLL_INTERVAL_S = 5.0   # check temperatures every 5 seconds
    HYSTERESIS_C    = 3.0   # require temp to drop 3°C below threshold to un-throttle

    def __init__(self, bus, vagus=None):
        self._bus       = bus
        self._vagus     = vagus
        self._states:  Dict[str, ThermalState] = {}
        self._running  = False
        self._lock     = threading.Lock()

        # Current cluster-wide thermal status
        self._max_temp     = 40.0
        self._motor_inhibit = 0.0
        self._pub_scale    = 1.0
        self._thermal_level = "nominal"

    def _poll_all_temps(self) -> Dict[str, float]:
        """SSH-poll all throttleable nodes for their temperature."""
        temps = {}
        threads = []
        results = {}
        def poll(name, ip):
            results[name] = read_node_temp(ip)
        for name, cfg in THROTTLEABLE_NODES.items():
            t = threading.Thread(target=poll, args=(name, cfg["ip"]))
            threads.append(t); t.start()
        for t in threads: t.join(timeout=6)
        return results

    def _apply_throttle(self, node: str, temp_C: float, level: str):
        """Apply thermal throttling to a single node."""
        cfg  = THROTTLEABLE_NODES.get(node)
        if not cfg: return
        ip   = cfg["ip"]
        prev = self._states.get(node)

        if level in ("hot", "high", "critical", "shutdown"):
            # Switch to powersave governor
            if not prev or prev.governor == "performance":
                ok = set_cpu_governor(ip, "powersave")
                logger.warning(f"Thermal: {node} @ {temp_C:.0f}°C → cpufreq=powersave (ok={ok})")

        if level == "high":
            # Disable GPU on social node (face recog)
            if node == "social":
                ssh_cmd(ip, "nvidia-smi -pm 0 2>/dev/null || true")

        if level in ("nominal", "warm") and prev and prev.throttled:
            if temp_C < (T_WARM - self.HYSTERESIS_C):
                set_cpu_governor(ip, "performance")
                logger.info(f"Thermal: {node} @ {temp_C:.0f}°C → cpufreq=performance (recovered)")

        throttled  = level not in ("nominal", "warm")
        pub_scale  = {"nominal":1.0,"warm":0.9,"hot":0.6,"high":0.3,"critical":0.1,"shutdown":0.0}.get(level,0.0)
        governor   = "powersave" if throttled else "performance"
        mi         = {"nominal":0.0,"warm":0.0,"hot":0.1,"high":0.4,"critical":0.8,"shutdown":1.0}.get(level,1.0)

        self._states[node] = ThermalState(
            node=node, temp_C=temp_C, level=level,
            throttled=throttled, governor=governor,
            pub_scale=pub_scale, motor_inhibit=mi)

    def _regulation_loop(self):
        while self._running:
            t0 = time.time()
            temps = self._poll_all_temps()

            if not temps:
                # If SSH fails (BeagleBoard only, offline) — use local temp
                try:
                    local = float(Path("/sys/class/thermal/thermal_zone0/temp").read_text())/1000
                    temps = {"hypothalamus": local}
                except Exception:
                    temps = {}

            with self._lock:
                for node, temp in temps.items():
                    level = classify_temp(temp)
                    self._apply_throttle(node, temp, level)

                max_temp  = max(temps.values()) if temps else 40.0
                max_level = classify_temp(max_temp)
                self._max_temp      = max_temp
                self._thermal_level = max_level
                self._motor_inhibit = {"nominal":0.0,"warm":0.0,"hot":0.1,
                                       "high":0.4,"critical":0.8,"shutdown":1.0}.get(max_level,0.0)
                self._pub_scale     = {"nominal":1.0,"warm":0.9,"hot":0.6,
                                       "high":0.3,"critical":0.1,"shutdown":0.0}.get(max_level,0.0)

            # Publish thermal state as part of HYPO_STATE
            self._bus.publish(b"LMB_HYPO", {
                "thermal_level":  max_level,
                "max_temp_C":     max_temp,
                "motor_inhibit":  self._motor_inhibit,
                "pub_rate_scale": self._pub_scale,
                "node_temps":     {k: round(v,1) for k,v in temps.items()},
                "timestamp_ns":   time.time_ns(),
                "source":         "thermal_regulation",
            })

            if max_level == "critical":
                logger.error(f"THERMAL CRITICAL: {max_temp:.0f}°C — emergency motor stop")
                self._bus.publish(b"SFY_FREEZE", {
                    "type": "thermal_emergency", "temp_C": max_temp,
                    "action": "stop_all_motors", "timestamp_ns": time.time_ns(),
                })

            if max_level == "shutdown":
                logger.critical(f"THERMAL SHUTDOWN: {max_temp:.0f}°C — triggering vagus nerve")
                if self._vagus:
                    self._vagus.fire(f"thermal_shutdown_{max_temp:.0f}C")

            elapsed = time.time() - t0
            time.sleep(max(0, self.POLL_INTERVAL_S - elapsed))

    def start(self):
        self._running = True
        threading.Thread(target=self._regulation_loop, daemon=True).start()
        logger.info("ThermalRegulationLoop started | 5s poll | "
                    f"{len(THROTTLEABLE_NODES)} throttleable nodes")

    def stop(self): self._running = False

    @property
    def max_temp_C(self) -> float: return self._max_temp
    @property
    def motor_inhibit(self) -> float: return self._motor_inhibit
    @property
    def level(self) -> str: return self._thermal_level
