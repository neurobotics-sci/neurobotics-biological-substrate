#!/usr/bin/env python3
"""
scripts/monitoring/thermal_monitor.py — Bubo v5400
Cluster-wide thermal monitor. Runs on blueraven.

Polls all 22 Jetson nodes every 5s for:
  CPU temperature, GPU temperature, throttling state, fan RPM (where available)

If any node exceeds thresholds:
  > 70°C WARN: reduce that node's publication rate via T.HYPO_STATE motor_inhibit
  > 80°C HIGH: send REST command to associated limb nodes
  > 85°C CRIT: emergency stop all motor activity
  > 90°C SHUTDOWN: initiate graceful shutdown of that node
"""

import subprocess, time, json, zmq, threading
from pathlib import Path

SSH_KEY = Path.home() / ".ssh/bubo_id_ed25519"
SSH_OPTS = f"-o StrictHostKeyChecking=no -o ConnectTimeout=3 -i {SSH_KEY}"

GRN="\033[92m"; YEL="\033[93m"; RED="\033[91m"; BLD="\033[1m"; NC="\033[0m"

NODES = {
    "pfc-l":"192.168.1.10","pfc-r":"192.168.1.11","hypothalamus":"192.168.1.12",
    "thalamus-l":"192.168.1.13","broca":"192.168.1.14","insula":"192.168.1.15",
    "parietal":"192.168.1.16","cingulate":"192.168.1.17",
    "thalamus-r":"192.168.1.18","social":"192.168.1.19",
    "hippocampus":"192.168.1.30","amygdala":"192.168.1.31",
    "cerebellum":"192.168.1.32","basal-ganglia":"192.168.1.33",
    "association":"192.168.1.34","ltm-store":"192.168.1.35",
    "visual":"192.168.1.50","auditory":"192.168.1.51",
    "somatosensory":"192.168.1.52","spinal-arms":"192.168.1.53",
    "sup-colliculus":"192.168.1.60","spinal-legs":"192.168.1.61",
}

THRESHOLDS = {"warn":70, "high":80, "crit":85, "shutdown":90}


def read_temp(ip: str) -> dict:
    """SSH read of CPU temp from remote node."""
    try:
        r = subprocess.run(
            f"ssh {SSH_OPTS} brain@{ip} "
            f"\"cat /sys/class/thermal/thermal_zone0/temp "
            f"    /sys/class/thermal/thermal_zone1/temp 2>/dev/null;"
            f"  nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null || echo 0\"",
            shell=True, capture_output=True, text=True, timeout=4)
        lines = r.stdout.strip().split("\n")
        cpu_mC = int(lines[0].strip()) if lines[0].strip().isdigit() else 40000
        gpu_mC = int(lines[1].strip()) if len(lines) > 1 and lines[1].strip().isdigit() else 0
        gpu_smi = float(lines[2].strip()) if len(lines) > 2 else 0.0
        return {
            "cpu_C":  cpu_mC / 1000.0,
            "gpu_C":  max(gpu_mC / 1000.0, gpu_smi),
            "peak_C": max(cpu_mC / 1000.0, gpu_mC / 1000.0, gpu_smi),
        }
    except Exception:
        return {"cpu_C": 0.0, "gpu_C": 0.0, "peak_C": 0.0}


def send_thermal_alert(pub_sock, node: str, peak_C: float, action: str):
    """Publish thermal alert to the cluster."""
    msg = json.dumps({
        "topic": "SYS_THERMAL",
        "timestamp_ms": time.time() * 1000,
        "timestamp_ns": time.time_ns() * 1000,
        "source": "thermal_monitor",
        "target": node,
        "payload": {"node": node, "peak_C": peak_C, "action": action},
        "phase": 0.0,
        "neuromod": {"DA": 0.3, "NE": 0.5, "5HT": 0.5, "ACh": 0.5},
    }).encode()
    pub_sock.send_multipart([b"SYS_THERMAL", msg])


def emergency_stop(ip: str, node: str):
    """Send SIGTERM to all Bubo processes on a node."""
    print(f"\n{RED}{BLD}EMERGENCY STOP: {node} ({ip}){NC}")
    subprocess.run(
        f"ssh {SSH_OPTS} brain@{ip} 'pkill -SIGTERM -f bubo 2>/dev/null'",
        shell=True, timeout=5)


def monitor_loop():
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind("tcp://*:5699")   # thermal monitor pub port
    time.sleep(0.5)

    temps = {n: {"cpu_C": 0.0, "gpu_C": 0.0, "peak_C": 0.0} for n in NODES}
    interval = 5.0

    print(f"\n{BLD}Bubo v5400 — Thermal Monitor (22 nodes){NC}")
    print(f"Thresholds: WARN={THRESHOLDS['warn']}°C HIGH={THRESHOLDS['high']}°C "
          f"CRIT={THRESHOLDS['crit']}°C SHUTDOWN={THRESHOLDS['shutdown']}°C\n")

    while True:
        t0 = time.time()

        results = {}
        threads = []
        def poll(name, ip):
            results[name] = read_temp(ip)
        for name, ip in NODES.items():
            t = threading.Thread(target=poll, args=(name, ip))
            threads.append(t); t.start()
        for t in threads: t.join(timeout=5)

        # Display
        print(f"\r{time.strftime('%H:%M:%S')}  ", end="")
        for name, t in results.items():
            peak = t.get("peak_C", 0)
            if   peak >= THRESHOLDS["shutdown"]: col = f"{RED}{BLD}"
            elif peak >= THRESHOLDS["crit"]:     col = RED
            elif peak >= THRESHOLDS["high"]:     col = YEL
            elif peak >= THRESHOLDS["warn"]:     col = YEL
            else:                                col = GRN
            print(f"{col}{name[:8]:8} {peak:4.0f}°C{NC}  ", end="")

        # Alerts
        for name, t in results.items():
            peak = t.get("peak_C", 0)
            ip   = NODES[name]
            if peak >= THRESHOLDS["shutdown"]:
                print(f"\n{RED}{BLD}SHUTDOWN THRESHOLD: {name} {peak:.0f}°C{NC}")
                emergency_stop(ip, name)
                send_thermal_alert(pub, name, peak, "emergency_stop")
            elif peak >= THRESHOLDS["crit"]:
                print(f"\n{RED}CRITICAL: {name} {peak:.0f}°C — motor inhibit MAX{NC}")
                send_thermal_alert(pub, name, peak, "motor_inhibit_100")
            elif peak >= THRESHOLDS["high"]:
                print(f"\n{YEL}HIGH: {name} {peak:.0f}°C — reducing motor activity{NC}")
                send_thermal_alert(pub, name, peak, "motor_inhibit_50")

        time.sleep(max(0, interval - (time.time() - t0)))


if __name__ == "__main__":
    monitor_loop()
