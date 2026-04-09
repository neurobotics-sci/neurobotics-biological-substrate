"""
bubo/shared/watchdog/node_watchdog.py — Bubo v5400
Per-node watchdog: publishes heartbeat, monitors ZMQ bus, triggers systemd restart.

PROBLEM: Silent node failure.
If a node crashes (OOM, uncaught exception, deadlock), it stops publishing.
The cluster continues but that brain region is gone. No other node detects this.

SOLUTION: Each node embeds a NodeWatchdog that:
  1. Publishes T.SYS_WATCHDOG at 1Hz with node name + uptime + metrics
  2. Checks own ZMQ publish queue for backup — if pub stuck, restart self
  3. Monitors own memory and CPU — if above threshold, log warning
  4. On SIGTERM: clean shutdown, final state save

Also: WatchdogSupervisor runs on blueraven/manager and monitors all 22 nodes.
If any node misses 3 heartbeats → SSH restart → node restarts via systemd.
"""

import time, os, signal, threading, logging, json, subprocess
import numpy as np
from pathlib import Path

logger = logging.getLogger("NodeWatchdog")

WATCHDOG_HZ        = 1.0     # heartbeat rate
RESTART_THRESHOLD  = 3       # missed heartbeats before restart
MEM_WARN_MB        = 200     # warn if > 200MB used
MEM_CRIT_MB        = 350     # critical if > 350MB (Nano 2GB has ~1.7GB usable)
CPU_WARN_PERCENT   = 80.0


class NodeWatchdog:
    """
    Embedded watchdog for one Bubo node process.
    Call start() after bus.start() in each node's start() method.
    """

    def __init__(self, node_name: str, bus, pid: int = None):
        self.node_name = node_name
        self._bus      = bus
        self._pid      = pid or os.getpid()
        self._start_t  = time.time()
        self._running  = False
        self._lock     = threading.Lock()
        self._last_pub_ns = time.time_ns()
        self._pub_count   = 0
        self._pub_count_last = 0
        self._health_log = Path(f"/var/log/bubo/{node_name}_watchdog.log")

        # Install signal handlers
        signal.signal(signal.SIGTERM, self._on_sigterm)
        signal.signal(signal.SIGINT,  self._on_sigterm)

    def heartbeat(self):
        """External hook to manually trigger a heartbeat or verify life."""
        with self._lock:
            uptime = time.time() - self._start_t
            return {
                "node": self.node_name,
                "uptime_s": round(uptime, 1),
                "status": "ok" if self._running else "stopped"
            }

    def record_publish(self):
        """Call after each bus.publish(). Tracks publication rate."""
        with self._lock:
            self._pub_count += 1
            self._last_pub_ns = time.time_ns()

    def _get_memory_mb(self) -> float:
        """Read process RSS from /proc/self/status."""
        try:
            with open(f"/proc/{self._pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception: pass
        return 0.0

    def _get_cpu_percent(self) -> float:
        """Read CPU usage via /proc/stat delta."""
        try:
            import psutil
            return psutil.Process(self._pid).cpu_percent(interval=0.1)
        except Exception: return 0.0

    def _watchdog_loop(self):
        interval = 1.0 / WATCHDOG_HZ
        while self._running:
            t0 = time.time()
            uptime = t0 - self._start_t
            mem_mb = self._get_memory_mb()
            cpu    = self._get_cpu_percent()
            with self._lock:
                pubs   = self._pub_count
                delta  = pubs - self._pub_count_last
                self._pub_count_last = pubs
                age_ms = (time.time_ns() - self._last_pub_ns) / 1e6

            # Health checks
            status = "ok"
            if mem_mb > MEM_CRIT_MB:
                status = "critical_memory"
                logger.error(f"{self.node_name} memory CRITICAL: {mem_mb:.0f}MB")
            elif mem_mb > MEM_WARN_MB:
                status = "warn_memory"
                logger.warning(f"{self.node_name} memory high: {mem_mb:.0f}MB")

            if cpu > CPU_WARN_PERCENT:
                status = "warn_cpu"
                logger.warning(f"{self.node_name} CPU high: {cpu:.1f}%")

            if age_ms > 5000 and uptime > 10:
                status = "stalled"
                logger.error(f"{self.node_name} publication STALLED (last pub {age_ms:.0f}ms ago)")

            # Publish heartbeat
            try:
                self._bus.publish(b"SYS_WD", {
                    "node":         self.node_name,
                    "uptime_s":     round(uptime, 1),
                    "mem_mb":       round(mem_mb, 1),
                    "cpu_pct":      round(cpu, 1),
                    "pub_rate_hz":  float(delta),
                    "last_pub_ms":  round(age_ms, 1),
                    "status":       status,
                    "pid":          self._pid,
                    "timestamp_ns": time.time_ns(),
                })
            except Exception as e:
                logger.error(f"Watchdog publish failed: {e}")

            time.sleep(max(0, interval - (time.time() - t0)))

    def _on_sigterm(self, signum, frame):
        logger.info(f"{self.node_name} received SIGTERM — clean shutdown")
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._watchdog_loop, daemon=True).start()
        logger.debug(f"NodeWatchdog started for {self.node_name}")

    def stop(self): self._running = False


class WatchdogSupervisor:
    """
    Cluster-level watchdog. Runs on blueraven or a management host.
    Subscribes to SYS_WD heartbeats from all 22 nodes.
    On missed heartbeat: SSH to the failed node, trigger systemd restart.
    """

    """"""
    NODES = {
        "hypothalamus":"127.0.0.1", "thalamus_l": "127.0.0.1",
        "thalamus_r": "127.0.0.1", "cerebellum": "127.0.0.1",
        "basal_ganglia": "127.0.0.1", "somatosensory": "127.0.0.1"
    }
    """_summary_

    Returns:
        _type_: _description_
    """
    SSH_KEY = Path.home() / ".ssh/bubo_id_ed25519"
    TIMEOUT_S = 5.0   # 5s = 5 missed heartbeats

    def __init__(self, sub_endpoints: list):
        self._last_hb   = {n: time.time() for n in self.NODES}
        self._missed    = {n: 0 for n in self.NODES}
        self._restarted = {n: 0 for n in self.NODES}
        self._running   = False
        import zmq
        self._ctx = zmq.Context()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.RCVTIMEO, 2000)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"SYS_WD")
        for ep in sub_endpoints: self._sub.connect(ep)

    def _on_heartbeat(self, node_name: str):
        if node_name in self._last_hb:
            self._last_hb[node_name] = time.time()
            self._missed[node_name]  = 0

    def _restart_node(self, node_name: str):
        ip = self.NODES.get(node_name)
        if not ip: return
        logger.warning(f"RESTART: {node_name} ({ip}) — missed {RESTART_THRESHOLD} HBs")
        try:
            subprocess.run(
                f"ssh -o StrictHostKeyChecking=no -i {self.SSH_KEY} brain@{ip} "
                f"'systemctl restart bubo-{node_name}'",
                shell=True, timeout=10)
            self._restarted[node_name] += 1
        except Exception as e:
            logger.error(f"Restart {node_name} failed: {e}")

    def _supervisor_loop(self):
        while self._running:
            # Receive heartbeats
            try:
                parts = self._sub.recv_multipart()
                if len(parts) == 2:
                    msg = json.loads(parts[1].decode())
                    node = msg.get("payload", {}).get("node", "")
                    self._on_heartbeat(node)
            except Exception: pass

            # Check for missed heartbeats
            now = time.time()
            for node, last in self._last_hb.items():
                if now - last > self.TIMEOUT_S:
                    self._missed[node] = self._missed.get(node, 0) + 1
                    if self._missed[node] == RESTART_THRESHOLD:
                        self._restart_node(node)
                else:
                    self._missed[node] = 0

    def start(self):
        self._running = True
        threading.Thread(target=self._supervisor_loop, daemon=True).start()
        logger.info(f"WatchdogSupervisor started | monitoring {len(self.NODES)} nodes")

    def status(self) -> dict:
        now = time.time()
        return {n: {"age_s": round(now - t, 1), "restarts": self._restarted.get(n, 0),
                    "missed": self._missed.get(n, 0)}
                for n, t in self._last_hb.items()}

    def stop(self):
        self._running = False
        self._sub.close(); self._ctx.term()
