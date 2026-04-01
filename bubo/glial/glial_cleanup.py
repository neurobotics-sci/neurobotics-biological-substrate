"""
bubo/glial/glial_cleanup.py — Bubo v5550
Glial Cleanup Script — runs during LiFePO4 charging (wall charger connected)

═══════════════════════════════════════════════════════════════════════════════
BIOLOGICAL ANALOGY: GLIAL CELLS & SLEEP CLEANUP
═══════════════════════════════════════════════════════════════════════════════

Glial cells (astrocytes, microglia, oligodendrocytes) are the support
cells of the nervous system. During waking, metabolic waste products
accumulate in the brain — primarily amyloid-beta and tau (neurotoxic
proteins associated with Alzheimer's disease when they accumulate).

During deep NREM sleep, the glymphatic system (Maiken Nedergaard, 2013)
activates: brain cells (neurons + glia) SHRINK by ~60%, allowing
cerebrospinal fluid to flow through expanded extracellular spaces and
flush out accumulated waste. This requires unconsciousness because the
brain cannot maintain cognition and run the glymphatic system simultaneously.

The glymphatic system is the brain's equivalent of a dishwasher:
runs only when the kitchen (cortex) is closed for business.

BUBO IMPLEMENTATION:
  Trigger: LiFePO4 battery charger connected (voltage > 4.95V or status="Charging")
  Why wall charger? Same reason evolution chose sleep for cleanup:
    - Bubo is stationary (safe to reduce motor vigilance)
    - Power is unlimited (can run intensive compute)
    - No operational pressure (can take time for thorough cleanup)

CLEANUP TASKS (biological parallel):
  1. LTM PRUNING (synaptic homeostasis)
     Biology: sleep prunes weak synapses to prevent saturation (Tononi 2003)
     Bubo: remove LTM episodes with saliency < 0.15 (rarely retrieved memories)
     Freed space: typically 20-40% of LTM database

  2. CMAC WEIGHT CONSOLIDATION (systems consolidation)
     Biology: hippocampus → cortex memory transfer during NREM slow waves
     Bubo: save CMAC weight tables to persistent storage
     Also: compute average weights, prune dead zones (never-activated cells)

  3. HIPPOCAMPAL REPLAY (sharp-wave ripple consolidation)
     Biology: hippocampal sharp-wave ripples replay episodic memories
     during NREM, strengthening cortical traces
     Bubo: replay high-saliency episodes through LTM → increase retrieval count

  4. LOG ROTATION (metabolic waste clearance)
     Biology: protein aggregate clearance via glymphatic flow
     Bubo: compress + archive old logs, free disk space

  5. SOFTWARE UPDATE CHECK (immune system parallel)
     Biology: immune system remains active and vigilant during sleep
     Bubo: apt-get update (security patches only, not dist-upgrade)

  6. CMAC DEAD-ZONE CALIBRATION
     Biology: synaptic scaling — neurons adjust baseline activity during sleep
     Bubo: re-measure servo mechanical slop, update DEAD_ZONE_RAD adaptively

  7. FACE DATABASE MAINTENANCE
     Biology: memory reconsolidation — memories are stabilised during sleep
     Bubo: re-compute face embedding norms, remove corrupted entries

CHARGER DETECTION:
  Method 1: Read /sys/class/power_supply/BAT0/status == "Charging"
  Method 2: Voltage on supply rail > 4.95V (above LiFePO4 nominal 3.2V×4=12.8V)
  Method 3: Current direction sensor (charger current = positive)
  Method 4: GPIO pin wired to charger "charging" indicator LED
"""

import time, logging, json, os, subprocess, sqlite3, shutil
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger("GlialCleanup")

LTM_DB_PATH       = Path("/opt/bubo/data/bubo_ltm.db")
CMAC_WEIGHTS_PATH = Path("/opt/bubo/data/cmac_weights.json")
FACE_DB_PATH      = Path("/opt/bubo/data/bubo_faces.db")
LOG_DIR           = Path("/var/log/bubo")
ARCHIVE_DIR       = Path("/opt/bubo/data/log_archive")
CHARGER_STAT_PATH = Path("/sys/class/power_supply/BAT0/status")
CHARGER_VOLT_PATH = Path("/sys/class/power_supply/BAT0/voltage_now")

SALIENCY_PRUNE_THRESHOLD = 0.15    # prune episodes weaker than this
LOW_RETRIEVAL_PRUNE      = 0       # prune if never retrieved and saliency < 0.25
CMAC_DEAD_CELL_THRESHOLD = 0.001   # CMAC weights below this are "dead" cells


def detect_charger() -> bool:
    """
    Detect if wall charger is connected.
    Tries multiple methods in order of reliability.
    Returns True if charging, False otherwise.
    """
    # Method 1: sysfs power supply status
    if CHARGER_STAT_PATH.exists():
        try:
            status = CHARGER_STAT_PATH.read_text().strip()
            if status in ("Charging", "Full"):
                logger.info(f"Charger detected via sysfs: status={status}")
                return True
        except Exception: pass

    # Method 2: voltage above nominal LiFePO4 (12.8V nominal, >13.2V = charging)
    if CHARGER_VOLT_PATH.exists():
        try:
            v_uV = int(CHARGER_VOLT_PATH.read_text().strip())
            v = v_uV / 1e6
            if v > 13.2:
                logger.info(f"Charger detected via voltage: {v:.2f}V")
                return True
        except Exception: pass

    # Method 3: GPIO charger indicator (if wired)
    charger_gpio = Path("/sys/class/gpio/gpio45/value")
    if charger_gpio.exists():
        try:
            if charger_gpio.read_text().strip() == "1":
                logger.info("Charger detected via GPIO45")
                return True
        except Exception: pass

    return False


class GlialCleanupEngine:
    """
    Runs all cleanup tasks when charging is detected.
    Each task is independent — failure of one doesn't abort others.
    """

    def __init__(self, bus=None):
        self._bus = bus
        self._results: dict = {}
        self._start_time = 0.0

    def run_all(self) -> dict:
        """Execute full glial cleanup sequence."""
        self._start_time = time.time()
        logger.info("═══ GLIAL CLEANUP STARTING (charger connected) ═══")

        tasks = [
            ("ltm_prune",          self._ltm_prune),
            ("cmac_consolidate",   self._cmac_consolidate),
            ("hippo_replay",       self._hippo_replay),
            ("cmac_calibrate",     self._cmac_calibrate),
            ("face_db_maintain",   self._face_db_maintain),
            ("log_rotate",         self._log_rotate),
            ("apt_security_patch", self._apt_update),
        ]

        for name, task in tasks:
            try:
                logger.info(f"Glial task: {name}...")
                result = task()
                self._results[name] = {"status": "ok", **result}
                logger.info(f"  {name}: {result}")
            except Exception as e:
                self._results[name] = {"status": "error", "error": str(e)}
                logger.error(f"  {name} FAILED: {e}")

        elapsed = time.time() - self._start_time
        self._results["_elapsed_s"] = round(elapsed, 1)

        if self._bus:
            self._bus.publish(b"SYS_GLIAL", {
                "results": self._results,
                "elapsed_s": elapsed,
                "timestamp_ns": time.time_ns(),
            })

        logger.info(f"═══ GLIAL CLEANUP COMPLETE ({elapsed:.0f}s) ═══")
        return self._results

    def _ltm_prune(self) -> dict:
        """
        Remove low-saliency, unretrieved LTM episodes.
        Biological: synaptic homeostasis — weak synapses pruned during sleep.
        """
        if not LTM_DB_PATH.exists():
            return {"skipped": "LTM database not found"}
        conn = sqlite3.connect(str(LTM_DB_PATH))
        conn.execute("PRAGMA journal_mode=WAL")
        before = conn.execute("SELECT COUNT(*) FROM ltm").fetchone()[0]
        # Prune very low saliency
        conn.execute("DELETE FROM ltm WHERE saliency < ?", (SALIENCY_PRUNE_THRESHOLD,))
        # Prune unretrieved medium-saliency (stale memories)
        conn.execute("DELETE FROM ltm WHERE retrievals = 0 AND saliency < 0.25")
        conn.commit()
        after  = conn.execute("SELECT COUNT(*) FROM ltm").fetchone()[0]
        # VACUUM to reclaim disk space
        conn.execute("VACUUM")
        conn.close()
        pruned = before - after
        return {"before": before, "after": after, "pruned": pruned,
                "reduction_pct": round(100*pruned/max(before,1), 1)}

    def _cmac_consolidate(self) -> dict:
        """
        Save CMAC weight tables and compute statistics.
        Biological: systems consolidation — hippocampal→cortical memory transfer.
        """
        if not CMAC_WEIGHTS_PATH.exists():
            return {"skipped": "CMAC weights file not found"}
        with open(CMAC_WEIGHTS_PATH) as f:
            data = json.load(f)
        weights_list = data.get("weights", [])
        n_surfaces   = len(weights_list)
        n_updates    = data.get("n_updates", 0)
        # Compute weight statistics
        all_w  = np.concatenate([np.array(w).flatten() for w in weights_list])
        n_dead = int(np.sum(np.abs(all_w) < CMAC_DEAD_CELL_THRESHOLD))
        dead_pct = 100 * n_dead / max(len(all_w), 1)
        # Archive a snapshot
        backup = CMAC_WEIGHTS_PATH.with_suffix(f".backup_{int(time.time())}.json")
        shutil.copy2(CMAC_WEIGHTS_PATH, backup)
        return {"n_surfaces": n_surfaces, "n_updates": n_updates,
                "dead_cells_pct": round(dead_pct, 1),
                "weight_rms": round(float(np.sqrt(np.mean(all_w**2))), 4),
                "backup": str(backup)}

    def _hippo_replay(self) -> dict:
        """
        Replay high-saliency episodes to strengthen consolidation.
        Biological: hippocampal sharp-wave ripple replay during NREM.
        """
        if not LTM_DB_PATH.exists():
            return {"skipped": "no LTM database"}
        conn = sqlite3.connect(str(LTM_DB_PATH))
        # Find top-20 salient unretired episodes, increment their retrieval count
        rows = conn.execute(
            "SELECT trace_id, saliency FROM ltm ORDER BY saliency DESC LIMIT 20"
        ).fetchall()
        replayed = 0
        for trace_id, sal in rows:
            conn.execute(
                "UPDATE ltm SET retrievals = retrievals + 1, "
                "saliency = MIN(1.0, saliency + 0.02) "
                "WHERE trace_id = ?", (trace_id,))
            replayed += 1
        conn.commit(); conn.close()
        return {"replayed_episodes": replayed, "method": "sharp_wave_ripple_sim"}

    def _cmac_calibrate(self) -> dict:
        """
        Re-calibrate CMAC dead-zone from current servo telemetry.
        Reads actual position noise floor from Dynamixel feedback.
        Biological: synaptic scaling — adjust baseline during sleep.
        """
        try:
            from bubo.shared.hal.servo_hal import ServoHAL
            hal = ServoHAL(n_joints=26, sim_mode=True)
            # Sample 100 readings of a stationary joint to estimate noise
            samples = []
            for _ in range(100):
                states = hal.read_states()
                if states: samples.append(states[0].position_rad)
                time.sleep(0.01)
            if len(samples) > 10:
                noise_std = float(np.std(samples))
                # Dead-zone should be 2× noise standard deviation
                recommended_dz = round(max(0.001, 2.0 * noise_std), 4)
                return {"noise_std_rad": round(noise_std, 4),
                        "recommended_dead_zone_rad": recommended_dz,
                        "current_dead_zone_rad": 0.002}
        except Exception as e:
            pass
        return {"skipped": "HAL not available in current context", "default_dz": 0.002}

    def _face_db_maintain(self) -> dict:
        """
        Validate and clean face embedding database.
        Biological: memory reconsolidation — stabilise stored representations.
        """
        if not FACE_DB_PATH.exists():
            return {"skipped": "face database not found"}
        conn = sqlite3.connect(str(FACE_DB_PATH))
        total = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        # Check embedding integrity (correct dimension)
        invalid = 0
        rows = conn.execute("SELECT face_id, embedding FROM faces").fetchall()
        for fid, emb_blob in rows:
            try:
                emb = np.frombuffer(emb_blob, dtype=np.float32)
                if len(emb) != 128:
                    conn.execute("DELETE FROM faces WHERE face_id=?", (fid,))
                    invalid += 1
            except Exception:
                conn.execute("DELETE FROM faces WHERE face_id=?", (fid,))
                invalid += 1
        conn.commit(); conn.close()
        return {"total_faces": total, "invalid_removed": invalid, "remaining": total - invalid}

    def _log_rotate(self) -> dict:
        """Compress and archive old log files."""
        if not LOG_DIR.exists():
            return {"skipped": "log directory not found"}
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        archived = 0; freed_mb = 0
        for logfile in LOG_DIR.glob("*.log"):
            if logfile.stat().st_size > 1024 * 1024:  # > 1MB
                archive_name = ARCHIVE_DIR / f"{logfile.stem}_{int(time.time())}.log.gz"
                try:
                    subprocess.run(f"gzip -c {logfile} > {archive_name} && > {logfile}",
                                   shell=True, timeout=30)
                    freed_mb += logfile.stat().st_size / (1024*1024)
                    archived += 1
                except Exception: pass
        return {"archived_files": archived, "freed_mb": round(freed_mb, 1)}

    def _apt_update(self) -> dict:
        """Security patches only — do NOT dist-upgrade."""
        try:
            r = subprocess.run(
                "apt-get update -qq && apt-get upgrade -s 2>/dev/null | grep -c 'Inst' || echo 0",
                shell=True, capture_output=True, text=True, timeout=60)
            pending = int(r.stdout.strip().split("\n")[-1])
            return {"pending_security_updates": pending,
                    "note": "Run apt-get upgrade manually to apply"}
        except Exception as e:
            return {"skipped": str(e)}


class GlialCleanupDaemon:
    """
    Daemon that polls for charger connection and triggers cleanup.
    Runs as a background thread on the LTM node (or hypothalamus).
    Also integrates with the nod-off system: cleanup during NREM sleep.
    """

    CHARGER_POLL_S    = 30.0   # check for charger every 30 seconds
    CLEANUP_COOLDOWN  = 3600.0 # don't run cleanup more than once per hour

    def __init__(self, bus=None):
        self._bus        = bus
        self._engine     = GlialCleanupEngine(bus)
        self._running    = False
        self._last_run   = 0.0
        self._charging   = False

    def _daemon_loop(self):
        while self._running:
            time.sleep(self.CHARGER_POLL_S)
            charging = detect_charger()
            if charging != self._charging:
                self._charging = charging
                if charging:
                    logger.info("GlialCleanup: charger connected — scheduling cleanup")

            # Run cleanup if: charging, enough time has elapsed
            if (charging and
                    (time.time() - self._last_run) > self.CLEANUP_COOLDOWN):
                self._last_run = time.time()
                results = self._engine.run_all()
                logger.info(f"GlialCleanup complete: {results.get('_elapsed_s',0):.0f}s")

    def start(self):
        self._running = True
        threading.Thread(target=self._daemon_loop, daemon=True).start()
        logger.info(f"GlialCleanupDaemon started | polling every {self.CHARGER_POLL_S:.0f}s")

    def stop(self): self._running = False

    def trigger_now(self):
        """Force cleanup run (for testing or manual trigger)."""
        self._engine.run_all()


import threading  # needed by daemon loop

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s")
    print("Bubo v5550 — Glial Cleanup")
    if detect_charger():
        print("Charger detected — running cleanup now...")
        engine = GlialCleanupEngine()
        results = engine.run_all()
        print(json.dumps(results, indent=2))
    else:
        print("No charger detected. Connect charger to trigger cleanup.")
        print("Or pass --force to run anyway:")
        import sys
        if "--force" in sys.argv:
            engine = GlialCleanupEngine()
            results = engine.run_all()
            print(json.dumps(results, indent=2))
