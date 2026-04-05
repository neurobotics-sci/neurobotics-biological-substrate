"""
bubo/nodes/thalamus/distributed/distributed_thalamus.py — Bubo v6000

Distributed Thalamus — configuration guide and failover coordinator.

══════════════════════════════════════════════════════════════════════
BIOLOGY: THE THALAMUS AS A DISTRIBUTED RELAY ARCHITECTURE
══════════════════════════════════════════════════════════════════════

The human thalamus is a paired structure (~8cm³ each) with ~50 named nuclei.
It is NOT a simple relay — it is the primary gateway for all sensory information
reaching the cortex AND the primary pathway for cortical↔basal ganglia↔cerebellar
communication.

Key thalamic nuclei and their functions:
  VPL (ventroposterolateral): somatosensory relay (touch, pain, temp)
  VPM (ventroposteromedial):  face touch + taste relay
  LGN (lateral geniculate):  visual relay (6 laminae, M and P pathways)
  MGN (medial geniculate):   auditory relay
  Pulvinar:                  multisensory, SC relay, spatial attention
  MD (mediodorsal):          PFC↔limbic relay (emotion-cognition integration)
  VA/VL (ventral anterior/lateral): BG→M1 + Cerebellum→M1 relay
  Reuniens:                  Hippocampus↔PFC communication
  CM/PF (centromedian/parafascicular): arousal, attention, pain modulation
  Reticular nucleus (TRN):  NOT a relay — pure inhibitory, alpha-gates all input

══════════════════════════════════════════════════════════════════════
CONFIGURATIONS FOR BUBO (current and suggested)
══════════════════════════════════════════════════════════════════════

CURRENT (v5550/v5900/v6000):
  2 nodes: Thalamus-L + Thalamus-R on 2 Orin 8GB

  Thalamus-L (192.168.1.13): VPL + LGN + MGN + Pulvinar (sensory relay)
  Thalamus-R (192.168.1.18): VA/VL + MD + Reuniens (motor/PFC relay)

  Failover: L monitors R heartbeat and vice versa. If either fails,
  surviving node activates backup relay tables.

SUGGESTED CONFIGURATIONS:

  Option A: 3-node thalamus (add Nano 4GB at 192.168.1.36)
    Node 1 (Orin): VPL + LGN + MGN — raw sensory (high bandwidth, needs GPU)
    Node 2 (Orin): Pulvinar + MD + Reuniens — integrative relay
    Node 3 (Nano): VA/VL + CM/PF + TRN — motor + arousal + alpha-gate
    Advantage: isolates high-rate visual relay from slower cognitive relay
    Cost: +~$50 (Nano 4GB)

  Option B: Dedicated TRN node (recommended addition)
    The Thalamic Reticular Nucleus (TRN) is entirely inhibitory.
    It deserves its own node because:
    - It gates ALL thalamic nuclei simultaneously
    - Alpha oscillations originate here
    - Attention direction changes TRN activity INSTANTLY
    Currently: TRN function is approximated inside each thalamus node.
    Better: dedicated TRN node publishing alpha_gate values at 100Hz.
    
  Option C: Distributed relay matrix (4+ nodes, advanced)
    Each thalamic nucleus on a separate thread pool, orchestrated by
    a coordinator. Allows nucleus-specific QoS and independent failure recovery.

ADDITIONS NEEDED FOR FULL FUNCTION:

  1. VOR integration: thalamus receives from vestibular nuclei → must forward
     to oculomotor node at <5ms. Currently bypassed via direct IMU→SC path.
     Fix: add T.VEST_THAL_RELAY topic, Thalamus-L processes + relays with timestamp.

  2. CM/PF arousal modulation: centromedian nucleus is the "arousal gate."
     Currently modelled inside hypothalamus. Should be in thalamus with
     separate input from RF_AROUSAL (reticular formation).

  3. Pulvinar attention priority map: pulvinar should compute a 2D salience map
     from SC + cortical top-down input, not just a scalar alpha gate.
     Requires 2D array computation → justifies GPU (stays on Orin).

  4. MD nucleus social modulation: MD relay carries oxytocin-modulated
     limbic→PFC signals. Currently social modulation is applied at Thalamus-R
     but biological MD nucleus has its own gating mechanism separate from VA/VL.

  5. Thalamic relay timing (CRITICAL):
     All thalamic relay adds ~2ms latency (message serialization + routing).
     For pain/reflex: bypass thalamus entirely (spinal→amygdala direct).
     For vision at 30fps: thalamic latency is 2ms vs 33ms frame period → acceptable.
     For VOR at 200Hz: thalamic latency is 2ms vs 5ms period → PROBLEMATIC.
     Solution: VOR keeps its direct path (IMU→SC→oculomotor). Thalamus relays
     non-time-critical visual context only.
"""

import time, logging, threading
import numpy as np
from bubo.shared.bus.neural_bus import NeuralBus, T

logger = logging.getLogger("DistributedThalamus")

HB_INTERVAL_S  = 0.05   # 20Hz heartbeat
HB_TIMEOUT_S   = 0.15   # 3 missed → failover

# Nucleus assignment map — which relay node handles which nuclei
NUCLEUS_MAP = {
    "thalamus-l": {
        "VPL":  {"function":"somatosensory relay", "inputs":["S1","spinal"], "outputs":["PFC","M1"]},
        "LGN":  {"function":"visual relay M/P paths","inputs":["V1"],"outputs":["PFC","association"]},
        "Pulv": {"function":"spatial attention / SC relay","inputs":["SC","MT"],"outputs":["PFC","parietal"]},
    },
    "thalamus-r": {
        "VA":     {"function":"BG→M1 relay","inputs":["basal_ganglia"],"outputs":["M1","premotor"]},
        "VL":     {"function":"cerebellum→M1","inputs":["cerebellum"],"outputs":["M1"]},
        "Reun":   {"function":"hippo→PFC spatial","inputs":["hippocampus"],"outputs":["PFC","association"]},
        "CM_PF":  {"function":"arousal / attention","inputs":["RF","hypothalamus"],"outputs":["ALL"]},
    },
}

# Failover relay table — what Thalamus-L takes over if Thalamus-R fails
L_BACKUP_FOR_R = {
    "VA":   {"redirect": "CTX_PFC_CMD", "comment": "Pass BG action directly to PFC"},
    "VL":   {"redirect": "CRB_DELTA",   "comment": "Pass cerebellar delta directly"},
    "MD":   {"redirect": "LMB_CEA_OUT", "comment": "Pass limbic output directly to cortex"},
    "Reun": {"redirect": "LMB_HCONTEXT","comment": "Pass hippocampal context to PFC directly"},
}

R_BACKUP_FOR_L = {
    "VPL": {"redirect": "AFF_TOUCH_SA1", "comment": "S1 signals → cortex unfiltered"},
    "LGN": {"redirect": "AFF_VIS_V1",    "comment": "V1 → cortex unfiltered"},
    "MGN": {"redirect": "AFF_AUD_A1",    "comment": "A1 → cortex unfiltered"},
    "Pulv":{"redirect": "BS_SC_SACC",    "comment": "SC saccade → cortex unfiltered"},
}


class ThalamusCoordinator:
    """
    Monitors both thalamic nodes and manages failover.
    Runs on hypothalamus node as a background thread.
    Publishes T.THAL_FAILOVER if either node becomes unresponsive.
    """

    def __init__(self, bus: NeuralBus):
        self._bus   = bus
        self._hb_l  = time.time()
        self._hb_r  = time.time()
        self._l_ok  = True
        self._r_ok  = True
        self._running = False
        self._lock  = threading.Lock()

    def on_heartbeat(self, msg):
        node = msg.payload.get("node","")
        with self._lock:
            if "thalamus-l" in node: self._hb_l = time.time()
            if "thalamus-r" in node: self._hb_r = time.time()

    def _monitor(self):
        while self._running:
            time.sleep(0.1)
            now = time.time()
            with self._lock:
                l_age = now - self._hb_l
                r_age = now - self._hb_r
                l_was_ok = self._l_ok
                r_was_ok = self._r_ok
                self._l_ok = l_age < HB_TIMEOUT_S
                self._r_ok = r_age < HB_TIMEOUT_S

            if l_was_ok and not self._l_ok:
                logger.warning("Thalamus-L FAILED → Thalamus-R taking over sensory relay")
                self._bus.publish(T.THAL_FAILOVER, {
                    "failed": "thalamus-l", "backup": "thalamus-r",
                    "redirect_table": R_BACKUP_FOR_L,
                    "timestamp_ns": time.time_ns(),
                })
            elif not l_was_ok and self._l_ok:
                logger.info("Thalamus-L recovered")
                self._bus.publish(T.THAL_FAILOVER, {
                    "failed": "none", "backup": "none",
                    "status": "recovered", "node": "thalamus-l",
                    "timestamp_ns": time.time_ns(),
                })

            if r_was_ok and not self._r_ok:
                logger.warning("Thalamus-R FAILED → Thalamus-L taking over motor relay")
                self._bus.publish(T.THAL_FAILOVER, {
                    "failed": "thalamus-r", "backup": "thalamus-l",
                    "redirect_table": L_BACKUP_FOR_R,
                    "timestamp_ns": time.time_ns(),
                })
            elif not r_was_ok and self._r_ok:
                logger.info("Thalamus-R recovered")
                self._bus.publish(T.THAL_FAILOVER, {
                    "failed": "none", "backup": "none",
                    "status": "recovered", "node": "thalamus-r",
                    "timestamp_ns": time.time_ns(),
                })

    def start(self):
        self._running = True
        threading.Thread(target=self._monitor, daemon=True).start()
        logger.info("ThalamusCoordinator started | 2-node failover monitor")

    def stop(self): self._running = False

    @property
    def status(self) -> dict:
        with self._lock:
            return {"thalamus_l": self._l_ok, "thalamus_r": self._r_ok}
