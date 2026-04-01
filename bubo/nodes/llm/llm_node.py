"""
bubo/nodes/llm/llm_node.py — Bubo v5900
LLM Reasoning Node — runs on PFC-L (192.168.1.10) or Social (192.168.1.19)

Subscribes to high-level events that need common-sense reasoning:
  - Novel object / unknown situation
  - Social language input
  - Complex multi-step task requests
  - Self-monitoring queries from PFC

Publishes T.CTX_LLM_RESP with reasoning results.
Adapts quantization mode based on system state (hypothalamus telemetry).
"""
import time, json, logging, threading
import numpy as np
from bubo.shared.bus.neural_bus import NeuralBus, T
from bubo.llm.edge_llm import EdgeLLMEngine
from bubo.shared.watchdog.node_watchdog import NodeWatchdog

logger = logging.getLogger("LLMNode")


class LLMNode:
    def __init__(self, config: dict):
        self.name = "LLMNode"
        self.bus  = NeuralBus(self.name, config["pub_port"], config["sub_endpoints"])
        self.wd   = NodeWatchdog(self.name, self.bus)
        self.llm  = EdgeLLMEngine(self.bus)

        # System state for quantization decisions
        self._battery    = 1.0
        self._thermal_C  = 45.0
        self._motor_on   = False
        self._charging   = False
        self._da         = 0.6
        self._social_ctx = {}
        self._pfc_action = "stand_still"
        self._query_q: list = []
        self._running    = False
        self._lock       = threading.Lock()

    def _on_hypo(self, msg):
        p = msg.payload
        with self._lock:
            self._battery   = float(p.get("battery_frac", 1.0))
            self._thermal_C = float(p.get("cpu_temp_C", 45.0))
            self._charging  = bool(p.get("battery_frac", 1.0) > 0.99)
        self._update_quant()

    def _on_da(self, msg):
        with self._lock: self._da = float(msg.payload.get("DA", 0.6))

    def _on_pfc(self, msg):
        with self._lock:
            self._pfc_action = msg.payload.get("action","stand_still")
            self._motor_on   = self._pfc_action not in ("stand_still","rest_posture","sleep")

    def _on_social(self, msg):
        with self._lock:
            self._social_ctx = {
                "nearby_person": msg.payload.get("name","stranger"),
                "bond_level":    msg.payload.get("bond_level", 0.0),
            }
            # Novel/significant social interaction → query LLM
            if float(msg.payload.get("bond_level",0)) > 0.3:
                name = msg.payload.get("name","someone")
                self._query_q.append({
                    "question": f"A person named {name} is nearby. How should I greet them?",
                    "priority": 2,
                })

    def _on_broca_speech(self, msg):
        """Speech act that needs elaboration → LLM."""
        utt = msg.payload.get("utterance","")
        if utt and len(utt) > 30:   # complex utterances only
            self._query_q.append({
                "question": f"I am about to say: '{utt}'. Is this appropriate and helpful?",
                "priority": 1,
            })

    def _on_emergency(self, msg):
        """Emergency situation → fast LLM query for advice."""
        etype = msg.payload.get("type","")
        if etype in ("low_battery","thermal_emergency"):
            self._query_q.insert(0, {
                "question": f"Emergency: {etype}. What should I do immediately?",
                "priority": 10,
            })

    def _update_quant(self):
        with self._lock:
            bat = self._battery; tc = self._thermal_C
            mot = self._motor_on; chrg = self._charging; da = self._da
        self.llm.update_system_state(bat, tc, mot, chrg, da)

    def _query_loop(self):
        while self._running:
            time.sleep(0.5)
            self.wd.heartbeat()

            with self._lock:
                q_copy = sorted(self._query_q, key=lambda x: -x.get("priority",0))
                self._query_q.clear()

            for item in q_copy[:1]:   # process one query per half-second
                ctx = {}
                with self._lock:
                    ctx["battery_pct"]  = self._battery * 100
                    ctx["temp_C"]       = self._thermal_C
                    ctx["da_level"]     = self._da
                    ctx.update(self._social_ctx)

                result = self.llm.query(item["question"], ctx, timeout_s=12.0)
                logger.info(f"LLM [{result['mode']} {result['model_B']}B]: "
                            f"'{item['question'][:50]}' → "
                            f"'{result['response'][:60]}' ({result['latency_ms']:.0f}ms)")
                self.wd.record_publish()

            # Periodic self-monitoring query (every 5 minutes)
            if int(time.time()) % 300 == 0:
                with self._lock: action = self._pfc_action
                self._query_q.append({
                    "question": f"I am currently doing '{action}'. Does this make sense given my goals?",
                    "priority": 0,
                })

    def _stats_loop(self):
        while self._running:
            time.sleep(60)
            s = self.llm.stats()
            self.bus.publish(b"CTX_LLM_STATS", {**s, "timestamp_ns": time.time_ns()})
            logger.info(f"LLM stats: mode={s['mode']} queries={s['n_queries']} "
                        f"avg_lat={s['avg_latency_ms']:.0f}ms "
                        f"commonsense={s['commonsense_pct']:.0f}%")

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.HYPO_STATE,      self._on_hypo)
        self.bus.subscribe(T.DA_VTA,          self._on_da)
        self.bus.subscribe(T.CTX_PFC_CMD,     self._on_pfc)
        self.bus.subscribe(T.SOCIAL_FACE,     self._on_social)
        self.bus.subscribe(T.BROCA_SPEECH_ACT, self._on_broca_speech)
        self.bus.subscribe(T.SYS_EMERGENCY,   self._on_emergency)
        self.wd.start()
        self._running = True
        threading.Thread(target=self._query_loop, daemon=True).start()
        threading.Thread(target=self._stats_loop, daemon=True).start()
        logger.info(f"{self.name} v5900 | 13B Q2 balanced | adaptive quant | common-sense")

    def stop(self):
        self._running = False
        self.llm.stop()
        self.bus.stop()


if __name__ == "__main__":
    with open("/etc/bubo/config.json") as f:
        # LLM runs on pfc-l (primary) or social node
        cfg = json.load(f).get("pfc_l", {"pub_port":5600,"sub_endpoints":[]})
    n = LLMNode(cfg)
    n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
