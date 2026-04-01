"""
bubo/llm/edge_llm.py — Bubo v5900
Edge-LLM Runtime: adaptive quantization + common-sense reasoning on Orin 8GB.

════════════════════════════════════════════════════════════════════════
ARCHITECTURE: WHERE THE LLM FITS IN BUBO'S COGNITION
════════════════════════════════════════════════════════════════════════

The LLM serves as Bubo's "prefrontal common-sense oracle" — analogous to
the association cortex and anterior prefrontal cortex (BA10) that handle:
  - Novel situation interpretation
  - Goal decomposition ("how do I charge the battery?")
  - Social language understanding
  - Danger identification from verbal/contextual cues

The LLM is NOT involved in:
  - Real-time motor control (100Hz — too slow)
  - Fear responses (< 80ms via amygdala LA high-road)
  - Reflex arcs (< 30ms)
  - VOR (5ms)

The LLM IS involved in:
  - Interpreting ambiguous social situations (State S06 SOCIAL_INTERACT)
  - Generating extended speech acts (Broca node delegates to LLM for complex)
  - Planning novel multi-step tasks ("make me a cup of tea")
  - Self-monitoring: is my behaviour coherent with my goals?

LATENCY BUDGET (BALANCED mode, 13B Q2):
  Typical query: 50-token prompt + 20-token response
  Prefill:        50 tokens / (6 tok/s × 3×) = ~2.7s (prefill faster than gen)
  Generation:     20 tokens / 6 tok/s = 3.3s
  Overhead:       0.15s
  Total:          ~6s response latency

This is acceptable for non-real-time tasks. The LLM runs asynchronously
and publishes its output on T.CTX_LLM_RESPONSE when ready. PFC uses it
as advisory context, not hard constraint.

MEMORY MAP ON ORIN NANO 8GB:
  OS + drivers:         ~1.5 GB
  LLM model (Q2 13B):   ~3.4 GB
  KV cache (512 ctx):   ~0.2 GB
  Other Bubo process:   ~0.8 GB
  Available buffer:     ~2.1 GB
  Total:                ~8.0 GB ✓

SYSTEM PROMPT (Bubo's self-concept):
  "You are Bubo, a research humanoid robot. You have a body with arms, legs,
   eyes, and tactile sensors. Your goal is to be helpful, safe, and curious.
   Answer briefly and practically. If you are uncertain, say so.
   Context about your current state will be provided in each query."
"""

import subprocess, time, logging, threading, json
import numpy as np
from pathlib import Path
from typing import Optional
from bubo.shared.quantization.adaptive_quantization import AdaptiveQuantizationManager, QuantMode, QuantProfile, PROFILES, build_llama_cpp_args, estimate_generation_latency_ms

logger = logging.getLogger("EdgeLLM")

LLAMA_SERVER_BIN = "/usr/local/bin/llama-server"
LLAMA_CLI_BIN    = "/usr/local/bin/llama-cli"
MODEL_BASE_PATH  = Path("/opt/bubo/models")
LLM_SERVER_PORT  = 8080
LLM_SERVER_HOST  = "127.0.0.1"

BUBO_SYSTEM_PROMPT = """You are Bubo, a research humanoid robot assistant.
You have a body with arms, legs, stereo vision, tactile sensors, and a microphone.
You navigate the world, recognise faces, pick up objects, and interact socially.
Answer questions briefly and practically (1-3 sentences max).
State when you are uncertain. Current sensor context follows each question."""


class LLMServerProcess:
    """
    Manages a llama.cpp server subprocess.
    Provides REST API at http://127.0.0.1:8080/completion
    """

    def __init__(self, profile: QuantProfile):
        self._profile = profile
        self._proc: Optional[subprocess.Popen] = None
        self._running = False
        self._ready   = False
        self._port    = LLM_SERVER_PORT

    def start(self) -> bool:
        """Launch llama-server subprocess. Returns True if started successfully."""
        model_path = MODEL_BASE_PATH / self._profile.mode.name.lower()

        # Find GGUF file
        gguf_candidates = list(MODEL_BASE_PATH.glob("*.gguf"))
        if not gguf_candidates:
            logger.warning(f"No GGUF models found in {MODEL_BASE_PATH}. "
                           f"Simulation mode active.")
            self._ready   = True   # simulation: pretend ready
            self._running = True
            return True

        # Select best matching model for profile
        size_key = f"{self._profile.model_size_B}b"
        bit_key  = f"q{self._profile.bits_ffn}_k"
        model    = None
        for c in gguf_candidates:
            if size_key in c.name.lower() and bit_key in c.name.lower():
                model = c; break
        if model is None:
            model = gguf_candidates[0]  # fallback to any model
            logger.warning(f"Exact model not found, using {model.name}")

        cmd = [
            LLAMA_SERVER_BIN if Path(LLAMA_SERVER_BIN).exists() else "llama-server",
            "--model",         str(model),
            "--n-gpu-layers",  "99",
            "--ctx-size",      str(self._profile.context_len),
            "--threads",       "4",
            "--batch-size",    "512",
            "--port",          str(self._port),
            "--host",          LLM_SERVER_HOST,
            "--mlock",
            "--log-disable",
        ]

        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True)
            self._running = True

            # Wait for server to be ready (max 30s)
            for _ in range(60):
                time.sleep(0.5)
                try:
                    import urllib.request
                    urllib.request.urlopen(
                        f"http://{LLM_SERVER_HOST}:{self._port}/health", timeout=1)
                    self._ready = True
                    logger.info(f"LLM server ready: {model.name} @ {self._profile.tokens_per_s:.0f} tok/s")
                    return True
                except Exception:
                    pass

            logger.error("LLM server did not become ready within 30s")
            return False

        except FileNotFoundError:
            logger.warning("llama-server not installed. Simulation mode.")
            self._ready = True; self._running = True
            return True
        except Exception as e:
            logger.error(f"LLM server start failed: {e}")
            return False

    def stop(self):
        if self._proc:
            self._proc.terminate()
            try: self._proc.wait(timeout=5)
            except: self._proc.kill()
        self._running = False; self._ready = False

    @property
    def is_ready(self) -> bool: return self._ready


class EdgeLLMEngine:
    """
    Main Edge-LLM engine with adaptive quantization.
    Runs on Social node (192.168.1.19) or PFC-L (192.168.1.10).
    Publishes T.CTX_LLM_RESPONSE when reasoning complete.
    """

    def __init__(self, bus=None):
        self._bus        = bus
        self._quant_mgr  = AdaptiveQuantizationManager()
        self._server:    Optional[LLMServerProcess] = None
        self._lock       = threading.Lock()
        self._pending_q: list = []
        self._running    = False
        self._n_queries  = 0
        self._total_lat  = 0.0

        # Start with BALANCED mode (13B Q2)
        self._start_server(QuantMode.BALANCED)

    def _start_server(self, mode: QuantMode):
        """(Re)start LLM server with new quantization mode."""
        if self._server:
            self._server.stop()
        profile = PROFILES[mode]
        self._server = LLMServerProcess(profile)
        ok = self._server.start()
        if ok:
            logger.info(f"LLM: {profile.model_size_B}B Q{profile.bits_ffn} "
                        f"({profile.ram_gb:.1f}GB RAM, "
                        f"~{estimate_generation_latency_ms(profile):.0f}ms latency)")
        return ok

    def update_system_state(self, battery_frac: float, thermal_C: float,
                             motor_active: bool, charging: bool, da_level: float):
        """Called by hypothalamus/insula with current system state."""
        self._quant_mgr.update_state(battery_frac, thermal_C, motor_active, charging, da_level)
        new_mode = self._quant_mgr.step()
        if new_mode is not None:
            logger.info(f"LLM: reloading model for mode {new_mode.value}")
            threading.Thread(target=self._start_server, args=(new_mode,), daemon=True).start()

    def query(self, question: str, context: dict = None, timeout_s: float = 15.0) -> dict:
        """
        Synchronous query to LLM with sensor context injection.
        Returns: {"response": str, "latency_ms": float, "mode": str, "tokens": int}
        """
        if not self._server or not self._server.is_ready:
            return {"response": "[LLM not ready]", "latency_ms": 0, "mode": "offline", "tokens": 0}

        # Build context-enriched prompt
        ctx_str = ""
        if context:
            ctx_parts = []
            if "battery_pct" in context:
                ctx_parts.append(f"Battery: {context['battery_pct']:.0f}%")
            if "temp_C" in context:
                ctx_parts.append(f"CPU temp: {context['temp_C']:.0f}°C")
            if "da_level" in context:
                ctx_parts.append(f"Motivation: {'high' if context['da_level'] > 0.6 else 'low'}")
            if "location" in context:
                ctx_parts.append(f"Location: {context['location']}")
            if "nearby_person" in context:
                ctx_parts.append(f"Nearby: {context['nearby_person']}")
            if ctx_parts:
                ctx_str = f"\n[Context: {', '.join(ctx_parts)}]"

        prompt = f"{BUBO_SYSTEM_PROMPT}\n\nQuestion: {question}{ctx_str}\nAnswer:"

        t0 = time.time()
        response_text = self._call_llama(prompt, timeout_s)
        lat_ms = (time.time() - t0) * 1000

        self._n_queries += 1
        self._total_lat += lat_ms

        result = {
            "response":      response_text.strip(),
            "latency_ms":    round(lat_ms, 0),
            "mode":          self._quant_mgr.current_mode.value,
            "model_B":       self._quant_mgr.current_profile.model_size_B,
            "tokens_per_s":  self._quant_mgr.current_profile.tokens_per_s,
            "timestamp_ns":  time.time_ns(),
        }

        if self._bus:
            self._bus.publish(b"CTX_LLM_RESP", result)

        return result

    def _call_llama(self, prompt: str, timeout_s: float) -> str:
        """Call llama-server REST API or CLI."""
        try:
            import urllib.request, urllib.error
            payload = json.dumps({
                "prompt":           prompt,
                "n_predict":        48,
                "temperature":      0.2,
                "repeat_penalty":   1.1,
                "stop":             ["\n\n", "Question:", "[Context:"],
            }).encode()

            req = urllib.request.Request(
                f"http://{LLM_SERVER_HOST}:{LLM_SERVER_PORT}/completion",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST")

            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                resp = json.loads(r.read().decode())
                return resp.get("content", "[empty response]")

        except Exception as e:
            # Simulation fallback
            logger.debug(f"LLM call failed ({e}) — simulation response")
            return self._simulate_response(prompt)

    def _simulate_response(self, prompt: str) -> str:
        """Generate a canned simulation response based on prompt keywords."""
        p = prompt.lower()
        if "battery" in p or "charge" in p:
            return "I should navigate to the charging station and dock."
        if "danger" in p or "obstacle" in p:
            return "I will stop and assess the obstacle before proceeding."
        if "hello" in p or "hi " in p:
            return "Hello! I am Bubo. How can I help you?"
        if "pick up" in p or "grasp" in p:
            return "I will approach the object, open my hand, and close fingers around it."
        if "warm" in p or "hot" in p or "temperature" in p:
            return "My CPU temperature is elevated. Reducing motor activity to cool down."
        if "tired" in p or "sleep" in p:
            return "My adenosine levels are high. I will enter a safe rest posture."
        return "I understand. Let me think about the best course of action."

    def stats(self) -> dict:
        profile = self._quant_mgr.current_profile
        return {
            "mode":          self._quant_mgr.current_mode.value,
            "model_B":       profile.model_size_B,
            "bits_ffn":      profile.bits_ffn,
            "ram_gb":        profile.ram_gb,
            "n_queries":     self._n_queries,
            "avg_latency_ms": round(self._total_lat / max(self._n_queries, 1), 0),
            "commonsense_pct": profile.commonsense_pct,
        }

    def stop(self):
        self._running = False
        if self._server: self._server.stop()
