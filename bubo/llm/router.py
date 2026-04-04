#!/usr/bin/env python3
# =============================================================================
# bubo/llm/router.py — Bubo v1.0 Sterile Release
# Neurobotics — SBALF Cognitive Routing Layer
# =============================================================================

import os
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger("bubo.llm.router")

# Sterile baseline prompt for research platform
BUBO_SYSTEM = """You are Bubo, an open-source research humanoid robot assistant.
Your goal is to be helpful, safe, and curious. Answer questions briefly and practically.
When asked about the physical world, reason step-by-step. State when you are uncertain.
Context about your current sensor and system state will be provided in each query."""

# ─── 1. Abstract Provider Interface ──────────────────────────────────────────

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, use_fast_model: bool = False) -> str:
        pass

# ─── 2. Concrete Implementations ─────────────────────────────────────────────

class LLMAdapter(LLMProvider):
    def __init__(self, api_key: str, primary_model: str, fast_model: str):
        import LLM
        self.client = LLM.LLM(api_key=api_key)
        self.primary_model = primary_model
        self.fast_model = fast_model

    def generate(self, prompt: str, max_tokens: int, use_fast_model: bool = False) -> str:
        model = self.fast_model if use_fast_model else self.primary_model
        system_prompt = BUBO_SYSTEM
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

class OpenAICompatibleAdapter(LLMProvider):
    def __init__(self, base_url: str, api_key: str, primary_model: str, fast_model: str):
        import openai
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.primary_model = primary_model
        self.fast_model = fast_model

    def generate(self, prompt: str, max_tokens: int, use_fast_model: bool = False) -> str:
        model = self.fast_model if use_fast_model else self.primary_model
        response = self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": BUBO_SYSTEM},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

# ─── 3. The Router / Factory ─────────────────────────────────────────────────

class CognitiveRouter:
    def __init__(self, config_path: str = "/etc/bubo/llm_config.json"):
        self.config = self._load_config(config_path)
        self.provider = self._initialize_provider()

    def _load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            return {"provider": "local_openai_compatible", "base_url": "http://127.0.0.1:8080/v1"}
        with open(path, 'r') as f:
            return json.load(f)

    def _initialize_provider(self) -> LLMProvider:
        provider_type = self.config.get("provider", "local_openai_compatible").lower()
        api_key = os.getenv("BUBO_API_KEY", "sk-dummy-key-for-local")
        
        if provider_type == "LLM":
            return LLMAdapter(
                api_key=api_key,
                primary_model=self.config.get("primary_model", "XXXXXXXXXXXX"),
                fast_model=self.config.get("fast_model", "XXXXXXXXXXXX")
            )
        elif provider_type in ["openai", "local_openai_compatible", "llama_cpp"]:
            return OpenAICompatibleAdapter(
                base_url=self.config.get("base_url"),
                api_key=api_key,
                primary_model=self.config.get("primary_model", "local-70b"),
                fast_model=self.config.get("fast_model", "local-13b")
            )
        raise ValueError(f"Unknown LLM provider: {provider_type}")

    def route_task(self, prompt: str, bloom_level: int, max_tokens: int = 400) -> str:
        use_fast_model = bloom_level <= 3
        return self.provider.generate(prompt, max_tokens, use_fast_model)
