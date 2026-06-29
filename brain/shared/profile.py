"""
brain/shared/profile.py — Brain Unified V10 (Air-Gapped Edition)

Profile Loader: Single source of truth for deployment configuration.

This module is the architectural keystone of the unified codebase.
Every brain module, LLM backend, and deployment script imports from here
rather than hard-coding any environment-specific value.

Usage:
    from brain.shared.profile import profile

    # Get LLM backend type
    if profile.llm_backend == "local_70b":
        ...

    # Get a node's IP
    ip = profile.node_ip("hypothalamus")

    # Check substrate
    if profile.is_hardware:
        # activate servos

    # Get full config for a specific node role
    node = profile.node_config("pfc_l")
"""

import os, yaml, logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger("Profile")

PROFILES_DIR = Path(__file__).parent.parent.parent / "profiles"
ENV_OVERRIDE = "BRAIN_PROFILE"

# Valid profile names (Sovereign OSS Only)
VALID_PROFILES = {
    "local":           "Local RTX 4060 8B",  # <--- YOUR VIP PASS
    "hardware_local":  "Physical Jetson cluster + AGX Orin 70B",
}


@dataclass
class NodeConfig:
    name:     str
    role:     str
    port:     int
    ip:       str = ""
    instance: str = ""
    hw:       str = ""
    tier:     int = 0


@dataclass
class LLMConfig:
    backend:           str = "local_70b"   # local_70b|local_13b
    endpoint:          str = ""
    model:             str = ""
    has_agx_node:      bool = False
    has_gpu_node:      bool = False
    fallback:          str = "local_13b"
    fallback_endpoint: str = ""
    endpoint_port:     int = 8080


@dataclass
class HardwareConfig:
    servos:            bool = False
    gpio:              bool = False
    galvanic_barrier:  bool = False
    vagus_nerve:       bool = False
    stm32:             bool = False
    preempt_rt:        bool = False


@dataclass
class NetworkConfig:
    subnet:          str = ""
    vlans:           list = field(default_factory=list)
    ptp_sync:        bool = False


class BrainProfile:
    """
    Immutable deployment profile.
    Loaded once at startup from YAML + environment variables.
    """

    def __init__(self, data: dict):
        self._data = data
        self.name          = data["name"]
        self.description   = data.get("description", "")
        self.version       = data.get("version", "9000")
        self.substrate     = data.get("substrate", "hardware")

        # Identity
        _role   = data.get("role", "")
        _gender = data.get("gender", "")
        if _gender:
            self.gender        = _gender
        elif _role == "eve":
            self.gender        = "female"
        else:
            self.gender        = "male"          # default: Adam
        self.instance_name = data.get("instance_name",
                                      "Brain Eve" if self.gender == "female"
                                      else "Brain Adam")

        # Parse sub-configs
        llm_d  = data.get("llm", {})
        hw_d   = data.get("hardware", {})
        net_d  = data.get("network", {})

        self.llm     = LLMConfig(**{k: v for k, v in llm_d.items()
                                   if k in LLMConfig.__dataclass_fields__})
        self.hardware= HardwareConfig(**{k: v for k, v in hw_d.items()
                                        if k in HardwareConfig.__dataclass_fields__})
        self.network = NetworkConfig(**{k: v for k, v in net_d.items()
                                       if k in NetworkConfig.__dataclass_fields__})

        # Parse nodes
        self._nodes: Dict[str, NodeConfig] = {}
        for node_name, nd in data.get("nodes", {}).items():
            self._nodes[node_name] = NodeConfig(
                name=node_name,
                role=nd.get("role", node_name),
                port=nd.get("port", 5600),
                ip=nd.get("ip", ""),
                instance=nd.get("instance", ""),
                hw=nd.get("hw", ""),
                tier=nd.get("tier", 0),
            )

    # ── Convenience properties ───────────────────────────────────────────────

    @property
    def is_hardware(self) -> bool:
        return self.substrate == "hardware"

    @property
    def is_local(self) -> bool:
        return self.substrate == "local"

    @property
    def llm_backend(self) -> str:
        """Effective LLM backend, considering env override."""
        return (os.environ.get("BRAIN_LLM_BACKEND") or self.llm.backend).lower()

    @property
    def has_llm_node(self) -> bool:
        """True if a dedicated LLM node exists in this profile."""
        return self.llm.has_agx_node or self.llm.has_gpu_node

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    # ── Node access ──────────────────────────────────────────────────────────

    def node_ip(self, name: str) -> str:
        """Get IP for a named node. Empty string if node not in this profile."""
        node = self._nodes.get(name)
        return node.ip if node else ""

    def node_port(self, name: str) -> int:
        node = self._nodes.get(name)
        return node.port if node else 5600

    def node_config(self, name: str) -> Optional[NodeConfig]:
        return self._nodes.get(name)

    def all_nodes(self) -> Dict[str, NodeConfig]:
        return dict(self._nodes)

    def nodes_by_tier(self, tier: int):
        return {n: c for n, c in self._nodes.items() if c.tier == tier}

    def node_endpoint(self, name: str) -> str:
        """ZMQ endpoint string for a named node."""
        node = self._nodes.get(name)
        if not node: return ""
        return f"tcp://{node.ip}:{node.port}"

    def all_sub_endpoints(self, exclude: str = "") -> list:
        """All ZMQ endpoints except the named one (for subscriber setup)."""
        return [f"tcp://{n.ip}:{n.port}"
                for name, n in self._nodes.items()
                if name != exclude and n.ip]

    # ── LLM endpoint resolution ──────────────────────────────────────────────

    def llm_endpoint(self, env: str = "prod") -> str:
        """Resolve the LLM endpoint URL."""
        if self.llm.endpoint:
            return self.llm.endpoint
        # Hardware: use AGX Orin default
        if self.is_hardware and "agx_llm" in self._nodes:
            n = self._nodes["agx_llm"]
            return f"http://{n.ip}:{n.port}"
        return ""

    def __repr__(self) -> str:
        return (f"BrainProfile(name={self.name!r}, substrate={self.substrate!r}, "
                f"llm_backend={self.llm_backend!r}, nodes={self.node_count})")


# ── Module-level singleton ───────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_profile(name: Optional[str] = None) -> BrainProfile:
    """
    Load and cache the active deployment profile.
    Profile name resolved from: argument → BRAIN_PROFILE env → auto-detect.
    """
    # Determine profile name
    if name is None:
        name = os.environ.get(ENV_OVERRIDE, "").lower()
    if not name:
        name = _auto_detect_profile()
    if name not in VALID_PROFILES:
        logger.warning(f"Unknown profile '{name}' — defaulting to 'hardware_local'")
        name = "hardware_local"

    yaml_path = PROFILES_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Profile file not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    p = BrainProfile(data)
    logger.info(f"Profile loaded: {p}")
    return p


def _auto_detect_profile() -> str:
    """
    Heuristic: detect which profile fits the current environment.
    Hardware Jetsons have /etc/nv_tegra_release. 
    """
    # Check for Jetson
    if Path("/etc/nv_tegra_release").exists():
        return "hardware_local"

    # Default safest assumption for local development
    return "local"


# Module-level convenience — import these directly
profile: BrainProfile = None   # populated on first import via _init()

def _init():
    global profile
    try:
        profile = load_profile()
    except Exception as e:
        logger.warning(f"Profile load failed: {e} — using hardware_local default")
        try:
            profile = load_profile("hardware_local")
        except Exception:
            pass

_init()
