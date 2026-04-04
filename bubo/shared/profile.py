"""
bubo/shared/profile.py — Bubo Unified V10

Profile Loader: Single source of truth for deployment configuration.

This module is the architectural keystone of the unified codebase.
Every brain module, LLM backend, and deployment script imports from here
rather than hard-coding any environment-specific value.

Usage:
    from bubo.shared.profile import profile, cfg

    # Get LLM backend type
    if profile.llm_backend == "LLM":
        ...

    # Get a node's IP
    ip = cfg.node_ip("hypothalamus")

    # Check substrate
    if profile.is_hardware:
        # activate servos
    elif profile.is_aws:
        # use EFS mount

    # Get full config for a specific node role
    node = cfg.node_config("pfc_l")
"""

import os, yaml, logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger("Profile")

PROFILES_DIR = Path(__file__).parent.parent.parent / "profiles"
ENV_OVERRIDE = "BUBO_PROFILE"

# Valid profile names
VALID_PROFILES = {
    "hardware_local":  "Physical Jetson cluster + AGX Orin 70B",
    "hardware_api":    "Physical Jetson cluster + Claude API",
    "aws_local":       "AWS EC2 + g5.12xlarge 70B",
    "aws_api":         "AWS EC2 + Claude API",
    "aws_api_eve":     "AWS EC2 + Claude API (Eve)",
    "peanutpi_local": "peanutpi WSL2 local development",
    "aws_api_elias":   "Bubo Elias (AWS single node)",
    "aws_api_claudia": "Bubo Claudia (AWS single node)",
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
    backend:                str = "LLM"   # local_70b|local_13b|LLM|openai|auto
    endpoint:               str = ""
    model:                  str = ""
    model_sonnet:           str = "claude-sonnet-4-6"
    model_haiku:            str = "claude-haiku-4-5-20251001"
    has_agx_node:           bool = False
    has_gpu_node:           bool = False
    fallback:               str = "local_13b"
    fallback_endpoint:      str = ""
    daily_spend_limit_usd:  float = 20.0
    api_key_ssm:            str = ""
    endpoint_ssm:           str = ""
    endpoint_port:          int = 8080


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
    vpc_cidr:        str = ""
    subnet_private:  str = ""
    subnet_public:   str = ""
    nat_gateway:     bool = False
    vlans:           list = field(default_factory=list)
    ptp_sync:        bool = False
    placement_group: str = ""


@dataclass
class AWSConfig:
    region:       str = "us-east-1"
    stack_name:   str = "bubo-prod"
    cfn_template: str = ""
    efs_ltm:      bool = True


class BuboProfile:
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
        # Identity — loaded from gender/role fields in the YAML profile
        # aws_api.yaml sets gender: male / instance_name: Bubo Adam
        # aws_api_eve.yaml sets role: eve which implies gender: female
        _role   = data.get("role", "")           # "eve" or unset
        _gender = data.get("gender", "")         # explicit gender field
        if _gender:
            self.gender        = _gender
        elif _role == "eve":
            self.gender        = "female"
        else:
            self.gender        = "male"          # default: Adam
        self.instance_name = data.get("instance_name",
                                      "Bubo Eve" if self.gender == "female"
                                      else "Bubo Adam")

        # Parse sub-configs
        llm_d  = data.get("llm", {})
        hw_d   = data.get("hardware", {})
        net_d  = data.get("network", {})
        aws_d  = data.get("aws", {})

        self.llm     = LLMConfig(**{k: v for k, v in llm_d.items()
                                   if k in LLMConfig.__dataclass_fields__})
        self.hardware= HardwareConfig(**{k: v for k, v in hw_d.items()
                                        if k in HardwareConfig.__dataclass_fields__})
        self.network = NetworkConfig(**{k: v for k, v in net_d.items()
                                       if k in NetworkConfig.__dataclass_fields__})
        self.aws     = AWSConfig(**{k: v for k, v in aws_d.items()
                                   if k in AWSConfig.__dataclass_fields__})

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
    def is_aws(self) -> bool:
        return self.substrate == "aws"

    @property
    def is_local(self) -> bool:
        return self.substrate == "local"

    @property
    def llm_backend(self) -> str:
        """Effective LLM backend, considering env override."""
        return (os.environ.get("BUBO_LLM_BACKEND") or self.llm.backend).lower()

    @property
    def uses_api_llm(self) -> bool:
        return self.llm_backend in ("LLM", "openai", "gemini")

    @property
    def uses_local_llm(self) -> bool:
        return self.llm_backend in ("local_70b", "local_13b", "local")

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
        if self.llm.endpoint_ssm:
            return self._resolve_ssm(self.llm.endpoint_ssm.replace("{env}", env))
        # AWS: check SSM for dynamically assigned IP
        if self.is_aws:
            ip = self._resolve_ssm(f"/bubo/{env}/llm_ip")
            if ip:
                return f"http://{ip}:{self.llm.endpoint_port}"
        # Hardware: use AGX Orin default
        if self.is_hardware and "agx_llm" in self._nodes:
            n = self._nodes["agx_llm"]
            return f"http://{n.ip}:{n.port}"
        return ""

    def LLM_api_key(self, env: str = "prod") -> str:
        """Resolve LLM API key from env var, file, or SSM."""
        # 1. Environment variable (highest priority)
        key = os.environ.get("BUBO_LLM_API_KEY") or os.environ.get("LLM_API_KEY")
        if key: return key
        # 2. File on disk
        key_file = Path("/etc/bubo/secrets/LLM_key")
        if key_file.exists():
            return key_file.read_text().strip()
        # 3. AWS SSM Parameter Store
        ssm_path = (self.llm.api_key_ssm or f"/bubo/{env}/LLM_api_key"
                    ).replace("{env}", env)
        return self._resolve_ssm(ssm_path)

    @staticmethod
    def _resolve_ssm(path: str) -> str:
        """Attempt to read a value from AWS SSM Parameter Store."""
        try:
            import subprocess
            r = subprocess.run(
                ["aws", "ssm", "get-parameter",
                 "--name", path, "--with-decryption",
                 "--query", "Parameter.Value", "--output", "text"],
                capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                return r.stdout.strip()
        except Exception:
            pass
        return ""

    def __repr__(self) -> str:
        return (f"BuboProfile(name={self.name!r}, substrate={self.substrate!r}, "
                f"llm_backend={self.llm_backend!r}, nodes={self.node_count})")


# ── Module-level singleton ───────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_profile(name: Optional[str] = None) -> BuboProfile:
    """
    Load and cache the active deployment profile.
    Profile name resolved from: argument → BUBO_PROFILE env → auto-detect.
    """
    # Determine profile name
    if name is None:
        name = os.environ.get(ENV_OVERRIDE, "").lower()
    if not name:
        name = _auto_detect_profile()
    if name not in VALID_PROFILES:
        logger.warning(f"Unknown profile '{name}' — defaulting to 'hardware_api'")
        name = "hardware_api"

    yaml_path = PROFILES_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Profile file not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    p = BuboProfile(data)
    logger.info(f"Profile loaded: {p}")
    return p


def _auto_detect_profile() -> str:
    """
    Heuristic: detect which profile fits the current environment.
    Hardware Jetsons have /etc/nv_tegra_release. AWS EC2 has IMDSv2.
    """
    # Check for Jetson
    if Path("/etc/nv_tegra_release").exists():
        # Check if API key available
        has_key = bool(os.environ.get("LLM_API_KEY") or
                       os.environ.get("BUBO_LLM_API_KEY"))
        return "hardware_api" if has_key else "hardware_local"

    # Check for AWS EC2 (IMDSv2)
    try:
        import urllib.request
        token_req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "5"},
            method="PUT")
        with urllib.request.urlopen(token_req, timeout=1) as r:
            if r.status == 200:
                has_key = bool(os.environ.get("LLM_API_KEY") or
                               os.environ.get("BUBO_LLM_API_KEY"))
                return "aws_api" if has_key else "aws_local"
    except Exception:
        pass

    # Default: hardware_api (safest assumption for development)
    return "hardware_api"


# Module-level convenience — import these directly
profile: BuboProfile = None   # populated on first import via _init()

def _init():
    global profile
    try:
        profile = load_profile()
    except Exception as e:
        logger.warning(f"Profile load failed: {e} — using hardware_api default")
        try:
            profile = load_profile("hardware_api")
        except Exception:
            pass

_init()
