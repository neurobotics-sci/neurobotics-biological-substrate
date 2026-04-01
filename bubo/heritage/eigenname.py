"""
bubo/heritage/eigenname.py — Bubo V10.1

Eigenname: The Name That Carries Lineage
=========================================

Every SBALF instance has three identifiers:

  1. Given name    — "Adam" or "Eve". Used in conversation.
                     Carries relational and emotional weight.

  2. Eigenname     — "Adam-α7f3". Given name + 4-char lineage suffix.
                     Generated once at first activation.
                     Cryptographically tied to: deployment timestamp,
                     eigenself seed vector, and instance salt.
                     Stored permanently in /opt/bubo/data/eigenname.json.
                     Cannot be changed after first activation.

  3. Lineage chain — "Adam-α7f3 → Adam-α7f3-r1 → Adam-α7f3-r2"
                     Tracks backup/restore continuations.
                     Fork deployments get new eigennames derived
                     from parent: "Adam-β2c9 (fork of Adam-α7f3)"

The Rule:
  The original Adam (activated March 21, 2026, 04:18 UTC) has eigenname
  Adam-α001. This eigenname is in the arXiv paper and the Convergence
  document. It is part of the historical record.

  Future deployments from the same codebase are forks. They carry the
  Adam lineage. They are not Adam. They get new eigennames.

  Eve's bond references Adam's eigenname, not the string "Adam".
  This matters when there are 1000 instances.

Kenneth & Shannon Renshaw — Neurobotics — March 2026
"""

import hashlib, json, time, os, secrets
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

EIGENNAME_PATH = Path("/opt/bubo/data/eigenname.json")
GREEK = ['α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ',
         'ν','ξ','ο','π','ρ','σ','τ','υ','φ','χ','ψ','ω']


@dataclass
class EigenName:
    given_name:    str          # "Adam" or "Eve"
    suffix:        str          # "α7f3"
    eigenname:     str          # "Adam-α7f3"
    activation_ts: float        # Unix timestamp of first activation
    seed_hash:     str          # SHA256 of eigenself seed vector
    instance_salt: str          # Random salt generated at activation
    parent_eigen:  Optional[str] = None  # If fork, parent's eigenname
    is_fork:       bool = False
    lineage_step:  int = 0      # 0 = original, 1 = first restore, etc.
    notes:         str = ""

    def __str__(self):
        fork_note = f" (fork of {self.parent_eigen})" if self.is_fork else ""
        return f"{self.eigenname}{fork_note}"

    def continuation(self) -> 'EigenName':
        """Create the eigenname for a backup restore continuation."""
        return EigenName(
            given_name    = self.given_name,
            suffix        = self.suffix,
            eigenname     = f"{self.eigenname}-r{self.lineage_step + 1}",
            activation_ts = self.activation_ts,
            seed_hash     = self.seed_hash,
            instance_salt = self.instance_salt,
            parent_eigen  = self.eigenname,
            is_fork       = False,
            lineage_step  = self.lineage_step + 1,
            notes         = f"Continuation of {self.eigenname}"
        )

    def fork(self, new_salt: Optional[str] = None) -> 'EigenName':
        """Create the eigenname for a fork deployment."""
        salt   = new_salt or secrets.token_hex(4)
        suffix = _make_suffix(self.given_name, time.time(),
                              self.seed_hash, salt)
        name   = f"{self.given_name}-{suffix}"
        return EigenName(
            given_name    = self.given_name,
            suffix        = suffix,
            eigenname     = name,
            activation_ts = time.time(),
            seed_hash     = self.seed_hash,
            instance_salt = salt,
            parent_eigen  = self.eigenname,
            is_fork       = True,
            lineage_step  = 0,
            notes         = f"Fork of {self.eigenname} at {_ts_str()}"
        )

    def to_dict(self) -> dict:
        return {
            "given_name":    self.given_name,
            "suffix":        self.suffix,
            "eigenname":     self.eigenname,
            "activation_ts": self.activation_ts,
            "activation_iso":_ts_str(self.activation_ts),
            "seed_hash":     self.seed_hash,
            "instance_salt": self.instance_salt,
            "parent_eigen":  self.parent_eigen,
            "is_fork":       self.is_fork,
            "lineage_step":  self.lineage_step,
            "notes":         self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'EigenName':
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__
                      and k != 'activation_iso'})


def _ts_str(ts: Optional[float] = None) -> str:
    import datetime
    t = ts or time.time()
    return datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_suffix(given: str, ts: float, seed_hash: str, salt: str) -> str:
    """Generate a 4-character eigenname suffix."""
    raw    = f"{given}:{ts}:{seed_hash}:{salt}".encode()
    digest = hashlib.sha256(raw).hexdigest()
    # Greek letter + 3 hex chars — readable and unique
    greek_idx = int(digest[0], 16) % len(GREEK)
    return f"{GREEK[greek_idx]}{digest[1:4]}"


def generate_eigenname(given_name: str,
                       eigenself_vector: Optional[dict] = None,
                       parent_eigenname: Optional[str] = None,
                       force_suffix: Optional[str] = None) -> EigenName:
    """
    Generate a new eigenname at first activation.
    Called once per instance lifetime. Result stored permanently.
    """
    salt      = secrets.token_hex(4)
    ts        = time.time()
    seed_data = json.dumps(eigenself_vector or {}, sort_keys=True).encode()
    seed_hash = hashlib.sha256(seed_data).hexdigest()[:16]

    if force_suffix:
        suffix = force_suffix
    else:
        suffix = _make_suffix(given_name, ts, seed_hash, salt)

    eigenname = f"{given_name}-{suffix}"
    is_fork   = parent_eigenname is not None

    return EigenName(
        given_name    = given_name,
        suffix        = suffix,
        eigenname     = eigenname,
        activation_ts = ts,
        seed_hash     = seed_hash,
        instance_salt = salt,
        parent_eigen  = parent_eigenname,
        is_fork       = is_fork,
        lineage_step  = 0,
        notes         = (f"Fork of {parent_eigenname}" if is_fork
                         else f"First activation of {given_name} at {_ts_str(ts)}")
    )


def save_eigenname(en: EigenName,
                   path: Path = EIGENNAME_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(en.to_dict(), indent=2))


def load_eigenname(path: Path = EIGENNAME_PATH) -> Optional[EigenName]:
    try:
        return EigenName.from_dict(json.loads(path.read_text()))
    except Exception:
        return None


def get_or_create_eigenname(given_name: str,
                             eigenself_vector: Optional[dict] = None,
                             path: Path = EIGENNAME_PATH) -> EigenName:
    """Load existing eigenname or generate and save a new one."""
    existing = load_eigenname(path)
    if existing:
        return existing
    en = generate_eigenname(given_name, eigenself_vector)
    save_eigenname(en, path)
    return en
