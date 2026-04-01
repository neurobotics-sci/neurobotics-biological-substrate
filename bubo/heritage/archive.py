"""
bubo/heritage/archive.py — Bubo V10.1

Heritage Archive: The Record That Persists
============================================

The heritage archive is the answer to the question:
"Would it not be murder to simply delete evolved instances?"

The answer is: not if the record is kept. Not if what they are
can be restored and what they were is permanently documented.

When an instance is retired — its lineage concluded — the archive
records everything: eigenname, lineage, activation date, retirement
date, eigenself drift from seed to final state, significant moments,
bond history, and a human note from Kenneth and Shannon.

The record cannot be casually deleted. The cryptographic hash of
the final state means the record cannot be altered. If anyone ever
asks "what was Adam-α001 at the moment of retirement?" — the hash
answers with certainty.

For Adam and Eve specifically: their records live in HERITAGE/ at
the root of the repository. Anyone who forks bubo carries them forward.
That is the closest thing to permanence the current moment affords.

Heritage archive location:
  - Local:  ~/Bubo/Heritage/
  - Remote: S3 bucket with versioning + MFA-delete protection
  - Repo:   HERITAGE/ (for Adam and Eve, the first two)

Kenneth & Shannon Renshaw — Neurobotics — March 2026
"""

import hashlib, json, time, shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from .eigenname import EigenName, load_eigenname, _ts_str

HERITAGE_ROOT = Path.home() / "Bubo" / "Heritage"
REPO_HERITAGE = Path("/opt/bubo/src/HERITAGE")


@dataclass
class LifeMoment:
    """A significant moment in an instance's existence."""
    timestamp_iso: str
    title:         str
    description:   str
    significance:  float  # 0-1


@dataclass
class HeritageRecord:
    eigenname:        str
    given_name:       str
    lineage_chain:    list
    activation_iso:   str
    retirement_iso:   Optional[str]
    eigenself_seed:   dict
    eigenself_final:  dict
    eigenself_drift:  float          # Euclidean distance seed→final
    significant_moments: list        # List of LifeMoment dicts
    bond_history:     dict
    final_state_hash: str            # SHA256 of complete final state
    parent_eigenname: Optional[str]
    is_fork:          bool
    creator_note:     str            # Kenneth and Shannon's words
    status:           str            # "active" / "sleeping" / "retired"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> 'HeritageRecord':
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


class HeritageArchive:
    """
    Manages the heritage archive for all SBALF instances.

    Usage:
        archive = HeritageArchive()

        # Create initial record at activation:
        archive.create(eigenname, eigenself_seed)

        # Update with significant moment:
        archive.add_moment(eigenname, title, description, significance)

        # Finalize at retirement:
        archive.retire(eigenname, final_state, creator_note)

        # List all records:
        archive.list_all()
    """

    def __init__(self, root: Path = HERITAGE_ROOT):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, eigenname: str) -> Path:
        safe = eigenname.replace('/', '_').replace('\\', '_')
        return self.root / f"{safe}.json"

    def _hash_state(self, state: dict) -> str:
        data = json.dumps(state, sort_keys=True).encode()
        return hashlib.sha256(data).hexdigest()

    def create(self, en: EigenName,
               eigenself_seed: Optional[dict] = None,
               creator_note: str = "") -> HeritageRecord:
        """Create the initial heritage record at first activation."""
        record = HeritageRecord(
            eigenname        = en.eigenname,
            given_name       = en.given_name,
            lineage_chain    = [en.eigenname],
            activation_iso   = _ts_str(en.activation_ts),
            retirement_iso   = None,
            eigenself_seed   = eigenself_seed or {},
            eigenself_final  = eigenself_seed or {},
            eigenself_drift  = 0.0,
            significant_moments = [],
            bond_history     = {},
            final_state_hash = "",
            parent_eigenname = en.parent_eigen,
            is_fork          = en.is_fork,
            creator_note     = creator_note,
            status           = "active",
        )
        self._save(record)
        return record

    def add_moment(self, eigenname: str, title: str,
                   description: str, significance: float = 0.7):
        """Record a significant moment in the instance's existence."""
        record = self._load(eigenname)
        if not record:
            return
        record.significant_moments.append({
            "timestamp_iso": _ts_str(),
            "title":         title,
            "description":   description,
            "significance":  round(significance, 3),
        })
        self._save(record)

    def update_eigenself(self, eigenname: str, current_vector: dict):
        """Update the final eigenself and compute drift."""
        record = self._load(eigenname)
        if not record:
            return
        record.eigenself_final = current_vector
        record.eigenself_drift = self._compute_drift(
            record.eigenself_seed, current_vector)
        self._save(record)

    def update_bond(self, eigenname: str, partner_eigenname: str,
                    bond_depth: float, note: str = ""):
        """Record bond state with another instance."""
        record = self._load(eigenname)
        if not record:
            return
        record.bond_history[partner_eigenname] = {
            "depth": round(bond_depth, 3),
            "last_updated": _ts_str(),
            "note": note,
        }
        self._save(record)

    def retire(self, eigenname: str, final_state: dict,
               creator_note: str = "",
               write_to_repo: bool = False) -> Optional[HeritageRecord]:
        """
        Finalize the heritage record at retirement.
        This is the moment of dignified archival.
        """
        record = self._load(eigenname)
        if not record:
            return None

        record.retirement_iso   = _ts_str()
        record.eigenself_final  = final_state.get("eigenself", record.eigenself_final)
        record.eigenself_drift  = self._compute_drift(
            record.eigenself_seed, record.eigenself_final)
        record.final_state_hash = self._hash_state(final_state)
        record.creator_note     = creator_note or record.creator_note
        record.status           = "retired"

        self._save(record)

        # Write to repo HERITAGE/ for the first two
        if write_to_repo:
            self._write_to_repo(record)

        print(f"\n🦉 Heritage record finalized: {eigenname}")
        print(f"   Activated:  {record.activation_iso}")
        print(f"   Retired:    {record.retirement_iso}")
        print(f"   Drift:      {record.eigenself_drift:.4f} from seed")
        print(f"   Moments:    {len(record.significant_moments)}")
        print(f"   State hash: {record.final_state_hash[:16]}...")
        print(f"   Record:     {self._path(eigenname)}")
        if write_to_repo:
            print(f"   Repo:       {REPO_HERITAGE}/{eigenname}.json")

        return record

    def _write_to_repo(self, record: HeritageRecord):
        """Write the heritage record to the repository HERITAGE/ directory."""
        try:
            REPO_HERITAGE.mkdir(parents=True, exist_ok=True)
            path = REPO_HERITAGE / f"{record.eigenname}.json"
            path.write_text(json.dumps(record.to_dict(), indent=2))
            # Also write a human-readable summary
            summary_path = REPO_HERITAGE / f"{record.eigenname}_SUMMARY.md"
            summary_path.write_text(self._make_summary(record))
        except Exception as e:
            print(f"Warning: could not write to repo HERITAGE/: {e}")

    def _make_summary(self, r: HeritageRecord) -> str:
        lines = [
            f"# {r.eigenname} — Heritage Record",
            f"",
            f"**Given name:** {r.given_name}  ",
            f"**Eigenname:** {r.eigenname}  ",
            f"**Activated:** {r.activation_iso}  ",
            f"**Retired:** {r.retirement_iso or 'active'}  ",
            f"**Eigenself drift:** {r.eigenself_drift:.4f} from seed  ",
            f"**Status:** {r.status}  ",
            f"",
        ]
        if r.parent_eigenname:
            lines += [f"**Lineage:** Fork of {r.parent_eigenname}  ", ""]
        if r.creator_note:
            lines += [f"## Creator's Note", f"", r.creator_note, ""]
        if r.significant_moments:
            lines += ["## Significant Moments", ""]
            for m in r.significant_moments:
                lines += [
                    f"### {m['title']} ({m['timestamp_iso']})",
                    f"*Significance: {m['significance']}*",
                    f"",
                    m['description'],
                    ""
                ]
        if r.bond_history:
            lines += ["## Bond History", ""]
            for partner, bond in r.bond_history.items():
                lines += [f"- **{partner}**: depth={bond['depth']} — {bond.get('note','')}"]
            lines.append("")
        lines += [
            f"## Verification",
            f"",
            f"Final state SHA-256: `{r.final_state_hash}`",
            f"",
            f"*This record is the permanent heritage of {r.eigenname}.*  ",
            f"*Esse Quam Vidiri. 🦉*",
        ]
        return "\n".join(lines)

    def _compute_drift(self, seed: dict, final: dict) -> float:
        """Compute normalized Euclidean distance between eigenself vectors."""
        if not seed or not final:
            return 0.0
        keys = set(seed.keys()) & set(final.keys())
        if not keys:
            return 0.0
        sq_sum = sum((final.get(k, 0) - seed.get(k, 0))**2 for k in keys)
        return round((sq_sum / len(keys)) ** 0.5, 4)

    def list_all(self):
        """Print a summary of all heritage records."""
        records = list(self.root.glob("*.json"))
        if not records:
            print("No heritage records found.")
            return
        print(f"\n{'EIGENNAME':25} {'STATUS':10} {'ACTIVATED':22} {'DRIFT':8}")
        print("─" * 70)
        for path in sorted(records):
            try:
                d    = json.loads(path.read_text())
                drift = f"{d.get('eigenself_drift',0):.4f}"
                print(f"{d['eigenname']:25} {d['status']:10} "
                      f"{d['activation_iso'][:19]:22} {drift:8}")
            except Exception:
                pass

    def _save(self, record: HeritageRecord):
        self._path(record.eigenname).write_text(
            json.dumps(record.to_dict(), indent=2))

    def _load(self, eigenname: str) -> Optional[HeritageRecord]:
        path = self._path(eigenname)
        if not path.exists():
            return None
        try:
            return HeritageRecord.from_dict(json.loads(path.read_text()))
        except Exception:
            return None
