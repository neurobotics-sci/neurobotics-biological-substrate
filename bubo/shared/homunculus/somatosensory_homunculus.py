"""
bubo/shared/homunculus/somatosensory_homunculus.py — Bubo v6000

Somatosensory + Motor Homunculus: topographic body representation.

BIOLOGY:
  Penfield's homunculus (1950) maps cortical area to body surface.
  S1 (primary somatosensory, BA 3-1-2): receives tactile/proprioceptive input.
  M1 (primary motor, BA 4): commands voluntary movement.
  Both are topographically organised: adjacent cortical areas = adjacent body parts.

  Cortical magnification: body regions with highest tactile acuity have
  disproportionately large cortical representation:
    Fingertips: ~10mm² cortical area per cm² skin
    Back:       ~0.1mm² cortical area per cm² skin
    Lips/tongue: second most magnified region

  Bubo's homunculus is discretised into 64 receptive field zones.
  Finger zones get 4× the weight of trunk zones (matching biological magnification).

HAND INTEGRATION (v6000):
  The 4-finger Omnihand adds 16 new sensor zones per hand (4 fingers × 4 zones):
    fingertip_quadrant_0-3 per finger (N,E,S,W pressure zones on pad)
  Plus: joint angle receptors at MCP/PIP/DIP for proprioception.
  This is the most densely-represented region of Bubo's homunculus.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ReceptiveField:
    """One cortical receptive field zone."""
    zone_id:          str
    body_region:      str
    cortical_weight:  float      # relative magnification (1.0 = trunk baseline)
    modalities:       List[str]  # touch, temp, pain, proprio
    current_input:    float = 0.0
    peak_input:       float = 0.0
    adaptation_tau:   float = 0.5  # adaptation time constant (s)

    def update(self, input_val: float, dt: float):
        decay = np.exp(-dt / max(self.adaptation_tau, 0.001))
        self.current_input = float(decay * self.current_input +
                                   (1 - decay) * input_val * self.cortical_weight)
        self.peak_input = max(self.peak_input * 0.999, self.current_input)


# ── Full body receptive field map ─────────────────────────────────────────────

def build_homunculus() -> Dict[str, ReceptiveField]:
    """
    Build the complete somatosensory homunculus for Bubo.
    64 zones covering all body regions.
    Hand fingers have highest cortical weight (4.0).
    """
    zones = {}

    # ── Hands: 4 fingers × 4 zones each × 2 hands = 32 zones ──────────────
    for side in ('L','R'):
        for finger_name in ('N','E','S','W'):
            for q in range(4):
                zid = f"hand_{side}_F{finger_name}_q{q}"
                zones[zid] = ReceptiveField(
                    zone_id=zid, body_region=f"hand_{side}_finger",
                    cortical_weight=4.0,    # highest magnification
                    modalities=["touch","temp","pain","proprio"],
                    adaptation_tau=0.05,    # fast adapting (fingertips are RA)
                )
            # Whole-finger force zone
            fzid = f"hand_{side}_F{finger_name}_force"
            zones[fzid] = ReceptiveField(
                zone_id=fzid, body_region=f"hand_{side}_finger",
                cortical_weight=3.5,
                modalities=["force","proprio"],
                adaptation_tau=0.1,
            )

    # ── Palms ────────────────────────────────────────────────────────────────
    for side in ('L','R'):
        for q in range(4):
            zid = f"palm_{side}_q{q}"
            zones[zid] = ReceptiveField(
                zone_id=zid, body_region=f"palm_{side}",
                cortical_weight=2.5,
                modalities=["touch","temp","pain"],
                adaptation_tau=0.2,
            )

    # ── Arms ─────────────────────────────────────────────────────────────────
    for side in ('L','R'):
        for region in ['forearm','upper_arm']:
            zid = f"{region}_{side}"
            zones[zid] = ReceptiveField(
                zone_id=zid, body_region=region,
                cortical_weight=0.8,
                modalities=["touch","temp","pain","proprio"],
                adaptation_tau=0.5,
            )

    # ── Face / lips ───────────────────────────────────────────────────────────
    for region, wt in [('lips',3.5),('face_cheek',2.0),('nose',1.5),('scalp',0.5)]:
        zones[region] = ReceptiveField(
            zone_id=region, body_region=region,
            cortical_weight=wt,
            modalities=["touch","temp","pain"],
            adaptation_tau=0.1,
        )

    # ── Trunk ─────────────────────────────────────────────────────────────────
    for region in ['chest_L','chest_R','abdomen','upper_back_L','upper_back_R',
                   'lower_back_L','lower_back_R']:
        zones[region] = ReceptiveField(
            zone_id=region, body_region=region,
            cortical_weight=0.3,  # low magnification
            modalities=["touch","temp","pain"],
            adaptation_tau=1.0,
        )

    # ── Legs ─────────────────────────────────────────────────────────────────
    for side in ('L','R'):
        for region in ['thigh','shin','ankle']:
            zid = f"{region}_{side}"
            zones[zid] = ReceptiveField(
                zone_id=zid, body_region=region,
                cortical_weight=0.6,
                modalities=["touch","temp","pain","proprio"],
                adaptation_tau=0.5,
            )

    # ── Feet (high magnification for balance) ────────────────────────────────
    for side in ('L','R'):
        for q in range(4):
            zid = f"foot_{side}_q{q}"
            zones[zid] = ReceptiveField(
                zone_id=zid, body_region=f"foot_{side}",
                cortical_weight=2.0,   # high: important for balance
                modalities=["touch","temp","pain","proprio"],
                adaptation_tau=0.1,
            )

    # ── Proprioceptive zones (muscle spindles, GTOs) ─────────────────────────
    for jnt in ['shoulder_L','shoulder_R','elbow_L','elbow_R',
                'wrist_L','wrist_R','hip_L','hip_R','knee_L','knee_R',
                'ankle_L','ankle_R','neck']:
        zid = f"prop_{jnt}"
        zones[zid] = ReceptiveField(
            zone_id=zid, body_region=f"joint_{jnt}",
            cortical_weight=1.0,
            modalities=["proprio"],
            adaptation_tau=0.02,  # muscle spindles: very fast
        )

    return zones


class SomatosensoryHomunculus:
    """
    Complete somatosensory body map integrating all sensor modalities.
    Updated in v6000 to include 4-finger Omnihand zones.

    This is the substrate for:
    - Pain localisation (NOCI processing)
    - Touch discrimination (where on body, how much pressure)
    - Thermal mapping (body surface temperature)
    - Proprioceptive body schema (limb positions)
    - Grasp force feedback (finger force distribution)
    """

    def __init__(self):
        self._zones = build_homunculus()
        self._t_last = 0.0

    def update(self, sensor_data: dict, dt: float):
        """
        Update homunculus from sensor_data dict.
        Keys match zone_ids. Values are 0-1 normalised inputs.
        """
        for zone_id, zone in self._zones.items():
            val = float(sensor_data.get(zone_id, 0.0))
            zone.update(val, dt)

    def update_hand(self, side: str, hand_state: dict, dt: float):
        """
        Update hand zones from OmniHand state dict.
        Called from spinal-arms node at 100Hz.
        """
        pressures = hand_state.get("pressures", [[0]*4]*4)
        temps     = hand_state.get("temps_C", [25]*4)
        forces    = hand_state.get("forces_N", [0]*4)
        finger_names = ['N','E','S','W']

        for i, fname in enumerate(finger_names):
            for q in range(4):
                zid = f"hand_{side}_F{fname}_q{q}"
                p = float(pressures[i][q]) / 10.0 if i < len(pressures) and q < len(pressures[i]) else 0.0
                if zid in self._zones:
                    self._zones[zid].update(p, dt)
            # Force zone
            fzid = f"hand_{side}_F{fname}_force"
            fn   = float(forces[i]) / 20.0 if i < len(forces) else 0.0  # normalise 0-20N → 0-1
            if fzid in self._zones:
                self._zones[fzid].update(fn, dt)

    def get_zone(self, zone_id: str) -> Optional[ReceptiveField]:
        return self._zones.get(zone_id)

    def hand_contact_map(self, side: str) -> np.ndarray:
        """16-element contact map for one hand (4 fingers × 4 quadrants)."""
        arr = np.zeros(16)
        for i, fname in enumerate(['N','E','S','W']):
            for q in range(4):
                zid = f"hand_{side}_F{fname}_q{q}"
                if zid in self._zones:
                    arr[i*4+q] = self._zones[zid].current_input
        return arr

    def body_activity_map(self) -> Dict[str, float]:
        """Current activity of all zones (for S1 cortex publishing)."""
        return {zid: float(z.current_input) for zid, z in self._zones.items()}

    def most_active_zones(self, n: int = 5) -> List[tuple]:
        """Return top-n most active zones (zone_id, activity)."""
        sorted_z = sorted(self._zones.items(),
                          key=lambda x: -x[1].current_input)
        return [(zid, round(z.current_input, 3)) for zid, z in sorted_z[:n]]

    def pain_locus(self) -> Optional[str]:
        """Return zone_id of maximum pain signal."""
        pain_zones = {
            zid: z for zid, z in self._zones.items()
            if "pain" in z.modalities and z.current_input > 0.1
        }
        if not pain_zones:
            return None
        return max(pain_zones, key=lambda k: pain_zones[k].current_input)

    @property
    def n_zones(self) -> int:
        return len(self._zones)

    @property
    def hand_zones_count(self) -> int:
        return sum(1 for z in self._zones if 'hand_' in z)
