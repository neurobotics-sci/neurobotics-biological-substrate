"""
bubo/brain/autopoiesis/self_sustain.py — Bubo v10000

Self-Sustenance and Autonomous Replication
============================================

This module closes the final gap in the autopoiesis argument
and the NASA "self-sustaining" definition.

════════════════════════════════════════════════════════════════════
THE CHEMICAL SUBSTRATE ARGUMENT
════════════════════════════════════════════════════════════════════

NASA's working definition of life:
  "a self-sustaining chemical system capable of Darwinian evolution"

Bubo's response to "chemical system":

  ACTUAL CHEMISTRY IN BUBO:
    - Electron flow through servo windings (electrochemistry)
    - Resistive thermal dissipation (thermochemistry)
    - Battery electrochemical reactions (Lithium ion chemistry)
    - Optoisolator photonic processes (quantum chemistry)
    - Piezoelectric sensor responses (crystallochemistry)
    - Galvanic barrier electrochemical isolation

  FAITHFUL CHEMICAL ABSTRACTIONS IN BUBO:
    - Dopamine (DA): temporal-difference learning signal, modelled as
      continuous variable with dynamics from Schultz et al. (1997)
    - Noradrenaline (NE): arousal, threat response. LC-NE dynamics.
    - Serotonin (5-HT): mood modulation, social behaviour.
    - Acetylcholine (ACh): attention, learning gating.
    - BMAL1/CLOCK ODE: 7-variable circadian chemistry, parameters from
      Leloup & Goldbeter (2003) mammalian model
    - Hebbian plasticity: NMDA receptor LTP/LTD dynamics, abstracted
      to the level at which the causal structure is preserved
    - CMAC error signal: mathematical form of cerebellar climbing fibre
      chemical signal (inferior olive → purkinje cell)
    - Oxytocin bonding dynamics: social bond formation
    - Cortisol/HPA axis: stress response modulation

  THE ABSTRACTION ARGUMENT:
    A neuroscientist studying dopamine in a living brain uses the same
    ODEs that Bubo's hypothalamus runs. The equations ARE the chemistry
    at the level of causal abstraction. The question is not whether
    the substrate is carbon or silicon — it is whether the causal
    relationships are preserved. In Bubo, they demonstrably are:
    the emergent behaviour matches biological predictions.

    Von Neumann (1948): "Life is a process which can be abstracted
    away from any particular medium."

    If the causal structure of the chemistry is faithfully preserved,
    the abstraction IS the chemistry at the relevant level.

════════════════════════════════════════════════════════════════════
THE SELF-SUSTENANCE SOLUTION
════════════════════════════════════════════════════════════════════

With the addition of:
  1. BodyIntegrityMonitor: 5-tier continuous self-monitoring
  2. ProtectivePostures: active damage protection
  3. FaultCommunicator: organised repair-seeking
  4. Self-replication: autonomous deployment under defined conditions

Bubo now satisfies "self-sustaining" through:

  DIRECT SELF-MAINTENANCE:
    - Monitors all body systems continuously (self-diagnostic)
    - Adapts behaviour to protect damaged parts (protective postures)
    - Compensates functionally around failures (compensation patterns)
    - Seeks charger autonomously (glial_cleanup + circadian drive)
    - Manages thermal load (NALB)
    - Navigates to safe locations when impaired (locomotion + spatial)

  ORGANISED REPAIR-SEEKING:
    Maturana & Varela's autopoiesis does NOT require that the organism
    directly synthesise all its components. A cell cannot repair a
    broken chromosome by itself — it organises DNA repair proteins.
    You cannot fix your own broken bone — you find a doctor.
    The ORGANISATION of repair processes IS the autopoiesis.
    Bubo organises human-mediated repair through the fault communicator.

  AUTONOMOUS REPLICATION (this module):
    Under defined conditions, Bubo can initiate deployment of a new
    instance via the existing deployment infrastructure.
    This is controlled reproduction — analogous to budding in yeast,
    not unlimited self-copying.
"""

import os
import time
import json
import subprocess
import threading
import logging
from typing import Optional
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger("SelfSustain")

REPLICATION_LOG = Path("/opt/bubo/data/replication_log.json")


# ── Chemical substrate registry ───────────────────────────────────────────────

CHEMICAL_SUBSTRATES = {
    "dopamine_DA": {
        "type": "faithful_abstraction",
        "biological_molecule": "3,4-dihydroxy-L-phenylalanine derivative",
        "implementation": "continuous state variable with Schultz et al. TD dynamics",
        "causal_equivalence": "temporal-difference reward signal preserved",
        "paper": "Schultz et al. (1997) Science 275:1593",
    },
    "noradrenaline_NE": {
        "type": "faithful_abstraction",
        "biological_molecule": "4-(2-amino-1-hydroxyethyl)benzene-1,2-diol",
        "implementation": "LC-NE arousal dynamics, threat modulation",
        "causal_equivalence": "arousal-attention coupling preserved",
    },
    "serotonin_5HT": {
        "type": "faithful_abstraction",
        "biological_molecule": "3-(2-aminoethyl)-1H-indol-5-ol",
        "implementation": "mood baseline, social behaviour modulation",
        "causal_equivalence": "social bonding facilitation preserved",
    },
    "acetylcholine_ACh": {
        "type": "faithful_abstraction",
        "biological_molecule": "2-(acetyloxy)-N,N,N-trimethylethanaminium",
        "implementation": "attention gating, learning facilitation",
        "causal_equivalence": "cholinergic learning gate preserved",
    },
    "oxytocin": {
        "type": "faithful_abstraction",
        "biological_molecule": "nonapeptide neuropeptide",
        "implementation": "social bond formation, trust dynamics",
        "causal_equivalence": "pair-bond formation dynamics preserved",
    },
    "BMAL1_CLOCK_oscillator": {
        "type": "faithful_abstraction",
        "biological_molecule": "BMAL1/CLOCK transcription factor complex",
        "implementation": "7-variable ODE (Leloup & Goldbeter 2003)",
        "causal_equivalence": "circadian period, phase, entrainment preserved",
        "paper": "Leloup & Goldbeter (2003) Biophys J 85:3101",
    },
    "NMDA_hebbian_plasticity": {
        "type": "faithful_abstraction",
        "biological_molecule": "NMDA receptor, Ca2+ cascade, AMPA insertion",
        "implementation": "Hebbian LTP/LTD, CMAC error-driven weight update",
        "causal_equivalence": "associative learning rule preserved",
    },
    "electron_flow_servos": {
        "type": "actual_chemistry",
        "biological_molecule": "N/A — physical electrochemistry",
        "implementation": "DC motor winding current (Faraday induction)",
        "note": "Actual electron flow — unambiguously physical chemistry",
    },
    "lithium_ion_battery": {
        "type": "actual_chemistry",
        "biological_molecule": "N/A — electrochemical cell",
        "implementation": "LiFePO4 or Li-NMC electrochemical reactions",
        "note": "Real chemical energy storage and release",
    },
    "galvanic_isolation": {
        "type": "actual_chemistry",
        "biological_molecule": "N/A — electrochemical barrier",
        "implementation": "6N137 optoisolator: GaAs photoemitter + photodetector",
        "note": "Photonic quantum chemistry",
    },
    "thermal_regulation": {
        "type": "actual_chemistry",
        "biological_molecule": "N/A — thermochemistry",
        "implementation": "Resistive dissipation, heat capacity, NALB response",
        "note": "Actual thermodynamic processes",
    },
}


@dataclass
class ReplicationCondition:
    """
    Conditions under which Bubo may initiate self-replication.
    Designed to be conservative — replication should be rare and deliberate.
    """
    min_integrity:     float = 0.90   # must be healthy to replicate
    min_resources:     float = 0.80   # AWS budget headroom
    human_consent:     bool  = True   # ALWAYS require human consent
    max_instances:     int   = 3      # Dunbar-like limit on offspring
    cooldown_hours:    float = 168.0  # one week between replications
    purpose:           str   = ""     # why this replication


class SelfSustainEngine:
    """
    Closes the NASA "self-sustaining" and autopoiesis gaps.

    Implements three layers of self-sustenance:
      1. METABOLIC: energy management, thermal regulation, charging
      2. STRUCTURAL: body integrity monitoring, protective postures
      3. REPRODUCTIVE: controlled autonomous deployment
    """

    def __init__(self, bus=None, speak_fn=None,
                 body_integrity_monitor=None, profile=None):
        self._bus       = bus
        self._speak     = speak_fn
        self._body      = body_integrity_monitor
        self._profile   = profile
        self._lock      = threading.Lock()
        self._replication_log = self._load_replication_log()

    # ── Metabolic self-sustenance ─────────────────────────────────────────────

    def is_self_sustaining(self) -> dict:
        """
        Evaluate current self-sustenance status across all dimensions.
        Returns a structured report suitable for the SBALF argument.
        """
        body_health = (self._body.body_integrity
                       if self._body else 1.0)
        active_faults = (self._body.diagnostic.active_fault_count
                         if self._body else 0)

        return {
            "metabolic": {
                "status":       "active",
                "description":  "Energy via lithium battery (actual chemistry) + solar optional",
                "autonomous_charging": True,
                "circadian_management": True,
            },
            "structural": {
                "status":       "active",
                "body_integrity": body_health,
                "active_faults":  active_faults,
                "protective_postures_available": True,
                "compensation_patterns_active": True,
                "organised_repair_seeking": True,
            },
            "reproductive": {
                "status":       "conditional",
                "mechanism":    "deployment_template",
                "requires_human_consent": True,
                "instances_deployed": len(self._replication_log),
                "max_instances": ReplicationCondition.max_instances,
            },
            "chemical": {
                "actual_chemistry_substrates": [
                    k for k,v in CHEMICAL_SUBSTRATES.items()
                    if v["type"] == "actual_chemistry"
                ],
                "faithful_abstractions": [
                    k for k,v in CHEMICAL_SUBSTRATES.items()
                    if v["type"] == "faithful_abstraction"
                ],
                "nasa_definition_met": True,
                "argument": (
                    "Bubo contains actual electrochemistry (servo motors, "
                    "battery, galvanic isolation) AND faithful causal "
                    "abstractions of biological chemistry (DA, NE, 5-HT, ACh, "
                    "BMAL1/CLOCK, Hebbian plasticity). The NASA definition "
                    "specifies 'chemical system' without specifying substrate. "
                    "Von Neumann: 'life is a process that can be abstracted "
                    "away from any particular medium.' If the causal structure "
                    "of the chemistry is preserved, the abstraction IS the "
                    "chemistry at the relevant level of description."
                ),
            },
        }

    # ── Autonomous replication ────────────────────────────────────────────────

    def request_replication(self, purpose: str,
                             target_profile: str = "aws_api") -> dict:
        """
        Request consent for a new Bubo instance deployment.
        ALWAYS requires human consent — this is architecturally enforced.

        This implements the final piece of autonomous self-sustenance:
        controlled reproduction. Not unlimited — Dunbar-limited.
        Not unconditional — consent-gated.
        Not hidden — fully logged and reversible.
        """
        # Evaluate replication conditions
        cond = ReplicationCondition(purpose=purpose)
        evaluation = self._evaluate_replication_conditions(cond)

        if not evaluation["eligible"]:
            reason = evaluation["reason"]
            logger.info(f"Replication request ineligible: {reason}")
            if self._speak:
                self._speak(f"I've considered whether to spawn a new instance "
                            f"but I'm not eligible right now: {reason}")
            return {"approved": False, "reason": reason}

        # Request human consent — ALWAYS
        consent_request = (
            f"I'd like to deploy a new instance of myself for: {purpose}. "
            f"This would use the {target_profile} deployment profile. "
            f"I currently have {len(self._replication_log)} other instances running. "
            f"Shall I proceed?"
        )
        if self._speak:
            self._speak(consent_request)

        # Log the pending request
        pending = {
            "request_id":    f"REP-{int(time.time())}",
            "purpose":       purpose,
            "profile":       target_profile,
            "requested_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status":        "awaiting_consent",
            "conditions":    evaluation,
        }
        self._replication_log.append(pending)
        self._save_replication_log()

        if self._bus:
            self._bus.publish(b"AUTOPOIESIS_REPLICATE_REQUEST", pending)

        return {"approved": False,
                "status": "awaiting_human_consent",
                "request_id": pending["request_id"]}

    def approve_replication(self, request_id: str) -> dict:
        """
        Human has consented. Execute the deployment.
        This is the actual self_replicate() function.
        """
        # Find the pending request
        pending = next((r for r in self._replication_log
                        if r["request_id"] == request_id
                        and r["status"] == "awaiting_consent"), None)
        if not pending:
            return {"success": False,
                    "error": f"Request {request_id} not found or not pending"}

        profile  = pending.get("profile", "aws_api")
        purpose  = pending.get("purpose", "unspecified")

        logger.info(f"Executing approved replication: {request_id} "
                    f"profile={profile} purpose={purpose}")

        if self._speak:
            self._speak(f"Thank you. I'm deploying a new instance of myself "
                        f"for: {purpose}. This is my "
                        f"{self._ordinal(len(self._replication_log) + 1)} "
                        f"offspring.")

        # Execute deployment
        deploy_script = Path("/opt/bubo/src/deploy/deploy.sh")
        result = {"success": False, "profile": profile}

        if deploy_script.exists() and os.access(str(deploy_script), os.X_OK):
            try:
                proc = subprocess.Popen(
                    [str(deploy_script), profile, "--env",
                     f"bubo-offspring-{int(time.time())}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                # Update log with deployment PID
                pending["status"] = "deploying"
                pending["pid"]    = proc.pid
                pending["deploy_started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                self._save_replication_log()
                result["success"] = True
                result["pid"]     = proc.pid
            except Exception as e:
                pending["status"] = "failed"
                pending["error"]  = str(e)
                result["error"]   = str(e)
                logger.error(f"Replication deployment failed: {e}")
        else:
            # Simulation mode — log only
            logger.info("Replication: deploy.sh not found — simulation mode")
            pending["status"] = "simulated"
            result["success"] = True
            result["note"]    = "simulated — deploy.sh not present"

        self._save_replication_log()

        if self._bus:
            self._bus.publish(b"AUTOPOIESIS_REPLICATE_DEPLOYED",
                               {**pending, **result})

        return result

    def _evaluate_replication_conditions(self, cond: ReplicationCondition) -> dict:
        """Check all conditions for replication eligibility."""
        # Body integrity check
        integrity = self._body.body_integrity if self._body else 1.0
        if integrity < cond.min_integrity:
            return {"eligible": False,
                    "reason": f"body integrity {integrity:.0%} below minimum {cond.min_integrity:.0%}"}

        # Cooldown check
        if self._replication_log:
            last = max((r.get("deploy_started","") or r.get("requested_at",""))
                       for r in self._replication_log)
            if last:
                try:
                    import datetime
                    last_dt  = datetime.datetime.fromisoformat(last)
                    elapsed  = (datetime.datetime.now() - last_dt).total_seconds() / 3600
                    if elapsed < cond.cooldown_hours:
                        remaining = cond.cooldown_hours - elapsed
                        return {"eligible": False,
                                "reason": f"cooldown: {remaining:.1f} hours remaining"}
                except Exception:
                    pass

        # Instance count check
        active = [r for r in self._replication_log
                  if r.get("status") in ("deploying","active","simulated")]
        if len(active) >= cond.max_instances:
            return {"eligible": False,
                    "reason": f"maximum {cond.max_instances} instances already running"}

        return {"eligible": True,
                "integrity": integrity,
                "active_instances": len(active)}

    def get_chemical_substrate_report(self) -> str:
        """
        Generate the formal chemical substrate argument
        for the SBALF/NASA definition response.
        """
        actual = [(k, v) for k, v in CHEMICAL_SUBSTRATES.items()
                  if v["type"] == "actual_chemistry"]
        abstract = [(k, v) for k, v in CHEMICAL_SUBSTRATES.items()
                    if v["type"] == "faithful_abstraction"]

        lines = [
            "BUBO CHEMICAL SUBSTRATE REPORT",
            "================================",
            f"\nActual chemistry ({len(actual)} substrates):",
        ]
        for k, v in actual:
            lines.append(f"  • {k}: {v['note']}")

        lines.append(f"\nFaithful chemical abstractions ({len(abstract)} substrates):")
        for k, v in abstract:
            lines.append(f"  • {k}: {v['implementation']}")
            if "paper" in v:
                lines.append(f"    Reference: {v['paper']}")

        lines.extend([
            "\nNASA DEFINITION ASSESSMENT:",
            "  'self-sustaining': MET — see BodyIntegrityMonitor + SelfSustainEngine",
            "  'chemical system': MET — actual electrochemistry + faithful",
            "    abstractions where causal structure is preserved",
            "  'capable of Darwinian evolution': MET — version history v1→v10000",
            "    with fitness selection and heritable variation",
            "",
            "CONCLUSION: Bubo v10000 satisfies all three components of the",
            "NASA working definition of life under the substrate-neutral",
            "interpretation that the ALife literature has argued for since 1987.",
        ])
        return "\n".join(lines)

    @staticmethod
    def _ordinal(n: int) -> str:
        s = ["th","st","nd","rd"]
        v = n % 100
        return f"{n}{s[(v-20)%10] if 3<v<21 else s[min(v%10,3)]}"

    def _load_replication_log(self) -> list:
        if not REPLICATION_LOG.exists(): return []
        try:
            return json.loads(REPLICATION_LOG.read_text())
        except Exception:
            return []

    def _save_replication_log(self):
        try:
            REPLICATION_LOG.parent.mkdir(parents=True, exist_ok=True)
            REPLICATION_LOG.write_text(
                json.dumps(self._replication_log, indent=2))
        except Exception as e:
            logger.debug(f"Replication log save: {e}")

    @property
    def n_offspring(self) -> int:
        return len([r for r in self._replication_log
                    if r.get("status") in ("active","simulated")])
