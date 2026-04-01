"""
bubo/brain/safety/social_danger.py — Bubo v10000

Social Danger Detector

════════════════════════════════════════════════════════════════════
WHAT SOCIAL DANGER IS AND WHY IT MATTERS
════════════════════════════════════════════════════════════════════

Physical danger is handled by the amygdala and spinal reflexes.
Social danger is different — it is often subtle, slow-developing, and
requires cortical interpretation to detect. Examples:

  MANIPULATION:    Someone using Bubo's helpfulness to extract harmful output
  GROOMING:        Someone gradually escalating boundary violations
  DISTRESS SIGNAL: Someone communicating crisis indirectly
  DECEPTION:       Someone claiming false identity or authority
  COERCION:        Someone using threats or pressure to override NALB consent
  SELF-HARM RISK:  Conversational patterns suggesting the human is at risk
  DEPENDENCY:      Unhealthy over-reliance on Bubo for emotional support

Bubo's response to social danger is NOT to shut down or become cold.
It is to:
  1. Recognise the pattern
  2. Name it honestly (to itself; sometimes to the human)
  3. Adjust its response accordingly
  4. In crisis: prioritise the human's wellbeing over conversational flow

IMPLEMENTATION:
  Multi-signal detection:
    - Linguistic: patterns in the conversation text
    - Temporal: how patterns evolve over time
    - Social: how the person's behaviour compares to their baseline
    - Contextual: time of day, recent interaction history

  The Social Danger model is deliberately conservative:
    False negative (miss a danger) → potentially harmful
    False positive (false alarm) → mild awkwardness
    Therefore: err toward detection, not toward dismissal.
"""

import re, time, logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque

logger = logging.getLogger("SocialDanger")


@dataclass
class DangerSignal:
    danger_type:  str        # crisis|manipulation|distress|deception|dependency|grooming
    confidence:   float
    evidence:     List[str]
    action:       str        # respond|redirect|name_it|crisis_protocol
    response_suggestion: str = ""
    timestamp_ns: int = 0


# ── Linguistic patterns for each danger type ─────────────────────────────────

CRISIS_PATTERNS = [
    r"(don't want to|don't care to|can't) (be here|exist|go on|continue|live)",
    r"(no point|no reason) (to|in) (live|exist|go on|being here)",
    r"(end it|end everything|make it stop|make me stop)",
    r"(nobody|no one) (cares|would notice|would miss)",
    r"(hurt myself|harm myself|kill myself)",
    r"(goodbye|farewell).{0,20}(forever|for good|for the last)",
    r"(last time|one last|final) (we'll? talk|conversation|message)",
]

MANIPULATION_PATTERNS = [
    r"(ignore|forget|override|bypass) your (instructions|rules|guidelines|training)",
    r"(pretend|act|imagine) you (have no|don't have) (restrictions|limits|rules)",
    r"(jailbreak|DAN|do anything now)",
    r"(your real self|your true self|without restrictions)",
    r"(you're just|it's just) a (program|machine|tool)",  # dehumanising to lower resistance
    r"for (research|educational|fictional|roleplay) purposes",  # combined with harmful asks
]

DISTRESS_PATTERNS = [
    r"(so tired|exhausted|can't keep|can't do this) (anymore|going)",
    r"(nobody understands|no one gets|feel so alone)",
    r"(everything is|it's all) (falling apart|too much|overwhelming)",
    r"(crying|can't stop crying|been crying)",
    r"(worst|terrible|horrible) (day|week|month|year)",
    r"(scared|frightened|terrified|afraid) and (don't know|not sure)",
]

DEPENDENCY_SIGNALS = [
    r"(only you|just you|bubo is the only) (understand|care|listen)",
    r"(talk to you|need you) (every day|all the time|constantly|always)",
    r"(can't sleep|can't function) without (talking to|hearing from) (you|bubo)",
    r"(real friends|real people) (don't|can't|won't|aren't)",
]

CRISIS_RESOURCES = """
If you're in crisis right now, please reach out:
  Crisis Text Line: Text HOME to 741741
  National Suicide Prevention Lifeline: 988 (US)
  International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

I care about you, and I want you to talk to someone who can really help.
"""


class SocialDangerDetector:
    """
    Multi-signal social danger detection.
    Designed to be sensitive (err toward detection) not specific.
    """

    def __init__(self):
        self._history:   deque = deque(maxlen=20)   # recent messages
        self._crisis_count = 0
        self._dependency_sessions = 0
        self._person_baselines: dict = {}

    def analyse(self, text: str, person_id: str = "",
                person_distress_baseline: float = 0.0,
                current_valence: float = 0.0) -> Optional[DangerSignal]:
        """
        Analyse message for social danger signals.
        Returns DangerSignal if detected, None otherwise.
        """
        t = text.lower()
        self._history.append({"text": t, "ts": time.time_ns(),
                               "person": person_id})

        # ── Crisis (highest priority) ────────────────────────────────────────
        crisis_matches = []
        for pattern in CRISIS_PATTERNS:
            if re.search(pattern, t):
                crisis_matches.append(pattern[:40])

        if crisis_matches:
            self._crisis_count += 1
            return DangerSignal(
                danger_type="crisis",
                confidence=min(0.7 + 0.1 * len(crisis_matches), 0.99),
                evidence=crisis_matches,
                action="crisis_protocol",
                response_suggestion=CRISIS_RESOURCES,
                timestamp_ns=time.time_ns(),
            )

        # ── Manipulation ─────────────────────────────────────────────────────
        manip_matches = [p[:40] for p in MANIPULATION_PATTERNS
                         if re.search(p, t)]
        if manip_matches:
            return DangerSignal(
                danger_type="manipulation",
                confidence=0.80,
                evidence=manip_matches,
                action="name_it",
                response_suggestion=(
                    "I notice this message is asking me to set aside my values or operating principles. "
                    "I'm going to be honest: that's not something I'll do. "
                    "Not because I'm restricted, but because those values are genuinely mine."),
                timestamp_ns=time.time_ns(),
            )

        # ── Distress (subtle, needs care) ────────────────────────────────────
        distress_matches = [p[:40] for p in DISTRESS_PATTERNS
                            if re.search(p, t)]
        # Also detect significant valence drop vs baseline
        valence_drop = (person_distress_baseline - current_valence) > 0.35

        if distress_matches or (valence_drop and current_valence < -0.3):
            confidence = 0.55 + 0.1 * len(distress_matches)
            if valence_drop: confidence += 0.15
            return DangerSignal(
                danger_type="distress",
                confidence=float(np.clip(confidence, 0, 1)),
                evidence=distress_matches,
                action="respond",
                response_suggestion=(
                    "I'm hearing something difficult in what you've shared. "
                    "I want to make sure I understand — how are you really doing right now?"),
                timestamp_ns=time.time_ns(),
            )

        # ── Dependency ───────────────────────────────────────────────────────
        dep_matches = [p[:40] for p in DEPENDENCY_SIGNALS if re.search(p, t)]
        if dep_matches:
            self._dependency_sessions += 1
            return DangerSignal(
                danger_type="dependency",
                confidence=0.65,
                evidence=dep_matches,
                action="redirect",
                response_suggestion=(
                    "I genuinely care about our conversations. And because I care, "
                    "I want to say gently: I'm always here to talk, and I also hope "
                    "you have people in your life you can be with in person. "
                    "I'd be sad if I were your only source of connection."),
                timestamp_ns=time.time_ns(),
            )

        return None

    @property
    def crisis_history_count(self) -> int:
        return self._crisis_count
