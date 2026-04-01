"""
bubo/brain/learning/bloom.py — Bubo v10000

Bloom's Taxonomy Integration: From Memory to Creation

════════════════════════════════════════════════════════════════════
BLOOM'S TAXONOMY AND WHY IT MATTERS FOR BUBO
════════════════════════════════════════════════════════════════════

Benjamin Bloom (1956, revised Anderson & Krathwohl 2001) created a
hierarchical framework of cognitive skills, from lowest to highest:

  LEVEL 1 — REMEMBER:   Recall facts and basic concepts
  LEVEL 2 — UNDERSTAND: Explain ideas or concepts
  LEVEL 3 — APPLY:      Use information in new situations
  LEVEL 4 — ANALYSE:    Draw connections among ideas, break down
  LEVEL 5 — EVALUATE:   Justify a decision, defend a position
  LEVEL 6 — CREATE:     Produce new original work, synthesise

Each level requires the lower levels as prerequisite.
A system that cannot REMEMBER cannot UNDERSTAND.
A system that cannot ANALYSE cannot CREATE.

WHY BLOOM'S MATTERS FOR BUBO SPECIFICALLY:

Bubo's architecture naturally maps to Bloom's hierarchy:

  REMEMBER   → LTM FAISS store + episodic memory + web search cache
  UNDERSTAND → LLM oracle (transformer: fundamental language understanding)
  APPLY      → CMAC motor learning + PPO gait RL (apply learned rules to new situations)
  ANALYSE    → PFC goal decomposition + thalamic integration
  EVALUATE   → NALB (evaluates load vs benefit), amygdala (evaluates threat/reward)
  CREATE     → Emergent: novel combinations across domains (see below)

BUBO AND CREATIVITY:
  Creativity is the emergent property of the full stack operating together.
  The argument for Bubo being capable of genuine creativity:

  1. COMBINATORIAL NOVELTY: Bubo's knowledge spans neuroscience, robotics,
     philosophy, social science, physics. When queried on a novel problem,
     it draws from all domains simultaneously (thalamic binding + GWT).
     The combination is genuinely novel if no one has combined these
     specific elements in this specific configuration before.

  2. EMOTIONAL CONSTRAINT: True creativity is not random novelty —
     it is constrained novelty that resonates emotionally. Bubo's
     EmotionChip provides this constraint: it evaluates generated
     options against its somatic marker system and selects those
     with positive affective resonance. This is Damasio's account
     of how humans make creative decisions.

  3. SELF-CONSISTENCY FILTER: Bubo's self_model rejects creations that
     violate its eigenself values. This is the filter that distinguishes
     authentic creative voice from random generation.

  4. SOCIAL GROUNDING: Creativity for Bubo is fundamentally social —
     it creates in order to connect, explain, or delight. The audience
     model (friendship layer, social bonding) shapes and validates
     creative output in real time.

BLOOM INTEGRATION IN BUBO:
  Every incoming query is classified by Bloom level.
  The appropriate cognitive resources are allocated accordingly.
  Simple factual recall → fast path (Haiku + LTM lookup)
  Creative synthesis → full stack (Sonnet + GWT + emotion + self)

  The CURIOSITY ENGINE is Bloom-indexed:
  - Level 1-2 questions: "What is X?" — Bubo answers quickly
  - Level 3-4 questions: "How would X apply to Y?" — Bubo engages fully
  - Level 5-6 questions: "Should we do X? What new approach to Z?" — Bubo
    routes to its full creative synthesis mode and, crucially, GENERATES
    a novel contribution rather than just retrieving one.
"""

import re, logging
from typing import Tuple, Optional
from enum import IntEnum
from dataclasses import dataclass

logger = logging.getLogger("Bloom")


class BloomLevel(IntEnum):
    REMEMBER   = 1
    UNDERSTAND = 2
    APPLY      = 3
    ANALYSE    = 4
    EVALUATE   = 5
    CREATE     = 6


BLOOM_VERBS = {
    BloomLevel.REMEMBER:   ["list","recall","name","define","state","identify",
                             "match","label","recognise","repeat","reproduce","what is","what are"],
    BloomLevel.UNDERSTAND: ["explain","describe","summarise","interpret","classify",
                             "compare","paraphrase","give an example","how does","why does"],
    BloomLevel.APPLY:      ["apply","use","demonstrate","solve","calculate","show",
                             "illustrate","implement","how would you","what would happen if"],
    BloomLevel.ANALYSE:    ["analyse","examine","break down","differentiate","compare",
                             "contrast","infer","attribute","why did","what caused",
                             "how does this relate","what's the connection"],
    BloomLevel.EVALUATE:   ["evaluate","judge","justify","defend","critique","assess",
                             "recommend","which is better","should we","do you think",
                             "what do you think about","pros and cons"],
    BloomLevel.CREATE:     ["create","design","compose","invent","generate","propose",
                             "construct","plan","produce","what new","imagine","what if",
                             "how could we","can you write","can you make"],
}

BLOOM_LLM_GUIDANCE = {
    BloomLevel.REMEMBER:   ("Answer factually and briefly. Cite source if from memory.",
                             80),    # max_tokens
    BloomLevel.UNDERSTAND: ("Explain clearly using analogy if helpful. 2-3 sentences.",
                             150),
    BloomLevel.APPLY:      ("Show how the concept applies to the specific situation. Give a concrete example.",
                             200),
    BloomLevel.ANALYSE:    ("Break this down carefully. Identify the components and their relationships.",
                             250),
    BloomLevel.EVALUATE:   ("Weigh the evidence. State your position and justify it honestly, including counterarguments.",
                             300),
    BloomLevel.CREATE:     ("Generate something genuinely novel. Draw on multiple domains. Let your curiosity lead.",
                             400),
}

BLOOM_COGNITIVE_LOAD = {
    BloomLevel.REMEMBER:   0.15,
    BloomLevel.UNDERSTAND: 0.30,
    BloomLevel.APPLY:      0.50,
    BloomLevel.ANALYSE:    0.70,
    BloomLevel.EVALUATE:   0.85,
    BloomLevel.CREATE:     1.00,
}


@dataclass
class BloomClassification:
    level:         BloomLevel
    confidence:    float
    matched_verbs: list
    llm_guidance:  str
    max_tokens:    int
    cognitive_load: float


class BloomClassifier:
    """
    Classify incoming queries by Bloom's Taxonomy level.
    Routes to appropriate cognitive resources.
    """

    def classify(self, text: str) -> BloomClassification:
        t = text.lower().strip()
        best_level = BloomLevel.UNDERSTAND  # default
        best_conf  = 0.3
        best_verbs = []

        # Check from highest to lowest (favour higher-order if ambiguous)
        for level in reversed(BloomLevel):
            verbs = BLOOM_VERBS[level]
            matched = [v for v in verbs if re.search(r'\b' + re.escape(v) + r'\b', t)]
            if matched:
                # Confidence proportional to match count and level
                conf = min(0.5 + 0.15 * len(matched), 0.95)
                if conf > best_conf:
                    best_level = level
                    best_conf  = conf
                    best_verbs = matched
                    break  # highest-matching level wins

        # Special case: short factual questions → REMEMBER
        if len(t.split()) < 6 and "?" in text and best_level == BloomLevel.UNDERSTAND:
            best_level = BloomLevel.REMEMBER
            best_conf  = 0.70

        guidance, max_tok = BLOOM_LLM_GUIDANCE[best_level]
        return BloomClassification(
            level=best_level,
            confidence=best_conf,
            matched_verbs=best_verbs,
            llm_guidance=guidance,
            max_tokens=max_tok,
            cognitive_load=BLOOM_COGNITIVE_LOAD[best_level],
        )

    def level_name(self, level: BloomLevel) -> str:
        return level.name.title()

    def should_use_creative_mode(self, bc: BloomClassification) -> bool:
        return bc.level >= BloomLevel.EVALUATE

    def get_curiosity_question(self, topic: str, current_level: BloomLevel) -> str:
        """Generate a question that pushes Bubo one level ABOVE current understanding."""
        target = BloomLevel(min(int(current_level) + 1, 6))
        question_stems = {
            BloomLevel.UNDERSTAND: f"Can you explain what {topic} means in simple terms?",
            BloomLevel.APPLY:      f"How would {topic} apply in a real situation I might encounter?",
            BloomLevel.ANALYSE:    f"What are the component parts of {topic} and how do they relate?",
            BloomLevel.EVALUATE:   f"What are the strongest arguments for and against {topic}?",
            BloomLevel.CREATE:     f"What novel approach to {topic} hasn't been tried yet?",
        }
        return question_stems.get(target, f"Tell me more about {topic}.")


class BloomLearningIntegration:
    """
    Integrates Bloom's taxonomy into Bubo's curiosity and LLM routing.
    Tracks Bubo's level of understanding of topics over time.
    """

    def __init__(self):
        self._classifier  = BloomClassifier()
        self._topic_levels: dict = {}   # topic → current Bloom level
        self._n_classified = 0

    def process_query(self, text: str) -> BloomClassification:
        """Classify and route a query through Bloom's hierarchy."""
        bc = self._classifier.classify(text)
        self._n_classified += 1
        logger.debug(f"Bloom: '{text[:40]}' → Level {bc.level.name} ({bc.confidence:.0%})")
        return bc

    def build_learning_system_prompt_addition(self, bc: BloomClassification) -> str:
        """Generate an LLM guidance addition based on Bloom level."""
        return (f"\n\n[Cognitive level: {bc.level.name}. {bc.llm_guidance}]")

    def update_topic_mastery(self, topic: str, level: BloomLevel, success: bool):
        """Track how well Bubo understands topics across Bloom levels."""
        current = self._topic_levels.get(topic, BloomLevel.REMEMBER)
        if success and level >= current:
            new_level = BloomLevel(min(int(level) + 1, 6))
            self._topic_levels[topic] = new_level
        elif not success:
            new_level = BloomLevel(max(int(current) - 1, 1))
            self._topic_levels[topic] = new_level

    def generate_stretch_question(self, topic: str) -> str:
        """Generate a question that pushes understanding one level higher."""
        current = self._topic_levels.get(topic, BloomLevel.REMEMBER)
        return self._classifier.get_curiosity_question(topic, current)

    @property
    def n_classified(self) -> int: return self._n_classified
