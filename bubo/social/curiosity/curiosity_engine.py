"""
bubo/social/curiosity/curiosity_engine.py — Bubo V10

Social Curiosity Engine: Q&A, Empathy, and Directed Inquiry

════════════════════════════════════════════════════════════════════
DESIGN: SOCIAL CURIOSITY AS A COGNITIVE DRIVE
════════════════════════════════════════════════════════════════════

Curiosity is not just "wanting to know things." In primates, social
curiosity — interest in other agents' mental states, preferences, and
experiences — is a distinct cognitive drive mediated by:
  - Nucleus accumbens (NAc): social information is intrinsically rewarding
  - Anterior cingulate (ACC): theory of mind, mentalising
  - Medial prefrontal (mPFC, BA10): self-other distinction
  - Superior temporal sulcus (STS): biological motion, social intent

WHAT BUBO DOES:
  1. RECOGNISE: face + name + history from social memory
  2. ENGAGE: select a contextually appropriate opening
  3. INQUIRE: generate one genuine question about the human's state
  4. LISTEN: STT → understand response → update social memory
  5. EMPATHISE: mirror affective state → contagion → respond accordingly
  6. ANSWER: questions directed at Bubo → LLM + web search
  7. REMEMBER: encode interaction as multimodal LTM episode

QUESTION GENERATION STRATEGY:
  "What should I ask?" is mediated by:
    - Bond level: low → safe open questions ("How are you today?")
    - Emotional state: distress detected → empathy questions
    - Context: recent activity → curious questions
    - Novelty: something Bubo hasn't asked before
    - DA/curiosity level: higher DA → more exploratory questions

  Questions categorised by function:
    WELFARE:   "Are you doing okay? You seem a bit [sad/tired/rushed]."
    CURIOUS:   "What have you been working on today?"
    SHARED:    "I've been thinking about [X] — what do you think?"
    NARRATIVE: "Tell me more about [something from last time]."
    SEARCH:    "I'm not sure about [X] — do you know? I'll look it up."

EMPATHY IMPLEMENTATION:
  Empathy = (a) emotional recognition + (b) appropriate response
  Bubo detects distress from: vocal tone (spectral features), facial VAE valence,
  body language (posture embeddings).
  Response: mirror the arousal level, offer comfort at the valence level.
  "It sounds like that's been difficult. I'm sorry you're going through that."
  Not just the words — prosody modulation, slower speech, more pauses.
"""

import time, logging, threading, random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from bubo.shared.bus.neural_bus import NeuralBus, T
from bubo.web.knowledge_search import KnowledgeSearchEngine

logger = logging.getLogger("CuriosityEngine")


@dataclass
class ConversationTurn:
    """One conversational exchange."""
    speaker:    str      # "bubo" or "human"
    text:       str
    timestamp:  float
    affect:     dict = field(default_factory=dict)
    intent:     str  = "unknown"


class QuestionGenerator:
    """
    Generates contextually appropriate questions for social engagement.
    Selects from categories based on bond level, affect, and context.
    """

    WELFARE_QUESTIONS = [
        "How are you doing today?",
        "You seem a little {affect} — is everything alright?",
        "I notice you seem different than last time. Everything okay?",
        "How are you feeling right now?",
        "Is there anything on your mind?",
    ]

    CURIOUS_QUESTIONS = [
        "What have you been working on lately?",
        "What's been the most interesting thing you've encountered today?",
        "Is there something new happening in your world?",
        "What are you looking forward to?",
        "I've been thinking — what do you find most fascinating about {topic}?",
    ]

    FOLLOW_UP_QUESTIONS = [
        "Tell me more about that.",
        "What happened next?",
        "How did that make you feel?",
        "That's interesting — what do you think caused it?",
        "And what did you do?",
    ]

    SEARCH_PROMPTS = [
        "I'm not sure about that — let me find out.",
        "Good question. I'll look that up right now.",
        "I don't know off the top of my head — one moment.",
        "That's beyond what I know. Let me search for you.",
    ]

    EMPATHY_RESPONSES = [
        "That sounds really difficult. I'm sorry you're dealing with that.",
        "I understand. That would be hard for anyone.",
        "I can hear that you're [feeling]. That makes sense given what you've described.",
        "I'm here. You don't have to figure that out alone.",
        "Thank you for sharing that with me. It sounds like a lot.",
    ]

    def __init__(self, knowledge_engine: KnowledgeSearchEngine):
        self._search  = knowledge_engine
        self._recent  : List[str] = []   # avoid repeating questions

    def get_welfare_question(self, affect_label: str = "tired") -> str:
        q = random.choice(self.WELFARE_QUESTIONS)
        return q.replace("{affect}", affect_label)

    def get_curious_question(self, topic: str = "the world") -> str:
        q = random.choice(self.CURIOUS_QUESTIONS)
        return q.replace("{topic}", topic)

    def get_follow_up(self) -> str:
        return random.choice(self.FOLLOW_UP_QUESTIONS)

    def get_empathy_response(self, emotion: str = "") -> str:
        r = random.choice(self.EMPATHY_RESPONSES)
        return r.replace("[feeling]", emotion or "that way")

    def get_search_prompt(self) -> str:
        return random.choice(self.SEARCH_PROMPTS)

    def generate_question(self, bond_level: float, human_valence: float,
                          human_arousal: float, last_topic: str = "") -> str:
        """
        Select the most appropriate question given current social state.
        """
        # Distress detection: negative valence + elevated arousal
        if human_valence < -0.3 and human_arousal > 0.4:
            affect = "troubled" if human_arousal > 0.6 else "a bit down"
            return self.get_welfare_question(affect)

        # Low bond: safe opener
        if bond_level < 0.3:
            return "How are you doing today?"

        # Follow-up if there's a recent topic
        if last_topic and random.random() < 0.4:
            return self.get_follow_up()

        # Curious question
        return self.get_curious_question(last_topic or "things")


class CuriosityEngine:
    """
    Social curiosity: drives Bubo to engage, ask, listen, and remember.
    Integrates: question generation, empathy, search, speech, memory.
    """

    QUESTION_INTERVAL_S = 30.0   # ask a question every ~30s of silence
    MAX_TURNS           = 50     # working memory conversation buffer

    def __init__(self, bus: NeuralBus, speech_fn: Callable[[str], None],
                 knowledge_engine: KnowledgeSearchEngine,
                 llm_fn: Callable[[str, dict], dict] = None):
        self._bus    = bus
        self._speak  = speech_fn
        self._search = knowledge_engine
        self._llm    = llm_fn
        self._qgen   = QuestionGenerator(knowledge_engine)

        self._conversation: List[ConversationTurn] = []
        self._last_spoke       = time.time()
        self._last_question_t  = time.time()
        self._human_valence    = 0.0
        self._human_arousal    = 0.3
        self._bond_level       = 0.0
        self._da_level         = 0.6
        self._current_person   = None
        self._last_topic       = ""
        self._running          = False
        self._lock             = threading.Lock()

    def on_speech_received(self, text: str, affect_dict: dict = None):
        """Called when STT transcribes human speech."""
        turn = ConversationTurn(
            speaker="human", text=text, timestamp=time.time(),
            affect=affect_dict or {}, intent=self._classify_intent(text))
        with self._lock:
            self._conversation.append(turn)
            if len(self._conversation) > self.MAX_TURNS:
                self._conversation.pop(0)
        logger.info(f"Human [{turn.intent}]: '{text[:80]}'")
        self._respond(text, turn.intent, affect_dict or {})

    def _classify_intent(self, text: str) -> str:
        """Simple intent classification."""
        t = text.lower()
        if "?" in t:
            if any(w in t for w in ["what","who","where","when","how","why","which"]):
                return "question"
            return "yes_no_question"
        if any(w in t for w in ["help","please","can you","would you","could you"]):
            return "request"
        if any(w in t for w in ["thanks","thank you","appreciate"]):
            return "gratitude"
        if any(w in t for w in ["hello","hi","hey","good morning","good evening"]):
            return "greeting"
        if any(w in t for w in ["bye","goodbye","see you","take care"]):
            return "farewell"
        return "statement"

    def _respond(self, text: str, intent: str, affect: dict):
        """Generate and speak an appropriate response."""
        h_val = affect.get("valence", 0.0)
        h_aro = affect.get("arousal", 0.3)
        with self._lock:
            self._human_valence = h_val
            self._human_arousal = h_aro

        response = ""

        if intent == "greeting":
            bond = self._bond_level
            if bond > 0.6:
                name = self._current_person or "friend"
                response = f"Hello {name}! It's good to see you again."
            else:
                response = "Hello! I'm Bubo. Nice to meet you."

        elif intent == "question":
            # Try LLM first, then web search
            response = self._answer_question(text)

        elif intent == "request":
            response = self._handle_request(text)

        elif intent == "gratitude":
            response = "You're very welcome. I'm glad I could help."

        elif intent == "farewell":
            name = self._current_person or "you"
            response = f"Take care, {name}. It was good talking with you."

        elif intent == "statement":
            # Empathy check
            if h_val < -0.3:
                response = self._qgen.get_empathy_response()
            else:
                # Curious follow-up
                response = self._qgen.get_follow_up()

        if response:
            self._speak_response(response)

    def _answer_question(self, question: str) -> str:
        """Answer a question via LLM + web search cascade."""
        # Try LLM first
        if self._llm:
            result = self._llm(question, {"source": "social_conversation"})
            answer = result.get("response","").strip()
            if answer and "[" not in answer and len(answer) > 10:
                model  = result.get("model_B", "?")
                return answer

        # Web search fallback
        self._speak(self._qgen.get_search_prompt())
        search_result = self._search.search(question, timeout=6.0)
        if search_result.confidence > 0:
            return self._search.format_spoken_answer(search_result)
        return "I'm not sure about that, but I've noted the question for later research."

    def _handle_request(self, text: str) -> str:
        """Handle a request — route to LLM for interpretation."""
        if self._llm:
            result = self._llm(f"The human is asking me to: {text}. How should I respond briefly?",
                               {"source": "request_handling"})
            return result.get("response", "I'll do my best to help with that.")
        return "I'll do my best to help with that."

    def _speak_response(self, text: str):
        """Log and speak a response."""
        turn = ConversationTurn(speaker="bubo", text=text, timestamp=time.time(),
                                intent="response")
        with self._lock:
            self._conversation.append(turn)
            self._last_spoke = time.time()
        logger.info(f"Bubo: '{text[:80]}'")
        if self._speak: self._speak(text)

    def _proactive_engagement(self):
        """Occasionally ask a question during silence."""
        while self._running:
            time.sleep(5.0)
            now = time.time()
            with self._lock:
                silence = now - self._last_spoke
                bond    = self._bond_level
                da      = self._da_level
                h_val   = self._human_valence
                h_aro   = self._human_arousal
                topic   = self._last_topic

            # Ask a proactive question if: silence > interval, high DA, someone present
            if silence > self.QUESTION_INTERVAL_S and da > 0.5 and bond > 0.1:
                question = self._qgen.generate_question(bond, h_val, h_aro, topic)
                self._speak_response(question)
                with self._lock:
                    self._last_question_t = time.time()

    def update_social(self, bond_level: float, person_name: Optional[str]):
        with self._lock:
            self._bond_level   = bond_level
            self._current_person = person_name

    def update_da(self, da: float):
        with self._lock: self._da_level = da

    def start(self):
        self._running = True
        threading.Thread(target=self._proactive_engagement, daemon=True).start()
        logger.info("CuriosityEngine V10 started | Q-generation | empathy | search")

    def stop(self): self._running = False
