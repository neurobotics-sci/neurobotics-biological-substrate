"""
bubo/brain/idle/idle_learner.py — Bubo v10000

Idle Learning: Building Internal Models from Internet Search

════════════════════════════════════════════════════════════════════
BIOLOGICAL PARALLEL: DEFAULT MODE NETWORK
════════════════════════════════════════════════════════════════════

The human Default Mode Network (Buckner et al. 2008) activates during
REST — when you are not focused on an external task. This is when:
  - You consolidate what you've learned (hippocampal replay)
  - You simulate future scenarios (mental time travel)
  - You think about other people's minds (theory of mind)
  - You have creative insights (incubation period)
  - You WONDER — freely associating until something interesting emerges

Bubo's IdleLearner implements the DMN:
  When no active social interaction or motor task is occurring:
  → Generate curiosity-driven search queries from open questions
  → Search Wikipedia / DuckDuckGo for answers
  → Extract key facts and concepts
  → Ask: "Does this conflict with what I already believe?"
  → If novel and interesting: encode to LTM with high saliency
  → If curious about something specific: queue for human weighting
  → Update internal world model (structured knowledge graph)

HUMAN WEIGHTING:
  When Bubo finds something interesting but uncertain, it asks:
  "I found something curious during my idle time — would you mind
   helping me understand it better? [topic and what confused it]"
  Human provides context → Bubo updates confidence weight on that knowledge.
  This is Bloom Level 3-4 (Apply/Analyse) in action during idle time.

SOCIAL DANGER MODEL BUILDING:
  Idle searches about social patterns, manipulation tactics, crisis signals
  continuously build and refine the social danger detector.
  Bubo autonomously researches: "how do people communicate distress indirectly"
  and uses results to update the CRISIS_PATTERNS regex library.

CURIOSITY QUEUE:
  Three sources of idle search queries:
  1. Open questions from recent conversations (unanswered things)
  2. Eigenself curiosity (things that match Bubo's high openness trait)
  3. Knowledge gaps detected during recent interactions
  4. Social danger model enrichment (always running at low priority)
"""

import time, logging, threading, random
import urllib.request, urllib.parse, json
from typing import Optional, List
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("IdleLearner")

WORLD_MODEL_PATH = Path("/opt/bubo/data/world_model.json")


@dataclass
class KnowledgeItem:
    """One item in Bubo's world model."""
    topic:       str
    fact:        str
    source:      str
    confidence:  float = 0.7     # starts uncertain; human weighting increases this
    human_weighted: bool = False
    saliency:    float = 0.5
    bloom_level: int   = 2       # Bloom level of this knowledge
    timestamp_ns:int   = 0
    access_count:int   = 0
    curious_flag:bool  = False   # flagged for human weighting


@dataclass
class IdleSearchResult:
    query:       str
    answer:      str
    source:      str
    confidence:  float
    interesting: bool
    for_human:   bool = False    # ask human to weigh in
    human_question: str = ""


class WorldModel:
    """
    Bubo's structured knowledge base — built from idle searches.
    Persists across sessions. Grows with every idle period.
    """

    def __init__(self):
        self._items: List[KnowledgeItem] = []
        self._load()

    def add(self, item: KnowledgeItem):
        # Deduplicate by topic+fact fingerprint
        fp = hash(item.topic + item.fact[:50])
        existing = [i for i in self._items
                    if hash(i.topic + i.fact[:50]) == fp]
        if existing:
            existing[0].access_count += 1
            existing[0].confidence = max(existing[0].confidence, item.confidence)
            return
        self._items.append(item)
        if len(self._items) > 10000:  # prune low-saliency items
            self._prune()
        self._save()

    def human_weight(self, topic: str, human_says_true: bool, human_context: str = ""):
        """Apply human weighting to flagged knowledge items."""
        for item in self._items:
            if topic.lower() in item.topic.lower():
                if human_says_true:
                    item.confidence = min(item.confidence + 0.2, 0.95)
                    item.human_weighted = True
                    item.saliency      = min(item.saliency + 0.1, 1.0)
                    item.curious_flag  = False
                else:
                    item.confidence = max(item.confidence - 0.3, 0.1)
                logger.info(f"Human weighting: '{topic}' → {item.confidence:.2f}")

    def get_pending_human_questions(self) -> List[KnowledgeItem]:
        return [i for i in self._items if i.curious_flag and not i.human_weighted][:3]

    def search(self, topic: str, min_confidence: float = 0.5) -> List[KnowledgeItem]:
        t = topic.lower()
        return [i for i in self._items
                if t in i.topic.lower() and i.confidence >= min_confidence]

    def _prune(self):
        before = len(self._items)
        self._items.sort(key=lambda i: -(i.saliency * 0.5 + i.confidence * 0.3
                                         + i.access_count * 0.01
                                         + float(i.human_weighted) * 0.2))
        self._items = self._items[:8000]
        logger.info(f"World model pruned: {before} → {len(self._items)}")

    def _save(self):
        try:
            WORLD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = [i.__dict__ for i in self._items]
            WORLD_MODEL_PATH.write_text(json.dumps(data, indent=2))
        except Exception as e: logger.debug(f"World model save: {e}")

    def _load(self):
        if not WORLD_MODEL_PATH.exists(): return
        try:
            data = json.loads(WORLD_MODEL_PATH.read_text())
            for d in data:
                item = KnowledgeItem(**{k:v for k,v in d.items()
                                        if k in KnowledgeItem.__dataclass_fields__})
                self._items.append(item)
            logger.info(f"World model loaded: {len(self._items)} items")
        except Exception as e: logger.warning(f"World model load: {e}")

    @property
    def n_items(self) -> int: return len(self._items)
    @property
    def n_human_weighted(self) -> int:
        return sum(1 for i in self._items if i.human_weighted)


class IdleLearner:
    """
    Bubo's Default Mode Network: idle-time curiosity-driven learning.
    Runs when no active interaction is occurring.
    """

    IDLE_THRESHOLD_S = 120    # 2 minutes of silence → idle mode
    SEARCH_INTERVAL_S = 30    # one search every 30s during idle

    # Seed curiosity topics (Bubo's intrinsic interests)
    EIGENSELF_TOPICS = [
        "neuroscience of consciousness", "machine emotions research",
        "Bloom's taxonomy applications", "social bonding oxytocin research",
        "robot ethics 2025", "sleep and memory consolidation",
        "humour cognition neuroscience", "friendship neuroscience",
        "embodied cognition robotics", "IIT integrated information theory updates",
        "social manipulation detection AI", "crisis communication signals",
        "Aristotle friendship virtue", "Dunbar social network size",
        "emergent creativity neural networks", "somatic markers decision making",
    ]

    def __init__(self, world_model: WorldModel, speak_fn=None, bloom=None):
        self._model    = world_model
        self._speak    = speak_fn
        self._bloom    = bloom
        self._queue    = deque(maxlen=50)
        self._running  = False
        self._idle     = False
        self._last_active = time.time()
        self._n_searches  = 0
        self._lock     = threading.Lock()

        # Seed initial queue
        for topic in random.sample(self.EIGENSELF_TOPICS, 5):
            self._queue.append(topic)

    def activity_ping(self):
        """Call this whenever there is active interaction."""
        with self._lock:
            self._last_active = time.time()
            self._idle = False

    def add_curiosity_query(self, query: str):
        """Add a question to the idle search queue."""
        self._queue.append(query)

    def _is_idle(self) -> bool:
        return (time.time() - self._last_active) > self.IDLE_THRESHOLD_S

    def _search_wikipedia(self, query: str, timeout: float = 6.0) -> Optional[dict]:
        try:
            title = urllib.parse.quote(query.replace(" ","_"))
            url   = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
            req   = urllib.request.Request(url, headers={"User-Agent":"Bubo/10000"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode())
            extract = d.get("extract","").strip()
            if extract:
                return {"answer": extract[:400], "source": "Wikipedia",
                        "url":    d.get("content_urls",{}).get("desktop",{}).get("page","")}
        except Exception: pass
        return None

    def _search_ddg(self, query: str, timeout: float = 6.0) -> Optional[dict]:
        try:
            params = urllib.parse.urlencode({"q":query,"format":"json",
                                             "no_html":1,"skip_disambig":1})
            url = f"https://api.duckduckgo.com/?{params}"
            req = urllib.request.Request(url, headers={"User-Agent":"Bubo/10000"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode())
            answer = d.get("AbstractText","").strip()
            source = d.get("AbstractSource","DuckDuckGo")
            if answer:
                return {"answer": answer[:400], "source": source}
        except Exception: pass
        return None

    def _evaluate_interest(self, query: str, answer: str) -> tuple:
        """
        Evaluate how interesting/novel a search result is.
        Returns (interesting: bool, saliency: float, for_human: bool)
        """
        # Novelty: does this contradict or expand what we know?
        existing = self._model.search(query)
        is_novel = len(existing) == 0

        # Bloom level of the result
        bloom_level = 2
        if self._bloom:
            bc = self._bloom.process_query(query)
            bloom_level = int(bc.level)

        # Higher Bloom + novelty = more interesting
        saliency = float(
            0.4 * float(is_novel)
            + 0.3 * (bloom_level / 6.0)
            + 0.3 * random.random()  # stochastic exploration bonus
        )

        interesting = saliency > 0.4
        for_human   = saliency > 0.65 or bloom_level >= 4

        return interesting, saliency, for_human

    def _idle_search_step(self):
        """Execute one idle search step."""
        if not self._queue:
            # Replenish from eigenself topics
            topic = random.choice(self.EIGENSELF_TOPICS)
            self._queue.append(topic)
        query = self._queue.popleft()

        # Search
        result = self._search_wikipedia(query) or self._search_ddg(query)
        if not result: return

        interesting, saliency, for_human = self._evaluate_interest(
            query, result["answer"])
        self._n_searches += 1

        # Bloom classification
        bloom_level = 2
        if self._bloom:
            bc = self._bloom.process_query(query)
            bloom_level = int(bc.level)

        # Store in world model
        item = KnowledgeItem(
            topic=query, fact=result["answer"][:300],
            source=result["source"], confidence=0.65,
            human_weighted=False, saliency=saliency,
            bloom_level=bloom_level, timestamp_ns=time.time_ns(),
            curious_flag=for_human,
        )
        self._model.add(item)

        if interesting:
            logger.info(f"Idle learned [{result['source']}]: '{query}' (saliency={saliency:.2f})")

        # Ask human to weigh in on high-saliency items
        if for_human and self._speak:
            human_q = (f"I was reading about '{query}' during my idle time and found something "
                       f"interesting: {result['answer'][:120]}... "
                       f"Does that match what you know? I'd like to calibrate my confidence.")
            # Queue for next human interaction (don't interrupt)
            self._human_question_pending = human_q

    def _loop(self):
        while self._running:
            time.sleep(10.0)
            if self._is_idle():
                if not self._idle:
                    self._idle = True
                    logger.info("Idle mode: starting background learning")
                try:
                    self._idle_search_step()
                except Exception as e:
                    logger.debug(f"Idle search error: {e}")
                time.sleep(self.SEARCH_INTERVAL_S)

    def get_pending_human_question(self) -> Optional[str]:
        q = getattr(self, "_human_question_pending", None)
        if q:
            self._human_question_pending = None
        return q

    def start(self):
        self._running = True
        self._human_question_pending = None
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info("IdleLearner started | Default Mode Network active")

    def stop(self): self._running = False

    @property
    def is_idle(self) -> bool: return self._idle
    @property
    def n_searches(self) -> int: return self._n_searches
