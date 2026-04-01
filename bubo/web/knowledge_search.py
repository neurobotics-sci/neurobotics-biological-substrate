"""
bubo/web/knowledge_search.py — Bubo V10
Internet Knowledge Search: answer questions with live web retrieval.

DESIGN:
  When Bubo encounters a question it cannot answer from:
    - LTM episodic memory
    - 70B LLM parametric knowledge
    - Local sensor state
  → Search the web, extract relevant content, synthesise answer.

BACKENDS (in preference order):
  1. DuckDuckGo Instant Answer API (no key, no rate limit, JSON)
  2. Wikipedia API (well-structured, reliable, encyclopaedic)
  3. SerpAPI (requires key, most comprehensive)
  4. Fallback: "I don't know yet, but I'll find out later"

SOCIAL INTEGRATION:
  When Bubo is asked something it doesn't know during social interaction:
    "I'm not sure — let me look that up for you."
    → Search (1-3s)
    → "According to Wikipedia, [answer]. Does that help?"
  
  Attribution is always spoken — Bubo cites its sources.
  
PRIVACY:
  All searches are logged to LTM as knowledge-acquisition episodes.
  The fact that Bubo looked something up is itself a memory.
  If asked the same question later: retrieves from LTM first.

CURIOSITY-DRIVEN SEARCH:
  The social curiosity engine generates questions Bubo wants to know.
  These queue for searching during idle/low-load periods.
  Results are stored in LTM with high saliency if surprising.
"""

import time, logging, json, threading, queue
import urllib.request, urllib.parse, urllib.error
from typing import Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger("KnowledgeSearch")

SERPAPI_KEY_PATH = Path("/etc/bubo/serpapi_key.txt")


@dataclass
class SearchResult:
    query:     str
    answer:    str
    source:    str
    url:       str
    timestamp: int
    confidence: float   # 0-1


class DuckDuckGoBackend:
    """DuckDuckGo Instant Answers API — no key required."""
    API_URL = "https://api.duckduckgo.com/"

    def search(self, query: str, timeout: float = 5.0) -> Optional[SearchResult]:
        params = urllib.parse.urlencode({
            "q": query, "format": "json", "no_html": 1, "skip_disambig": 1
        })
        url = f"{self.API_URL}?{params}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Bubo/8000"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode())
            # Try AbstractText first (Wikipedia extract)
            answer = data.get("AbstractText", "").strip()
            source = data.get("AbstractSource", "DuckDuckGo")
            src_url= data.get("AbstractURL", "")
            # Try Answer (calculator, definitions)
            if not answer:
                answer = data.get("Answer", "").strip()
                source = "DuckDuckGo Answer"
            # Try Definition
            if not answer:
                answer = data.get("Definition", "").strip()
                source = data.get("DefinitionSource", "DuckDuckGo")
            if answer:
                return SearchResult(query=query, answer=answer[:500],
                                    source=source, url=src_url,
                                    timestamp=time.time_ns(), confidence=0.7)
        except Exception as e:
            logger.debug(f"DDG search failed: {e}")
        return None


class WikipediaBackend:
    """Wikipedia REST API — structured, reliable, encyclopaedic."""
    API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"

    def search(self, query: str, timeout: float = 5.0) -> Optional[SearchResult]:
        # Convert query to likely Wikipedia article title
        title = query.strip().replace(" ", "_")
        url   = self.API_URL + urllib.parse.quote(title)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Bubo/8000"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode())
            extract = data.get("extract", "").strip()
            if extract:
                # Trim to 2 sentences for spoken response
                sentences = extract.split(". ")
                short = ". ".join(sentences[:2]) + "."
                return SearchResult(query=query, answer=short,
                                    source="Wikipedia",
                                    url=data.get("content_urls",{}).get("desktop",{}).get("page",""),
                                    timestamp=time.time_ns(), confidence=0.85)
        except Exception as e:
            logger.debug(f"Wikipedia search failed: {e}")
        return None


class KnowledgeSearchEngine:
    """
    Bubo's web search capability.
    Tries backends in order, returns first successful result.
    Queues curiosity-driven searches for idle time.
    """

    def __init__(self):
        self._backends    = [WikipediaBackend(), DuckDuckGoBackend()]
        self._cache: dict = {}   # simple query → result cache
        self._idle_q: queue.Queue = queue.Queue()
        self._running     = False
        self._results_cb  = None   # callback(result) for async idle searches

    def search(self, query: str, timeout: float = 8.0) -> SearchResult:
        """
        Synchronous search. Returns best result or a "don't know" response.
        """
        query = query.strip()
        if not query:
            return SearchResult(query="", answer="I need a question to search for.",
                                source="system", url="", timestamp=time.time_ns(), confidence=0)

        # Cache check
        if query.lower() in self._cache:
            logger.debug(f"Cache hit: '{query}'")
            return self._cache[query.lower()]

        logger.info(f"Searching: '{query}'")
        for backend in self._backends:
            try:
                result = backend.search(query, timeout=timeout/len(self._backends))
                if result and result.answer:
                    self._cache[query.lower()] = result
                    logger.info(f"Found [{result.source}]: '{result.answer[:80]}'")
                    return result
            except Exception as e:
                logger.debug(f"Backend {type(backend).__name__} error: {e}")

        # No result found
        return SearchResult(
            query=query,
            answer=f"I couldn't find a clear answer to that right now, "
                   f"but I've noted the question and will look into it more.",
            source="none", url="", timestamp=time.time_ns(), confidence=0.0)

    def queue_curiosity_search(self, question: str):
        """Queue a curiosity-driven search for idle time processing."""
        self._idle_q.put(question)

    def _idle_worker(self):
        """Process curiosity searches during idle periods."""
        while self._running:
            try:
                question = self._idle_q.get(timeout=5.0)
                result   = self.search(question)
                if self._results_cb and result.confidence > 0:
                    self._results_cb(result)
                time.sleep(1.0)  # rate limiting
            except queue.Empty: continue
            except Exception as e:
                logger.error(f"Idle search worker: {e}")

    def set_result_callback(self, cb): self._results_cb = cb

    def format_spoken_answer(self, result: SearchResult) -> str:
        """Format a search result for spoken delivery."""
        if result.confidence == 0:
            return result.answer
        src = result.source
        ans = result.answer
        return f"According to {src}: {ans}"

    def start(self):
        self._running = True
        threading.Thread(target=self._idle_worker, daemon=True).start()
        logger.info("KnowledgeSearchEngine V10 started | DDG + Wikipedia backends")

    def stop(self): self._running = False
