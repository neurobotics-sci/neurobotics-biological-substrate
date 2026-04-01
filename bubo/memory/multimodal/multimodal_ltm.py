"""
bubo/memory/multimodal/multimodal_ltm.py — Bubo V10

Long-Term Multimodal Limbic-System-Weighted Memory

════════════════════════════════════════════════════════════════════
DESIGN: WHY MEMORY NEEDS TO BE MULTIMODAL AND LIMBIC-WEIGHTED
════════════════════════════════════════════════════════════════════

Human episodic memory is not a filing cabinet. It is a reconstructive,
emotionally-modulated system with systematic biases that serve survival:

FLASHBULB MEMORIES (Brown & Kulik 1977):
  Emotionally intense events (9/11, first kiss, a car accident) are
  encoded with extraordinary vividness and detail. The amygdala's
  noradrenaline release during high-arousal events enhances hippocampal
  long-term potentiation (Cahill & McGaugh 1998). High emotion = better
  encoding = better retention = better retrieval.

MULTIMODAL BINDING:
  A memory of "the kitchen where Alex cooked for me" is:
    Visual:        warm yellow light, steam from pasta
    Auditory:      sizzling, Alex's voice
    Olfactory:     garlic, olive oil (Bubo: temperature sensor analogue)
    Somatosensory: warmth of the kitchen, plate in hand
    Emotional:     joy (0.82), contentment (0.71)
    Linguistic:    what was said
  The hippocampus binds all these streams into one episode.
  Retrieval of any one stream can cue the others (context-dependent memory).

BUBO'S MULTIMODAL LTM (V10):
  Each episode stores:
    - Visual embedding (128-dim mean V1/MT feature vector)
    - Auditory fingerprint (50-dim MFCC mean)
    - Spatial context (3D pose from RTABMap, room_id)
    - Linguistic summary (LLM-generated 2-sentence summary, stored verbatim)
    - Emotional state at encoding (AffectiveState snapshot)
    - Social context (who was present, bond levels)
    - NALB mode at encoding (were we stressed? curious? content?)
    - Saliency score (limbic-weighted: emotion × novelty × social)

LIMBIC WEIGHTING (saliency function):
  S = w_ne × NE                    # noradrenaline: arousal/stress → better encoding
    + w_da × DA × novelty          # dopamine × novelty: reward prediction error
    + w_fear × CEA                 # fear: survival relevance
    + w_social × bond × |valence|  # social importance
    + w_pain × pain                # pain: avoid this in future
    - w_fatigue × fatigue          # fatigue degrades encoding

  This produces a [0,1] saliency score. Episodes above 0.15 are kept
  through glial cleanup. Episodes above 0.7 are replayed during NREM.

RETRIEVAL:
  Query: multimodal embedding (partial) → cosine similarity search
  Returns: top-K episodes, sorted by: similarity × (saliency^0.5)
  The saliency^0.5 weighting gives emotional memories a boost without
  making neutral memories completely inaccessible.

VECTOR STORE:
  For fast similarity search: FAISS (Facebook AI Similarity Search)
  Falls back to numpy cosine similarity if FAISS not installed.
  Combined with SQLite for full episode metadata.
  Index: HNSW (approximate nearest neighbour, sub-linear search time)
"""

import time, logging, sqlite3, json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path

logger = logging.getLogger("MultimodalLTM")

DB_PATH  = Path("/opt/bubo/data/bubo_multimodal_ltm.db")
IDX_PATH = Path("/opt/bubo/data/bubo_faiss.index")

# Saliency weights (limbic weighting)
W_NE     = 0.25   # noradrenaline (stress/arousal → good encoding)
W_DA     = 0.20   # dopamine × novelty
W_FEAR   = 0.20   # amygdala CeA (survival)
W_SOCIAL = 0.20   # social importance
W_PAIN   = 0.10   # pain
W_NEG_FAT= 0.05   # fatigue penalty

# Embedding dimensions
DIM_VISUAL   = 128
DIM_AUDIO    = 50
DIM_SPATIAL  = 6    # [x, y, z, roll, pitch, yaw]
DIM_EMOTION  = 200  # from VAE latent
DIM_COMBINED = DIM_VISUAL + DIM_AUDIO + DIM_SPATIAL + 12  # +metadata
# Note: emotion dim kept separate (too large for fast search)


@dataclass
class EpisodeMemory:
    """One complete episodic memory, multimodal and limbic-weighted."""
    episode_id:      str

    # Timestamps
    encoded_ns:      int
    last_retrieved:  int = 0
    retrieval_count: int = 0

    # Multimodal content
    visual_emb:      np.ndarray = field(default_factory=lambda: np.zeros(DIM_VISUAL))
    audio_emb:       np.ndarray = field(default_factory=lambda: np.zeros(DIM_AUDIO))
    spatial_pose:    np.ndarray = field(default_factory=lambda: np.zeros(DIM_SPATIAL))
    emotion_at_enc:  np.ndarray = field(default_factory=lambda: np.zeros(DIM_EMOTION))

    # Linguistic + social
    summary:         str = ""      # LLM-generated 2-sentence summary
    location_name:   str = ""      # "Alex's kitchen", "living room"
    persons_present: List[str] = field(default_factory=list)
    bond_levels:     Dict[str, float] = field(default_factory=dict)

    # Emotional encoding context
    valence_enc:     float = 0.0
    arousal_enc:     float = 0.0
    dominant_emotion:str   = "neutral"
    nalb_mode:       str   = "nominal"

    # Limbic-weighted saliency
    saliency:        float = 0.0
    ne_level:        float = 0.0   # noradrenaline at encoding
    da_level:        float = 0.0
    fear_level:      float = 0.0
    pain_level:      float = 0.0
    novelty:         float = 0.0   # estimated novelty at encoding
    social_salience: float = 0.0

    def combined_embedding(self) -> np.ndarray:
        """Concatenated search embedding (visual + audio + spatial + metadata)."""
        meta = np.array([
            self.valence_enc, self.arousal_enc, self.saliency,
            self.ne_level, self.da_level, self.fear_level,
            self.pain_level, self.novelty, self.social_salience,
            float(len(self.persons_present) > 0),
            float(max(self.bond_levels.values()) if self.bond_levels else 0.0),
            float(self.retrieval_count) / 100.0,
        ])
        v = self.visual_emb / (np.linalg.norm(self.visual_emb) + 1e-8)
        a = self.audio_emb  / (np.linalg.norm(self.audio_emb)  + 1e-8)
        s = self.spatial_pose / 10.0  # normalise metres
        return np.concatenate([v[:DIM_VISUAL], a[:DIM_AUDIO], s[:DIM_SPATIAL], meta])

    def to_dict(self) -> dict:
        return {
            "episode_id":      self.episode_id,
            "encoded_ns":      self.encoded_ns,
            "last_retrieved":  self.last_retrieved,
            "retrieval_count": self.retrieval_count,
            "summary":         self.summary,
            "location_name":   self.location_name,
            "persons_present": self.persons_present,
            "bond_levels":     self.bond_levels,
            "valence_enc":     self.valence_enc,
            "arousal_enc":     self.arousal_enc,
            "dominant_emotion":self.dominant_emotion,
            "nalb_mode":       self.nalb_mode,
            "saliency":        self.saliency,
            "ne_level":        self.ne_level,
            "da_level":        self.da_level,
            "fear_level":      self.fear_level,
            "pain_level":      self.pain_level,
            "novelty":         self.novelty,
            "social_salience": self.social_salience,
        }


def compute_saliency(ne: float, da: float, novelty: float, cea: float,
                     bond_avg: float, valence_abs: float, pain: float,
                     fatigue: float) -> float:
    """
    Limbic-weighted saliency score for memory encoding.
    Implements Cahill & McGaugh (1998) noradrenaline-memory enhancement.
    """
    s = (W_NE     * ne
       + W_DA     * da * novelty
       + W_FEAR   * cea
       + W_SOCIAL * bond_avg * valence_abs
       + W_PAIN   * pain
       - W_NEG_FAT * fatigue)
    return float(np.clip(s, 0, 1))


class MultimodalLTM:
    """
    Long-term multimodal memory store with limbic-weighted encoding and
    multimodal retrieval. Hippocampus-equivalent for Bubo V10.
    """

    def __init__(self):
        self._episodes: Dict[str, EpisodeMemory] = {}
        self._embeddings: Optional[np.ndarray] = None  # stacked for numpy search
        self._ep_order: List[str] = []
        self._db  = self._init_db()
        self._load_from_db()
        self._faiss = self._init_faiss()

    def _init_db(self) -> sqlite3.Connection:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY,
            encoded_ns INTEGER, last_retrieved INTEGER, retrieval_count INTEGER,
            summary TEXT, location_name TEXT,
            persons_json TEXT, bonds_json TEXT,
            valence_enc REAL, arousal_enc REAL, dominant_emotion TEXT, nalb_mode TEXT,
            saliency REAL, ne_level REAL, da_level REAL, fear_level REAL,
            pain_level REAL, novelty REAL, social_salience REAL,
            visual_emb BLOB, audio_emb BLOB, spatial_pose BLOB, emotion_emb BLOB)""")
        conn.commit()
        return conn

    def _init_faiss(self):
        try:
            import faiss
            dim = DIM_COMBINED
            index = faiss.IndexHNSWFlat(dim, 32)  # HNSW, M=32
            if IDX_PATH.exists():
                index = faiss.read_index(str(IDX_PATH))
                logger.info(f"FAISS index loaded: {index.ntotal} episodes")
            else:
                logger.info("FAISS: new index created")
            return index
        except ImportError:
            logger.warning("FAISS not available — using numpy cosine similarity (slower)")
            return None

    def _load_from_db(self):
        """Load episodes from SQLite on startup."""
        rows = self._db.execute("""
            SELECT episode_id, encoded_ns, last_retrieved, retrieval_count,
                   summary, location_name, persons_json, bonds_json,
                   valence_enc, arousal_enc, dominant_emotion, nalb_mode,
                   saliency, ne_level, da_level, fear_level, pain_level,
                   novelty, social_salience,
                   visual_emb, audio_emb, spatial_pose, emotion_emb
            FROM episodes ORDER BY saliency DESC LIMIT 2000
        """).fetchall()

        for row in rows:
            ep = EpisodeMemory(
                episode_id=row[0], encoded_ns=row[1],
                last_retrieved=row[2], retrieval_count=row[3],
                summary=row[4] or "", location_name=row[5] or "",
                persons_present=json.loads(row[6] or "[]"),
                bond_levels=json.loads(row[7] or "{}"),
                valence_enc=row[8], arousal_enc=row[9],
                dominant_emotion=row[10] or "neutral", nalb_mode=row[11] or "nominal",
                saliency=row[12], ne_level=row[13], da_level=row[14],
                fear_level=row[15], pain_level=row[16],
                novelty=row[17], social_salience=row[18],
                visual_emb=np.frombuffer(row[19] or b"\x00"*DIM_VISUAL*4,
                                         dtype=np.float32)[:DIM_VISUAL],
                audio_emb=np.frombuffer(row[20] or b"\x00"*DIM_AUDIO*4,
                                        dtype=np.float32)[:DIM_AUDIO],
                spatial_pose=np.frombuffer(row[21] or b"\x00"*DIM_SPATIAL*4,
                                           dtype=np.float32)[:DIM_SPATIAL],
                emotion_at_enc=np.frombuffer(row[22] or b"\x00"*DIM_EMOTION*4,
                                             dtype=np.float32)[:DIM_EMOTION],
            )
            self._episodes[ep.episode_id] = ep
            self._ep_order.append(ep.episode_id)

        logger.info(f"MultimodalLTM: loaded {len(self._episodes)} episodes from DB")

    def encode(self, episode: EpisodeMemory):
        """Encode a new episodic memory."""
        import uuid
        if not episode.episode_id:
            episode.episode_id = str(uuid.uuid4())[:12]
        episode.encoded_ns = time.time_ns()

        self._episodes[episode.episode_id] = episode
        self._ep_order.append(episode.episode_id)

        # Add to FAISS if available
        if self._faiss is not None:
            emb = episode.combined_embedding().astype(np.float32)
            self._faiss.add(emb.reshape(1, -1))

        # Persist to SQLite
        self._db.execute("""INSERT OR REPLACE INTO episodes VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            episode.episode_id, episode.encoded_ns, episode.last_retrieved,
            episode.retrieval_count, episode.summary, episode.location_name,
            json.dumps(episode.persons_present), json.dumps(episode.bond_levels),
            episode.valence_enc, episode.arousal_enc,
            episode.dominant_emotion, episode.nalb_mode,
            episode.saliency, episode.ne_level, episode.da_level,
            episode.fear_level, episode.pain_level,
            episode.novelty, episode.social_salience,
            episode.visual_emb.astype(np.float32).tobytes(),
            episode.audio_emb.astype(np.float32).tobytes(),
            episode.spatial_pose.astype(np.float32).tobytes(),
            episode.emotion_at_enc.astype(np.float32).tobytes(),
        ))
        self._db.commit()
        logger.debug(f"Encoded episode {episode.episode_id}: saliency={episode.saliency:.2f} '{episode.summary[:40]}'")

    def retrieve(self, query_visual: np.ndarray = None,
                 query_audio: np.ndarray = None,
                 query_text: str = "",
                 k: int = 5) -> List[EpisodeMemory]:
        """
        Retrieve top-K relevant episodes via multimodal similarity search.
        Results boosted by sqrt(saliency) to favour emotionally significant memories.
        """
        if not self._episodes:
            return []

        # Build query embedding
        qv = (query_visual / (np.linalg.norm(query_visual)+1e-8)
              if query_visual is not None else np.zeros(DIM_VISUAL))
        qa = (query_audio  / (np.linalg.norm(query_audio)+1e-8)
              if query_audio is not None else np.zeros(DIM_AUDIO))
        q_meta = np.zeros(DIM_SPATIAL + 12)
        q_emb  = np.concatenate([qv[:DIM_VISUAL], qa[:DIM_AUDIO], q_meta]).astype(np.float32)

        if self._faiss is not None and self._faiss.ntotal > 0:
            k_search = min(k * 3, self._faiss.ntotal)
            D, I = self._faiss.search(q_emb.reshape(1,-1), k_search)
            candidates = [self._ep_order[i] for i in I[0] if 0 <= i < len(self._ep_order)]
        else:
            # Numpy fallback
            scores = {}
            for eid, ep in self._episodes.items():
                emb = ep.combined_embedding().astype(np.float32)
                sim = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb)*np.linalg.norm(emb)+1e-8))
                scores[eid] = sim
            candidates = sorted(scores, key=scores.get, reverse=True)[:k*3]

        # Boost by saliency and re-rank
        scored = []
        for eid in candidates:
            ep = self._episodes.get(eid)
            if ep is None: continue
            # Text relevance (simple keyword match)
            text_boost = 0.0
            if query_text:
                for word in query_text.lower().split():
                    if word in (ep.summary + ep.location_name + " ".join(ep.persons_present)).lower():
                        text_boost += 0.1
            combined_score = float(np.sqrt(ep.saliency) * 0.5 + text_boost)
            scored.append((combined_score, ep))

        scored.sort(key=lambda x: -x[0])
        results = [ep for _, ep in scored[:k]]

        # Update retrieval counts
        for ep in results:
            ep.retrieval_count += 1
            ep.last_retrieved   = time.time_ns()
            self._db.execute("UPDATE episodes SET retrieval_count=?, last_retrieved=? WHERE episode_id=?",
                             (ep.retrieval_count, ep.last_retrieved, ep.episode_id))
        self._db.commit()
        return results

    def prune(self, min_saliency: float = 0.15):
        """Glial cleanup: remove low-saliency episodes."""
        before = len(self._episodes)
        to_remove = [eid for eid, ep in self._episodes.items() if ep.saliency < min_saliency]
        for eid in to_remove:
            del self._episodes[eid]
            if eid in self._ep_order: self._ep_order.remove(eid)
        self._db.execute("DELETE FROM episodes WHERE saliency < ?", (min_saliency,))
        self._db.execute("VACUUM")
        self._db.commit()
        logger.info(f"Prune: {before} → {len(self._episodes)} episodes ({len(to_remove)} removed)")
        return len(to_remove)

    @property
    def n_episodes(self) -> int: return len(self._episodes)
    @property
    def mean_saliency(self) -> float:
        if not self._episodes: return 0.0
        return float(np.mean([ep.saliency for ep in self._episodes.values()]))
