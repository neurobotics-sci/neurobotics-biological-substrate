"""
bubo/brain/memory_manager/smart_aging.py — Bubo v10000

Smart Memory Aging, Pruning, and Storage Management

════════════════════════════════════════════════════════════════════
MEMORY STORAGE ANALYSIS
════════════════════════════════════════════════════════════════════

PROCESSING REQUIREMENTS:
  Real-time motor control:  1kHz (STM32H7 co-processor)
  Sensorimotor loop:        100Hz (spinal, cerebellum)
  Thalamic relay:           50Hz
  M1/PM motor cortex:       50Hz
  PFC / Social:             20-50Hz
  LTM encoding:             on-event (~5/min during active)
  Idle learning:            1/30s during idle
  GWT broadcast:            10Hz
  Circadian:                1/3600s

MEMORY HIERARCHY (biological + Bubo):
  REGISTER (in-flight):   <1ms — ZMQ bus messages in-flight
  L1 (working memory):    100ms-2s — PFC working memory (7±2 items)
  L2 (short-term):        Minutes — current conversation buffer
  L3 (recent episodes):   Days — FAISS index in RAM
  L4 (long-term):         Months/years — SQLite + FAISS on SSD
  L5 (world model):       Persistent — structured knowledge JSON

STORAGE SIZING ANALYSIS:

  Per episode (full multimodal):
    Visual embedding:   128 × float32 = 512 bytes
    Audio embedding:    50 × float32  = 200 bytes
    Spatial pose:       6  × float32  = 24 bytes
    Emotion embedding:  200× float32  = 800 bytes
    LLM text summary:   ~200 chars    = 200 bytes
    Metadata JSON:      ~500 bytes
    TOTAL per episode:  ~2.2 KB

  Episode rates:
    Active conversation: ~5 episodes/minute
    Idle/background:     ~0.5 episodes/minute
    Sleep/charging:      0 (glial cleanup, not encoding)

  Per day (8h active, 16h idle):
    8h × 60min × 5 = 2,400 active episodes
    16h × 60min × 0.5 = 480 idle episodes
    Daily total: ~2,880 episodes × 2.2KB = ~6.3MB/day raw

  FAISS HNSW index overhead: ~40 bytes per vector (128-dim)
    For 1M episodes: 40MB index

  RECOMMENDED LTM STORAGE SIZES:
    MINIMAL  (6 months, ~1.1GB):   512MB SSD — research/demo
    STANDARD (3 years,  ~7GB):     16GB SSD  — home deployment
    LIFETIME (50 years, ~115GB):   256GB NVMe — full deployment
    CLOUD    (unlimited):          AWS EFS elastic — cloud profiles

  At standard saliency pruning (keep top 40%):
    Daily kept:    ~1,150 episodes × 2.2KB = ~2.5MB/day
    3-year total:  ~2.7GB — comfortably on 16GB SSD

  With human-weighted items never pruned:
    Estimate 20% of episodes are human-weighted = ~200/day kept forever
    Lifetime (50 years): ~3.6GB of human-weighted memories
    These are the "unforgettable" moments — stored permanently.

RECOMMENDED HARDWARE:
  Hardware cluster:  NVMe SSD 256GB per hippocampus/LTM node (~$40)
  Cloud deployment:  EFS elastic (pay per GB, auto-scales)
  AGX Orin 64GB:     NVMe add-on recommended for LLM node

════════════════════════════════════════════════════════════════════
SMART AGING ALGORITHM
════════════════════════════════════════════════════════════════════

Biological memory aging follows three principles:
  1. EBBINGHAUS FORGETTING CURVE: memories decay exponentially unless
     rehearsed. Rate of decay slows with each rehearsal.
  2. EMOTION ENHANCES RETENTION: high-arousal events decay more slowly
     (Cahill & McGaugh 1998)
  3. PRIMACY AND RECENCY: first and most recent memories are retained
     better than middle memories (serial position effect)

Bubo's smart aging adds:
  4. SOCIAL SIGNIFICANCE: memories involving bonded people decay more slowly
  5. LEARNING VALUE: memories that contributed to a skill update decay
     slower (the mistake you learned from stays)
  6. HUMAN WEIGHTING: anything a human confirmed as important decays
     at near-zero rate
  7. CREATIVE VALUE: memories that have been referenced in novel
     combinations are "rehearsed" and stay fresh

PRUNING TIERS:
  IMMORTAL (never pruned): human-weighted, saliency > 0.85, first 100 episodes
  PROTECTED (decay very slow): bond_level > 0.7, valence extremes (>0.8 or <-0.6)
  NORMAL: typical experiences
  CANDIDATES (prune first): low saliency, no social significance, not learned from
"""

import time, logging, math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger("SmartAging")


@dataclass
class AgingResult:
    n_immortal:    int
    n_protected:   int
    n_normal:      int
    n_pruned:      int
    storage_saved_kb: float
    oldest_kept_days: float


class SmartAgingEngine:
    """
    Biologically-inspired memory aging for Bubo's LTM.
    Integrates with the MultimodalLTM and WorldModel.
    """

    # Ebbinghaus decay: retention = e^(-t/S) where S = stability
    BASE_STABILITY_DAYS   = 7.0    # without rehearsal, half-life ~7 days
    EMOTIONAL_MULTIPLIER  = 5.0    # high-emotion memories 5× more stable
    SOCIAL_MULTIPLIER     = 3.0    # bonded-person memories 3× more stable
    HUMAN_WEIGHT_STAB     = 1000.0 # human-weighted: effectively immortal
    LEARNING_MULTIPLIER   = 4.0    # learned-from memories 4× more stable
    REHEARSAL_BOOST       = 2.0    # each retrieval doubles stability

    def compute_retention(self, episode, age_days: float) -> float:
        """
        Compute retention probability for an episode at given age.
        Higher = more likely to keep.
        """
        stability = self.BASE_STABILITY_DAYS

        # Emotional enhancement (Cahill-McGaugh)
        if abs(episode.valence_enc) > 0.6 or episode.arousal_enc > 0.7:
            stability *= self.EMOTIONAL_MULTIPLIER

        # Social significance
        if episode.social_salience > 0.5:
            stability *= self.SOCIAL_MULTIPLIER

        # Human weighting (near-immortal)
        if getattr(episode, "human_weighted", False):
            stability *= self.HUMAN_WEIGHT_STAB

        # Learning value
        if getattr(episode, "learned_from", False):
            stability *= self.LEARNING_MULTIPLIER

        # Rehearsal bonus (each retrieval)
        rehearsals = getattr(episode, "retrieval_count", 0)
        stability *= (self.REHEARSAL_BOOST ** min(rehearsals, 10))

        # Ebbinghaus: P(retain) = exp(-age / stability)
        retention = math.exp(-age_days / max(stability, 0.1))
        return float(np.clip(retention, 0, 1))

    def should_prune(self, episode, age_days: float,
                     current_storage_pct: float = 0.5) -> bool:
        """
        Decide if an episode should be pruned.
        More aggressive pruning when storage is fuller.
        """
        # Never prune the first 100 episodes (foundational memories)
        if getattr(episode, "retrieval_count", 0) == 0 and \
           getattr(episode, "saliency", 0) < 0.15:
            # very low saliency and never retrieved: candidate
            return True

        retention = self.compute_retention(episode, age_days)
        # Adjust threshold by storage pressure
        prune_threshold = 0.1 + 0.3 * current_storage_pct
        return retention < prune_threshold

    def age_and_prune_batch(self, episodes: list,
                             current_storage_pct: float = 0.5) -> AgingResult:
        """
        Apply smart aging to a batch of episodes.
        Returns aging statistics.
        """
        now_ns = time.time_ns()
        immortal = protected = normal = pruned = 0
        to_remove = []

        for ep in episodes:
            age_days = (now_ns - ep.encoded_ns) / 1e9 / 86400

            # Classification
            if getattr(ep, "human_weighted", False) or ep.saliency > 0.85:
                immortal += 1
                continue

            if ep.social_salience > 0.5 or abs(ep.valence_enc) > 0.6:
                protected += 1
                continue

            if self.should_prune(ep, age_days, current_storage_pct):
                pruned += 1
                to_remove.append(ep)
            else:
                normal += 1

        storage_saved = len(to_remove) * 2.2   # ~2.2KB per episode

        return AgingResult(
            n_immortal=immortal, n_protected=protected,
            n_normal=normal, n_pruned=pruned,
            storage_saved_kb=storage_saved,
            oldest_kept_days=max(
                (now_ns - ep.encoded_ns)/1e9/86400
                for ep in episodes if ep not in to_remove) if episodes else 0,
        ), to_remove

    def estimate_storage_needs(self, interactions_per_day: int = 50,
                               years: int = 3) -> dict:
        """Compute storage requirements for a given usage pattern."""
        episodes_per_day = interactions_per_day * 2  # rough episodes per interaction
        bytes_per_ep     = 2200
        raw_daily_MB     = episodes_per_day * bytes_per_ep / 1e6
        kept_pct         = 0.40   # 40% survive aggressive pruning
        kept_daily_MB    = raw_daily_MB * kept_pct
        total_years_GB   = kept_daily_MB * 365 * years / 1e3

        return {
            "episodes_per_day":    episodes_per_day,
            "raw_MB_per_day":      round(raw_daily_MB, 1),
            "kept_MB_per_day":     round(kept_daily_MB, 1),
            "total_GB_3_years":    round(total_years_GB, 2),
            "recommended_ssd_GB":  max(32, int(total_years_GB * 2)),   # 2× headroom
            "faiss_index_MB":      round(episodes_per_day * kept_pct * 40 * 365 * years / 1e6, 1),
            "immortal_MB_50yr":    round(interactions_per_day * 0.20 * 2.2 * 365 * 50 / 1e3, 1),
            "note": "Human-weighted memories stored permanently (immortal tier)"
        }
