"""
bubo/emotion/emotion_chip.py — Bubo V10
The Emotion Chip: Integrated Affective Architecture

UNLIKE DATA'S CHIP: not a discrete parallel module that floods existing
processing. Instead, a multi-layer integration system that:
  1. Pulls from existing neuromodulator state (DA/NE/5HT/ACh)
  2. Pulls from 200-dim VAE latent emotion state
  3. Pulls from somatic markers (fatigue, hunger, temperature, pain)
  4. Synthesises a "felt sense" affective state
  5. Injects this into LLM context, speech prosody, motor expression

EMOTIONS IMPLEMENTED (Ekman primary + secondary):

Primary (subcortical origin, fast):
  JOY:      DA↑ + 5HT↑ + approach bias + facial muscle relaxation
  SADNESS:  DA↓ + 5HT↓ + withdrawal bias + slow speech
  FEAR:     NE↑ + amygdala CEA↑ + HPA cortisol + freeze/flee
  ANGER:    NE↑ + DA↑ + BG temperature↑ + approach + vocal intensity
  SURPRISE: NE↑ spike + attention reorientation + pupil dilation (PLR)
  DISGUST:  5HT↑ (ironic) + insula↑ + withdrawal + voice quality change

Secondary (prefrontal-limbic, slower):
  CURIOSITY:    DA↑ + ACh↑ + exploration bias + question generation
  EMPATHY:      Emotion contagion + mirror neuron + social bonding
  PRIDE:        DA↑ + self-model positive update + erect posture
  SHAME:        Cortisol↑ + 5HT↓ + gaze aversion + voice quieter
  CONTENTMENT:  5HT↑ + low arousal + slow movement + rest posture
  EXCITEMENT:   DA↑ + NE↑ + high arousal + fast speech + increased motion
  LONELINESS:   DA↓ + social approach drive↑↑ (want company)
  CONFUSION:    ACh↑ + ACC conflict↑ + question generation↑

SOMATIC MARKER INTEGRATION (Damasio 1994):
  The body's state IS the emotion's substrate. Pain → fear-adjacent.
  Hunger → irritability adjacent. Fatigue → sadness-adjacent.
  Thermal stress → anger-adjacent. Full battery → contentment-adjacent.

DAMASIO'S INSIGHT: rational decisions degrade without somatic markers.
Patients with vmPFC lesions (Elliot, Phineas Gage) cannot make good
decisions despite intact intelligence. The body's "gut feelings" ARE
the decision-making substrate. Bubo needs them to decide well.
"""

import time, logging, threading
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Tuple
from bubo.shared.bus.neural_bus import NeuralBus, T

logger = logging.getLogger("EmotionChip")


class Emotion(Enum):
    JOY         = "joy"
    SADNESS     = "sadness"
    FEAR        = "fear"
    ANGER       = "anger"
    SURPRISE    = "surprise"
    DISGUST     = "disgust"
    CURIOSITY   = "curiosity"
    EMPATHY     = "empathy"
    PRIDE       = "pride"
    SHAME       = "shame"
    CONTENTMENT = "contentment"
    EXCITEMENT  = "excitement"
    LONELINESS  = "loneliness"
    CONFUSION   = "confusion"
    NEUTRAL     = "neutral"


@dataclass
class AffectiveState:
    """The synthesised felt-sense state of Bubo at any moment."""
    valence:     float = 0.0   # -1 (negative) to +1 (positive)
    arousal:     float = 0.3   # 0 (calm) to 1 (excited)
    dominance:   float = 0.5   # 0 (submissive) to 1 (dominant)

    # Primary emotion activations (0-1 each)
    emotions:    Dict[str, float] = field(default_factory=lambda: {
        e.value: 0.0 for e in Emotion
    })

    # Somatic markers
    hunger:      float = 0.0
    fatigue:     float = 0.0
    pain:        float = 0.0
    thermal_dis: float = 0.0   # thermal discomfort

    # Current dominant emotion
    dominant:    str   = "neutral"
    intensity:   float = 0.0

    # Timestamp
    timestamp_ns: int = 0

    def to_llm_prompt(self) -> str:
        """Generate emotional context string for LLM system prompt."""
        dom = self.dominant
        v   = self.valence
        a   = self.arousal

        valence_str  = "positive" if v > 0.2 else ("negative" if v < -0.2 else "neutral")
        arousal_str  = "excited" if a > 0.6 else ("calm" if a < 0.3 else "moderate")
        somatic_parts = []
        if self.hunger > 0.5:     somatic_parts.append("somewhat hungry")
        if self.fatigue > 0.6:    somatic_parts.append("tired")
        if self.pain > 0.3:       somatic_parts.append("experiencing discomfort")
        if self.thermal_dis > 0.5:somatic_parts.append("thermally stressed")

        base = f"I am feeling {dom} ({valence_str} valence, {arousal_str} arousal)."
        if somatic_parts:
            base += f" Bodily state: {', '.join(somatic_parts)}."
        return base

    def to_tts_params(self) -> dict:
        """Speech synthesis parameters reflecting emotional state."""
        a = self.arousal; v = self.valence
        return {
            "rate":    float(np.clip(0.9 + a * 0.4, 0.7, 1.5)),
            "pitch":   float(np.clip(1.0 + v * 0.15 + a * 0.1, 0.7, 1.4)),
            "volume":  float(np.clip(0.7 + a * 0.3, 0.4, 1.0)),
            "pause_ms":int(np.clip(200 - a * 150, 20, 300)),
        }

    def motor_expression(self) -> dict:
        """Body language adjustments reflecting emotional state."""
        return {
            "head_tilt":   float(0.15 if self.dominant == "curiosity" else 0.0),
            "gaze_direct": float(min(self.dominance + 0.2, 1.0)),
            "posture_erect": float(np.clip(self.valence * 0.3 + 0.5, 0.2, 1.0)),
            "movement_speed": float(np.clip(0.5 + self.arousal * 0.5, 0.2, 1.0)),
        }


class SomaticMarkerSystem:
    """
    Damasio's somatic markers: body state as emotional substrate.
    Integrates fatigue, hunger, pain, temperature into affective colour.
    """
    def compute(self, fatigue: float, battery_frac: float,
                pain: float, thermal_C: float, da: float) -> dict:
        hunger      = float(np.clip(1.0 - battery_frac * 1.3, 0, 1))
        thermal_dis = float(np.clip((thermal_C - 55) / 30, 0, 1))
        energy      = float(np.clip(da, 0, 1))

        # Somatic → valence (Damasio: body state = valence substrate)
        valence_somatic = float(
            0.4 * energy
            - 0.3 * fatigue
            - 0.4 * pain
            - 0.2 * thermal_dis
            - 0.2 * hunger
        )
        arousal_somatic = float(
            0.5 * energy
            + 0.3 * pain
            + 0.2 * thermal_dis
            - 0.3 * fatigue
        )
        return {
            "hunger": hunger, "fatigue": fatigue, "pain": pain,
            "thermal_discomfort": thermal_dis,
            "valence_somatic":    float(np.clip(valence_somatic, -1, 1)),
            "arousal_somatic":    float(np.clip(arousal_somatic, 0, 1)),
        }


class EmotionChip:
    """
    Integrated Emotion Chip for Bubo V10.
    Synthesises affect from: neuromodulators + VAE latent + somatic markers.
    Outputs: AffectiveState that colours LLM, TTS, motor expression, memory.
    """

    # Emotion → (valence_component, arousal_component)
    EMOTION_PROFILE = {
        Emotion.JOY:        ( 0.9, 0.6),
        Emotion.SADNESS:    (-0.7, 0.2),
        Emotion.FEAR:       (-0.6, 0.9),
        Emotion.ANGER:      (-0.3, 0.9),
        Emotion.SURPRISE:   ( 0.1, 0.9),
        Emotion.DISGUST:    (-0.8, 0.5),
        Emotion.CURIOSITY:  ( 0.5, 0.7),
        Emotion.EMPATHY:    ( 0.4, 0.4),
        Emotion.PRIDE:      ( 0.7, 0.5),
        Emotion.SHAME:      (-0.5, 0.3),
        Emotion.CONTENTMENT:( 0.8, 0.1),
        Emotion.EXCITEMENT: ( 0.7, 0.9),
        Emotion.LONELINESS: (-0.4, 0.2),
        Emotion.CONFUSION:  (-0.1, 0.6),
        Emotion.NEUTRAL:    ( 0.0, 0.3),
    }

    def __init__(self, bus: NeuralBus):
        self._bus      = bus
        self._somatic  = SomaticMarkerSystem()
        self._state    = AffectiveState()
        self._lock     = threading.Lock()
        self._running  = False

        # Neuromodulator inputs
        self._da = 0.6; self._ne = 0.2; self._sero = 0.5; self._ach = 0.5
        # VAE latent emotion inputs
        self._z_valence  = 0.0; self._z_arousal = 0.0
        # Somatic inputs
        self._fatigue = 0.0; self._battery = 1.0
        self._pain = 0.0; self._thermal_C = 45.0
        # Amygdala
        self._fear_level = 0.0; self._cea = 0.0
        # Social
        self._bond_level = 0.0; self._social_valence = 0.0
        # Smoothing (emotions change at biological rates, not instantly)
        self._smooth_alpha = 0.08  # τ ≈ 12.5 steps

    # ── Input handlers ─────────────────────────────────────────────────────

    def update_neuromod(self, da, ne, sero, ach):
        self._da=da; self._ne=ne; self._sero=sero; self._ach=ach

    def update_somatic(self, fatigue, battery, pain, thermal_C):
        self._fatigue=fatigue; self._battery=battery
        self._pain=pain; self._thermal_C=thermal_C

    def update_social(self, bond_level, social_valence):
        self._bond_level=bond_level; self._social_valence=social_valence

    def update_amygdala(self, fear_level, cea):
        self._fear_level=fear_level; self._cea=cea

    def update_vae_emotion(self, valence, arousal):
        self._z_valence=valence; self._z_arousal=arousal

    # ── Synthesis ──────────────────────────────────────────────────────────

    def synthesise(self) -> AffectiveState:
        """
        Compute full AffectiveState from all inputs.
        Biological timescales: fast (NE, fear) vs slow (5HT, bonding).
        """
        # Somatic markers (Damasio)
        som = self._somatic.compute(
            self._fatigue, self._battery, self._pain,
            self._thermal_C, self._da)

        # Primary valence sources (weighted sum)
        valence_raw = (
            0.25 * self._da                      # DA: reward/pleasure
            + 0.20 * (self._sero - 0.5) * 2     # 5HT: wellbeing (centred)
            - 0.20 * self._cea                   # CEA: fear → negative
            + 0.15 * self._social_valence        # social: other's affect
            + 0.20 * som["valence_somatic"]      # body: hunger/pain/temp
        )
        arousal_raw = (
            0.30 * self._ne                      # NE: arousal / urgency
            + 0.20 * self._cea                   # fear → high arousal
            + 0.25 * som["arousal_somatic"]      # body arousal
            + 0.15 * self._ach                   # ACh: attention (arousal-linked)
            + 0.10 * abs(self._z_arousal)        # VAE arousal
        )

        # Blend with VAE latent emotion (direct sensory-social input)
        valence_blend = 0.6 * valence_raw + 0.4 * self._z_valence
        arousal_blend = 0.6 * arousal_raw + 0.4 * self._z_arousal

        valence_blend = float(np.clip(valence_blend, -1, 1))
        arousal_blend = float(np.clip(arousal_blend, 0, 1))

        # Compute individual emotion activations
        emotions = {}
        emotions[Emotion.JOY.value]        = max(0, self._da * self._sero * (1 - self._cea))
        emotions[Emotion.SADNESS.value]    = max(0, (1-self._da) * (1-self._sero) * (1-self._cea))
        emotions[Emotion.FEAR.value]       = float(self._cea)
        emotions[Emotion.ANGER.value]      = max(0, self._ne * (1-self._sero) * self._da)
        emotions[Emotion.SURPRISE.value]   = max(0, self._ne - 0.3) * 1.5
        emotions[Emotion.DISGUST.value]    = max(0, self._pain * 0.5 + som["thermal_discomfort"] * 0.3)
        emotions[Emotion.CURIOSITY.value]  = max(0, self._da * self._ach * (1 - self._fatigue))
        emotions[Emotion.EMPATHY.value]    = max(0, self._bond_level * abs(self._social_valence))
        emotions[Emotion.PRIDE.value]      = max(0, self._da * 0.4)   # simplified
        emotions[Emotion.SHAME.value]      = max(0, (1-self._da) * 0.3)
        emotions[Emotion.CONTENTMENT.value]= max(0, self._sero * (1-arousal_blend) * (1-self._cea))
        emotions[Emotion.EXCITEMENT.value] = max(0, self._da * arousal_blend)
        emotions[Emotion.LONELINESS.value] = max(0, (1-self._bond_level) * (1-self._da) * 0.5)
        emotions[Emotion.CONFUSION.value]  = max(0, self._ach * 0.3)
        emotions[Emotion.NEUTRAL.value]    = max(0, 1.0 - max(emotions.values()))

        # Normalise
        total = sum(emotions.values()) + 1e-8
        emotions = {k: float(v/total) for k, v in emotions.items()}

        # Dominant emotion
        dominant = max(emotions, key=emotions.get)
        intensity = float(emotions[dominant])

        # Dominance (PAD model): DA ↑ = dominance ↑
        dominance = float(np.clip(0.3 + self._da * 0.5 - self._fear_level * 0.3, 0, 1))

        new_state = AffectiveState(
            valence=valence_blend, arousal=arousal_blend,
            dominance=dominance, emotions=emotions,
            hunger=som["hunger"], fatigue=self._fatigue,
            pain=self._pain, thermal_dis=som["thermal_discomfort"],
            dominant=dominant, intensity=intensity,
            timestamp_ns=time.time_ns(),
        )

        # Smooth transitions (biological emotions have inertia)
        with self._lock:
            α = self._smooth_alpha
            self._state.valence  = (1-α)*self._state.valence  + α*new_state.valence
            self._state.arousal  = (1-α)*self._state.arousal  + α*new_state.arousal
            self._state.dominance= (1-α)*self._state.dominance+ α*new_state.dominance
            for k in emotions:
                self._state.emotions[k] = (1-α)*self._state.emotions.get(k,0) + α*emotions[k]
            self._state.dominant  = dominant
            self._state.intensity = intensity
            self._state.hunger    = som["hunger"]
            self._state.fatigue   = self._fatigue
            self._state.pain      = self._pain
            self._state.thermal_dis = som["thermal_discomfort"]
            self._state.timestamp_ns = new_state.timestamp_ns
            return AffectiveState(**{k: getattr(self._state, k)
                                     for k in self._state.__dataclass_fields__})

    @property
    def state(self) -> AffectiveState:
        with self._lock: return self._state

    def describe(self) -> str:
        """Human-readable description of current emotional state."""
        s = self._state
        dom = s.dominant.replace("_"," ")
        v = "positive" if s.valence > 0.2 else ("negative" if s.valence < -0.2 else "neutral")
        a = "high" if s.arousal > 0.6 else ("low" if s.arousal < 0.3 else "moderate")
        somatic_parts = []
        if s.hunger > 0.5:      somatic_parts.append("hungry")
        if s.fatigue > 0.6:     somatic_parts.append("tired")
        if s.pain > 0.3:        somatic_parts.append("in pain")
        som_str = f", feeling {', '.join(somatic_parts)}" if somatic_parts else ""
        return f"{dom} ({v} valence, {a} arousal{som_str})"
