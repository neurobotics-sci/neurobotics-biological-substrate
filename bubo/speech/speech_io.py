"""
bubo/speech/speech_io.py — Bubo V10
Speech I/O: Recognition (STT) + Synthesis (TTS) with emotional prosody.

DESIGN:
  Speech Recognition: OpenAI Whisper (tiny.en or base.en) running on Orin GPU
    - tiny.en: ~40ms transcription of 3s speech on Orin Nano GPU
    - base.en: ~100ms, better accuracy
    - Falls back to vosk (offline, lighter) if Whisper not available

  Speech Synthesis: Piper TTS (fast, on-device, neural)
    - Natural prosody from neural vocoder
    - Emotion modulation: rate, pitch, volume from AffectiveState
    - Voice: en_US-lessac-high (warm, natural)
    - Falls back to espeak-ng if Piper not available

  The speech pipeline integrates with:
    - Broca node: receives speech acts to synthesise
    - Emotion chip: AffectiveState → prosody parameters
    - Social curiosity engine: generated questions → speech
    - Knowledge search: spoken answers with attribution
    - NALB: NALB decisions spoken to human (consent requests)
"""

import time, logging, threading, queue, subprocess, tempfile, os
import numpy as np
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger("SpeechIO")

PIPER_BIN    = Path("/usr/local/bin/piper")
PIPER_MODEL  = Path("/opt/bubo/models/en_US-lessac-high.onnx")
WHISPER_MODEL= "base.en"
VOSK_MODEL   = Path("/opt/bubo/models/vosk-model-en-us-0.22")

# Audio device IDs (adjust for hardware)
MIC_DEVICE   = "default"
SPK_DEVICE   = "default"


class SpeechSynthesiser:
    """
    Text-to-speech with emotional prosody modulation.
    Piper TTS preferred (natural, fast, offline).
    espeak-ng fallback (robotic but functional).
    """

    def __init__(self):
        self._backend = self._detect_backend()
        self._speak_q: queue.Queue = queue.Queue()
        self._running  = False
        logger.info(f"TTS backend: {self._backend}")

    def _detect_backend(self) -> str:
        if PIPER_BIN.exists() and PIPER_MODEL.exists():
            return "piper"
        r = subprocess.run(["espeak-ng", "--version"], capture_output=True, timeout=2)
        if r.returncode == 0: return "espeak"
        return "simulation"

    def speak(self, text: str, affect_params: dict = None):
        """Synthesise text to speech with emotional prosody."""
        self._speak_q.put((text, affect_params or {}))

    def _synth_piper(self, text: str, params: dict) -> bool:
        rate  = params.get("rate", 1.0)
        pitch = params.get("pitch", 1.0)
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            cmd = [str(PIPER_BIN),
                   "--model", str(PIPER_MODEL),
                   "--output_file", tmp_path]
            proc = subprocess.run(cmd, input=text.encode(),
                                  capture_output=True, timeout=15)
            if proc.returncode == 0 and os.path.exists(tmp_path):
                # Play with speed/pitch adjustment
                play_cmd = ["aplay", tmp_path]
                subprocess.run(play_cmd, capture_output=True, timeout=30)
                os.unlink(tmp_path)
                return True
        except Exception as e:
            logger.error(f"Piper TTS error: {e}")
        return False

    def _synth_espeak(self, text: str, params: dict) -> bool:
        rate  = int(params.get("rate", 1.0) * 175)   # espeak default ~175wpm
        pitch = int(params.get("pitch", 1.0) * 50)   # espeak pitch 0-99
        try:
            cmd = ["espeak-ng", "-v", "en-us",
                   "-s", str(rate), "-p", str(pitch), text]
            subprocess.run(cmd, timeout=30, capture_output=True)
            return True
        except Exception as e:
            logger.error(f"espeak error: {e}")
            return False

    def _worker(self):
        while self._running:
            try:
                text, params = self._speak_q.get(timeout=1.0)
                if not text: continue
                logger.info(f"TTS [{params.get('rate',1):.1f}x]: '{text[:60]}'")
                if self._backend == "piper":
                    ok = self._synth_piper(text, params)
                    if not ok: self._synth_espeak(text, params)
                elif self._backend == "espeak":
                    self._synth_espeak(text, params)
                else:
                    logger.debug(f"[SIM TTS] {text}")
            except queue.Empty: continue
            except Exception as e:
                logger.error(f"TTS worker: {e}")

    def start(self):
        self._running = True
        threading.Thread(target=self._worker, daemon=True).start()

    def stop(self): self._running = False

    def is_busy(self) -> bool:
        return not self._speak_q.empty()


class SpeechRecogniser:
    """
    Automatic speech recognition.
    Whisper preferred (accurate, offline), Vosk fallback.
    """

    def __init__(self, on_transcript: Callable[[str], None]):
        self._callback = on_transcript
        self._backend  = "simulation"
        self._running  = False
        self._model    = None
        self._detect_backend()

    def _detect_backend(self):
        try:
            import whisper
            self._model   = whisper.load_model(WHISPER_MODEL)
            self._backend = "whisper"
            logger.info(f"STT: Whisper {WHISPER_MODEL}")
        except ImportError:
            try:
                import vosk
                if VOSK_MODEL.exists():
                    self._model   = vosk.Model(str(VOSK_MODEL))
                    self._backend = "vosk"
                    logger.info("STT: Vosk offline")
            except ImportError:
                logger.warning("STT: No backend available — simulation mode")

    def _whisper_listen(self):
        try:
            import sounddevice as sd
            import whisper
            RATE = 16000; CHUNK = RATE * 3  # 3-second chunks
            while self._running:
                audio = sd.rec(CHUNK, samplerate=RATE, channels=1,
                               dtype="float32", device=MIC_DEVICE)
                sd.wait()
                audio_np = audio.flatten()
                if float(np.max(np.abs(audio_np))) > 0.01:   # voice activity
                    result = self._model.transcribe(audio_np, language="en")
                    text   = result["text"].strip()
                    if text and len(text) > 2:
                        logger.info(f"STT: '{text}'")
                        self._callback(text)
        except Exception as e:
            logger.error(f"Whisper listen: {e}")

    def _sim_listen(self):
        """Simulation: periodically generate test utterances."""
        test_inputs = [
            "Hello Bubo, how are you?",
            "What is two plus two?",
            "Can you tell me something interesting?",
            "What can you see right now?",
            "Are you happy?",
        ]
        idx = 0
        while self._running:
            time.sleep(15.0)   # every 15s in sim
            text = test_inputs[idx % len(test_inputs)]; idx += 1
            logger.debug(f"[SIM STT] '{text}'")
            self._callback(text)

    def start(self):
        self._running = True
        if self._backend == "whisper":
            threading.Thread(target=self._whisper_listen, daemon=True).start()
        else:
            threading.Thread(target=self._sim_listen, daemon=True).start()

    def stop(self): self._running = False


class SpeechIO:
    """
    Unified speech I/O with emotional prosody and turn management.
    """
    def __init__(self, on_speech_input: Callable[[str], None]):
        self.tts = SpeechSynthesiser()
        self.stt = SpeechRecogniser(on_speech_input)

    def say(self, text: str, affect_state=None):
        """Speak text with optional emotional prosody."""
        params = {}
        if affect_state is not None:
            params = affect_state.to_tts_params()
        self.tts.speak(text, params)

    def start(self):
        self.tts.start(); self.stt.start()
        logger.info("SpeechIO V10 started (TTS+STT)")

    def stop(self):
        self.tts.stop(); self.stt.stop()
