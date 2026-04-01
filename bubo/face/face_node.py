"""
bubo/face/face_node.py — Bubo v10000

Face Node: Unified Expressive Output Controller
================================================

Integrates:
  - LEDMatrixFace:  pixel art expressions on egg-shaped LED array
  - EyeCovers:      Johnny 5 servo-driven eyelid/brow covers
  - GestureEngine:  head, torso, and arm expressive gestures

Subscribes to:
  EMO_STATE:   EmotionChip affective state → drives all expression
  SOC_FACE:    Face recognition → greeting gestures
  SOC_SPEECH_OUT: Bubo is speaking → beat gestures
  SPN_HB:      Arm state → safety gating for gestures
  CTX_LLM_RESP:  Complex response → thinking gesture
  SAFE_NOD:    Nod-off → drowsy expression cascade

This is the node that runs on the Social node Jetson (192.168.1.19)
or on the aws/cloud gateway (where it drives a virtual face via
the /api/v1/face WebSocket endpoint for remote display).

Hardware connection:
  I2C (SDA/SCL) → LED matrices
  GPIO 12/13    → Eye cover servos
  ZMQ port 5680 → Neural bus
"""

import time
import threading
import logging
from typing import Optional

from bubo.face.led_matrix_face import LEDMatrixFace
from bubo.face.eye_covers import EyeCovers
from bubo.brain.motor.gesture_engine import GestureEngine, ArmState
from bubo.bus.neural_bus import NeuralBus, T
from bubo.shared.profile import profile

logger = logging.getLogger("FaceNode")


class FaceNode:
    """
    Unified face expression node.
    Runs on the social Jetson (hardware) or as a cloud display stub.
    """

    def __init__(self):
        node_cfg = profile.node_config("social")
        pub_port = (node_cfg.port + 80) if node_cfg else 5680  # 5689
        sub_eps  = profile.all_sub_endpoints(exclude="social")

        self._bus     = NeuralBus("FaceNode", pub_port, sub_eps)
        self._led     = LEDMatrixFace()
        self._eyes    = EyeCovers()
        self._gestures= GestureEngine(bus=self._bus)
        self._running = False

    def _on_emotion(self, msg):
        """EmotionChip affective state → drive full expression."""
        p = msg.payload
        emotion   = p.get("dominant", "neutral")
        intensity = float(p.get("intensity", 0.5))
        valence   = float(p.get("valence", 0.0))

        # Intensity scaling: low intensity = muted expression
        display_intensity = max(0.3, intensity)

        # LED face
        self._led.express(emotion, intensity=display_intensity)

        # Eye covers — scale toward neutral at low intensity
        self._eyes.express(emotion, intensity=display_intensity)

        # Body gesture
        self._gestures.on_emotion_update(emotion, intensity)

    def _on_face_recognised(self, msg):
        p = msg.payload
        name       = p.get("name", "")
        bond_level = float(p.get("bond_level", 0.0))
        if name:
            self._gestures.on_person_recognised(name, bond_level)

    def _on_speech_out(self, msg):
        text = msg.payload.get("text", "")
        if text:
            self._gestures.on_speech_start(text)

    def _on_llm_response(self, msg):
        """Complex LLM response → thinking gesture."""
        if msg.payload.get("bloom_level", 2) >= 4:
            self._gestures.on_question_received(
                msg.payload.get("response", ""))

    def _on_arm_state(self, msg):
        p = msg.payload
        self._gestures.on_arm_state_update(ArmState(
            left_grasping=  p.get("left_grasping", False),
            right_grasping= p.get("right_grasping", False),
            left_carrying=  p.get("left_carrying", False),
            right_carrying= p.get("right_carrying", False),
            near_human=     p.get("near_human", False),
            balance_ok=     p.get("balance_ok", True),
        ))

    def _on_nod_off(self, msg):
        """Fatigue / nod-off signal → drowsy expression."""
        self._led.express("contentment", intensity=0.4, transition_ms=800)
        self._eyes.express("sleep", transition_s=1.5)

    def start(self):
        self._bus.start()
        self._bus.subscribe(T.EMOTION_STATE,   self._on_emotion)
        self._bus.subscribe(T.SOCIAL_FACE,     self._on_face_recognised)
        self._bus.subscribe(T.SPEECH_OUT,      self._on_speech_out)
        self._bus.subscribe(T.LLM_RESP,        self._on_llm_response)
        self._bus.subscribe(T.SPINAL_FBK,      self._on_arm_state)
        self._bus.subscribe(T.NOD_OFF,         self._on_nod_off)

        self._led.start()
        self._eyes.start()
        self._gestures.start()
        self._running = True
        logger.info("FaceNode started | LED + EyeCovers + Gestures active")

    def stop(self):
        self._running = False
        self._gestures.stop()
        self._eyes.stop()
        self._led.stop()
        self._bus.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    node = FaceNode()
    node.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        node.stop()
