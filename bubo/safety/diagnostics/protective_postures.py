"""
bubo/safety/diagnostics/protective_postures.py — Bubo v10000

Protective Postures: The Injured Arm Held Close
================================================

When a human breaks an arm, three things happen to the body:
  1. Pain-driven withdrawal (immediate, spinal reflex)
  2. Voluntary protection — holding the arm against the torso,
     the classic "sling" position — to prevent further damage
  3. Functional compensation — doing everything possible
     with the remaining healthy limbs

Bubo does the same.

DESIGN PRINCIPLE — "CLUTCHED UP":
  Kenneth's exact phrase captures the biological truth.
  When a body part is damaged, you hold it close.
  You don't let it dangle. You don't ignore it.
  You protect it, visibly, which also communicates the injury

  The protective posture serves TWO purposes:
  1. Physical: prevents further mechanical damage to the joint
     "Why is Bubo holding its arm like that?" → because it's hurt.

POSTURE LIBRARY:
  Each posture defines joint angle targets that:
  - Place the affected limb in a mechanically safe position
  - Hold it there with reduced torque (splinting rather than fighting)
  - Maintain balance and upright posture with remaining healthy parts
  - Can be held indefinitely without further damage

COMPENSATION PATTERNS:
  If the left arm is damaged:
    → All grasping tasks shift to right arm
    → Two-arm tasks are deferred or simplified
    → Gestures use right arm only (gesture_engine aware)

  If a leg is damaged:
    → Weight shifts to healthy leg
    → Step length reduces
    → Walking speed reduces
    → Human is asked to help Bubo sit down
"""

import time
import threading
import logging
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass

from bubo.safety.diagnostics.fault_model import FaultSeverity

logger = logging.getLogger("ProtectivePostures")


# ── Joint angle targets for protective postures ───────────────────────────────
# All angles in radians. 0 = anatomical neutral.
# Targets designed for SG90 / Dynamixel XL430 servo range.

@dataclass
class JointTarget:
    """A single joint's protective target."""
    joint_id:      int      # Dynamixel servo ID
    target_rad:    float    # target angle in radians
    torque_limit:  float    # fraction of max torque (0-1). Low = soft splint.
    move_speed:    float    # fraction of max speed (0-1). Slow = gentle.


@dataclass
class ProtectivePosture:
    """
    A full-body protective configuration for a specific fault.
    """
    name:           str
    description:    str
    affected_region:str
    joint_targets:  List[JointTarget]
    compensation:   dict     # what other limbs should do
    hold_torque:    float    # torque to hold posture (low = gentle splint)
    verbal_cue:     str      # what Bubo says when it adopts this posture
    recovery_cue:   str      # what Bubo says when fault resolves


# ── Protective posture definitions ────────────────────────────────────────────

POSTURES: Dict[str, ProtectivePosture] = {

    "left_arm_sling": ProtectivePosture(
        name="left_arm_sling",
        description="Left arm held across torso in sling position",
        affected_region="left_arm",
        joint_targets=[
            # Left shoulder: slight forward, adducted to body
            JointTarget(joint_id=10, target_rad=0.3,  torque_limit=0.25, move_speed=0.15),
            # Left elbow: bent to ~90° (forearm across body)
            JointTarget(joint_id=11, target_rad=1.57, torque_limit=0.25, move_speed=0.15),
            # Left wrist: neutral, slightly flexed
            JointTarget(joint_id=12, target_rad=0.1,  torque_limit=0.20, move_speed=0.10),
            # Left hand: lightly closed (protective grip)
            JointTarget(joint_id=13, target_rad=0.3,  torque_limit=0.15, move_speed=0.10),
        ],
        compensation={
            "right_arm": "primary",
            "grasping":  "right_only",
            "gestures":  "right_and_head_only",
        },
        hold_torque=0.25,
        verbal_cue=(
            "My left arm has a problem — I'm holding it close to protect it. "
            "I can still use my right arm."
        ),
        recovery_cue="My left arm is feeling better. I'm going to move it gently."
    ),

    "right_arm_sling": ProtectivePosture(
        name="right_arm_sling",
        description="Right arm held across torso in sling position",
        affected_region="right_arm",
        joint_targets=[
            JointTarget(joint_id=20, target_rad=0.3,  torque_limit=0.25, move_speed=0.15),
            JointTarget(joint_id=21, target_rad=1.57, torque_limit=0.25, move_speed=0.15),
            JointTarget(joint_id=22, target_rad=0.1,  torque_limit=0.20, move_speed=0.10),
            JointTarget(joint_id=23, target_rad=0.3,  torque_limit=0.15, move_speed=0.10),
        ],
        compensation={
            "left_arm":  "primary",
            "grasping":  "left_only",
            "gestures":  "left_and_head_only",
        },
        hold_torque=0.25,
        verbal_cue=(
            "My right arm has a problem — I'm holding it close to protect it. "
            "I can still use my left arm."
        ),
        recovery_cue="My right arm is feeling better. I'm going to move it gently."
    ),

    "left_hand_guard": ProtectivePosture(
        name="left_hand_guard",
        description="Left hand closed and held against body",
        affected_region="left_hand",
        joint_targets=[
            # Keep arm position normal, just close the hand gently
            JointTarget(joint_id=12, target_rad=0.2,  torque_limit=0.20, move_speed=0.10),
            JointTarget(joint_id=13, target_rad=0.5,  torque_limit=0.15, move_speed=0.08),
            JointTarget(joint_id=14, target_rad=0.5,  torque_limit=0.15, move_speed=0.08),
            JointTarget(joint_id=15, target_rad=0.5,  torque_limit=0.15, move_speed=0.08),
        ],
        compensation={"grasping": "right_preferred"},
        hold_torque=0.15,
        verbal_cue=(
            "There's something wrong with my left hand. "
            "I've closed it gently to protect the fingers."
        ),
        recovery_cue="My left hand is feeling better."
    ),

    "right_hand_guard": ProtectivePosture(
        name="right_hand_guard",
        description="Right hand closed and held against body",
        affected_region="right_hand",
        joint_targets=[
            JointTarget(joint_id=22, target_rad=0.2,  torque_limit=0.20, move_speed=0.10),
            JointTarget(joint_id=23, target_rad=0.5,  torque_limit=0.15, move_speed=0.08),
            JointTarget(joint_id=24, target_rad=0.5,  torque_limit=0.15, move_speed=0.08),
            JointTarget(joint_id=25, target_rad=0.5,  torque_limit=0.15, move_speed=0.08),
        ],
        compensation={"grasping": "left_preferred"},
        hold_torque=0.15,
        verbal_cue=(
            "There's something wrong with my right hand. "
            "I've closed it gently to protect the fingers."
        ),
        recovery_cue="My right hand is feeling better."
    ),

    "left_leg_favour": ProtectivePosture(
        name="left_leg_favour",
        description="Weight shifted to right leg, left leg minimally loaded",
        affected_region="left_leg",
        joint_targets=[
            # Left leg: slight bend, minimal load
            JointTarget(joint_id=30, target_rad=0.15, torque_limit=0.20, move_speed=0.10),
            JointTarget(joint_id=31, target_rad=0.20, torque_limit=0.20, move_speed=0.10),
            JointTarget(joint_id=32, target_rad=0.05, torque_limit=0.15, move_speed=0.10),
        ],
        compensation={
            "weight":     "right_dominant",
            "walking":    "reduced_left_stride",
            "standing":   "prefer_seated_if_available",
        },
        hold_torque=0.20,
        verbal_cue=(
            "My left leg isn't working right — I'm being careful with it. "
            "Could you help me find somewhere to sit down safely?"
        ),
        recovery_cue="My left leg is feeling better. I'm going to try walking carefully."
    ),

    "right_leg_favour": ProtectivePosture(
        name="right_leg_favour",
        description="Weight shifted to left leg, right leg minimally loaded",
        affected_region="right_leg",
        joint_targets=[
            JointTarget(joint_id=40, target_rad=0.15, torque_limit=0.20, move_speed=0.10),
            JointTarget(joint_id=41, target_rad=0.20, torque_limit=0.20, move_speed=0.10),
            JointTarget(joint_id=42, target_rad=0.05, torque_limit=0.15, move_speed=0.10),
        ],
        compensation={
            "weight":     "left_dominant",
            "walking":    "reduced_right_stride",
            "standing":   "prefer_seated_if_available",
        },
        hold_torque=0.20,
        verbal_cue=(
            "My right leg isn't working right — I'm being careful with it. "
            "Could you help me find somewhere to sit down safely?"
        ),
        recovery_cue="My right leg is feeling better. I'm going to try walking carefully."
    ),

    "torso_stabilise": ProtectivePosture(
        name="torso_stabilise",
        description="Minimal torso movement, arms used for balance support",
        affected_region="torso",
        joint_targets=[],  # handled by MPC balance controller directly
        compensation={
            "movement":   "minimal",
            "arms":       "balance_assist",
        },
        hold_torque=0.30,
        verbal_cue=(
            "I'm having trouble with my balance — I need to stay still. "
            "Could you give me something to hold onto or help me sit down?"
        ),
        recovery_cue="My balance is better. I'm going to move carefully."
    ),

    "head_stabilise": ProtectivePosture(
        name="head_stabilise",
        description="Head held still, neck minimally actuated",
        affected_region="head",
        joint_targets=[
            # Neck: all axes to neutral and hold
            JointTarget(joint_id=60, target_rad=0.0,  torque_limit=0.20, move_speed=0.08),
            JointTarget(joint_id=61, target_rad=0.0,  torque_limit=0.20, move_speed=0.08),
        ],
        compensation={
            "gaze":    "body_rotation_instead",
            "gestures":"body_and_arm_only",
        },
        hold_torque=0.20,
        verbal_cue=(
            "My head movement is restricted right now — "
            "I need to keep it still while this is sorted out."
        ),
        recovery_cue="My head is feeling better."
    ),

    "both_arms_guard": ProtectivePosture(
        name="both_arms_guard",
        description="Both arms held in protective position",
        affected_region="both_arms",
        joint_targets=[
            JointTarget(joint_id=10, target_rad=0.3,  torque_limit=0.20, move_speed=0.12),
            JointTarget(joint_id=11, target_rad=1.57, torque_limit=0.20, move_speed=0.12),
            JointTarget(joint_id=20, target_rad=0.3,  torque_limit=0.20, move_speed=0.12),
            JointTarget(joint_id=21, target_rad=1.57, torque_limit=0.20, move_speed=0.12),
        ],
        compensation={
            "grasping":   "deferred",
            "locomotion": "use_legs_only",
        },
        hold_torque=0.20,
        verbal_cue=(
            "Both my arms have a problem — I'm holding them close. "
            "I cannot grasp anything right now and I need help."
        ),
        recovery_cue="My arms are feeling better."
    ),

    "minimal_movement": ProtectivePosture(
        name="minimal_movement",
        description="All movement minimised — spine issue",
        affected_region="spine",
        joint_targets=[],  # signal to motion planner to freeze
        compensation={"movement": "freeze"},
        hold_torque=0.0,
        verbal_cue=(
            "I have a spine or structural issue — I need to stop moving. "
            "Please help me."
        ),
        recovery_cue="My spine feels stable. I'll try moving very carefully."
    ),
}


class ProtectivePostureController:
    """
    Deploys and maintains protective postures when faults are detected.
    Integrates with the ZMQ bus to publish joint targets.
    Tracks active protections and their compensation effects.
    """

    def __init__(self, bus=None, speak_fn=None):
        self._bus      = bus
        self._speak    = speak_fn
        self._active:  Dict[str, ProtectivePosture] = {}
        self._lock     = threading.Lock()

    def protect(self, body_region: str, severity: FaultSeverity):
        """
        Adopt the protective posture for a body region.
        Gently moves the affected part to a safe position.
        """
        posture_name = self._region_to_posture(body_region)
        if not posture_name: return
        posture = POSTURES.get(posture_name)
        if not posture: return

        with self._lock:
            already_protected = posture_name in self._active
            self._active[posture_name] = posture

        if not already_protected:
            self._deploy(posture, severity)

    def release(self, body_region: str):
        """Release protection when fault is resolved."""
        posture_name = self._region_to_posture(body_region)
        if not posture_name: return
        with self._lock:
            posture = self._active.pop(posture_name, None)
        if posture:
            if self._speak:
                self._speak(posture.recovery_cue)
            if self._bus:
                self._bus.publish(b"SAFE_RELEASE_POSTURE", {
                    "region":  body_region,
                    "posture": posture_name,
                })
            logger.info(f"Protective posture released: {posture_name}")

    def _deploy(self, posture: ProtectivePosture, severity: FaultSeverity):
        """Execute the protective posture on hardware."""
        logger.info(f"Adopting protective posture: {posture.name}")

        # Verbal announcement
        if self._speak:
            self._speak(posture.verbal_cue)

        # Publish joint targets to spinal nodes
        if self._bus and posture.joint_targets:
            targets = [
                {"joint_id":    t.joint_id,
                 "target_rad":  t.target_rad,
                 "torque_limit":t.torque_limit,
                 "move_speed":  t.move_speed}
                for t in posture.joint_targets
            ]

            # Determine which spinal topic
            if "arm" in posture.affected_region or "hand" in posture.affected_region:
                topic = b"EFF_M1_ARM_L" if "left" in posture.affected_region else b"EFF_M1_ARM_R"
            else:
                topic = b"SPN_CPG"

            self._bus.publish(topic, {
                "protective_posture": posture.name,
                "joint_targets":      targets,
                "torque_override":    posture.hold_torque,
                "compensation":       posture.compensation,
                "source":             "protective_posture_controller",
            })

        # Publish compensation instructions to arm/leg controllers
        if self._bus and posture.compensation:
            self._bus.publish(b"EFF_COMPENSATION", {
                "active_posture": posture.name,
                "compensation":   posture.compensation,
                "affected_region":posture.affected_region,
            })

    def _region_to_posture(self, region: str) -> Optional[str]:
        return REGION_POSTURE.get(region)

    @property
    def active_protections(self) -> List[str]:
        with self._lock:
            return list(self._active.keys())

    def is_protected(self, region: str) -> bool:
        posture_name = self._region_to_posture(region)
        with self._lock:
            return posture_name in self._active
