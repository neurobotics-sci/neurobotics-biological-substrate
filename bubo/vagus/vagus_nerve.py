"""
bubo/vagus/vagus_nerve.py — Bubo v5550
Vagus Nerve Physical Kill Switch

═══════════════════════════════════════════════════════════════════════════════
BIOLOGICAL ANALOGY
═══════════════════════════════════════════════════════════════════════════════

The vagus nerve (cranial nerve X) is the primary parasympathetic outflow from
the brainstem. It carries "rest and digest" signals and can rapidly inhibit
heart rate (vagal tone). "Vasovagal syncope" — fainting from shock — occurs
when massive vagal activation drops heart rate and blood pressure, rendering
the body suddenly inert. This is the biological template for our kill switch:
a single nerve that can instantly shut down all bodily activity.

═══════════════════════════════════════════════════════════════════════════════
HARDWARE DESIGN — TWO-STAGE IEC 60204-1 CATEGORY 1 STOP
═══════════════════════════════════════════════════════════════════════════════

IEC 60204-1 (Safety of Machinery, Electrical Equipment) defines:
  Category 0: Immediate removal of power (uncontrolled stop)
  Category 1: Controlled stop, then power removal after standstill
  Category 2: Controlled stop, power maintained

For a bipedal robot, Category 0 is DANGEROUS:
  - Sudden power loss on a standing biped = fall (500N impact force)
  - Cannot retract arms first = collision hazard
  - Servo current at stall position → heat spike
Therefore: **Category 1 with 2-second timeout**

TWO-STAGE PROTOCOL:
  Stage 1 (Software, t=0):
    - "VAGUS_FIRE" signal received by all nodes via T.SYS_EMERGENCY
    - All nodes: cease motor commands, publish safe-stop posture
    - Spinal nodes: command rest posture (arms lower, knees flex to catch fall)
    - Cerebellum: switch to ZMP stabilisation mode only
    - Timer starts: t = 0

  Stage 2 (Hardware, t = 2s):
    - Hardware relay opens regardless of software state
    - Galvanic Barrier 12V body rail → DISCONNECTED
    - All servo PWM signals go HIGH-Z → servos enter zero-torque mode
    - No software command can prevent this — it is HARDWARE

CIRCUIT:
  ┌─────────────────────────────────────────────────────────────┐
  │  BRAIN SIDE (logic 3.3V)         │  BODY SIDE (12V servo)  │
  │                                  │                         │
  │  BeagleBoard GPIO P9_14 ─────→   │  Relay coil (+)         │
  │                          [6N137] │  Relay coil (-) → SGND  │
  │  Kill button ──────────→         │                         │
  │  (normally open, momentary)      │  NC contact ──→ 12V IN  │
  │                                  │  Common    ──→ Servo rail│
  │  Hardware timer (555/NE556):     │  NO contact ──→ (unused)│
  │    On kill signal: start 2s      │                         │
  │    After 2s: release relay coil  │                         │
  │    → relay opens → body dead     │                         │
  └──────────────────────────────────┴─────────────────────────┘

RELAY SPECIFICATION:
  Tyco Electronics TE Connectivity V23057-B0002-A101
  Coil: 12V, 150mA
  Contacts: SPDT, 10A @ 250VAC (more than adequate for 12V/20A servo rail)
  Normally Closed (NC) in energised state — power cut = relay drops → body dead
  This is a "fail-safe" design: any power loss to relay coil = motors disabled

═══════════════════════════════════════════════════════════════════════════════
SOFTWARE COMPONENT
═══════════════════════════════════════════════════════════════════════════════

The VagusNerve class monitors:
  1. T.SYS_EMERGENCY messages (software kill command from any node)
  2. Watchdog heartbeat timeout (if all nodes die → auto-kill)
  3. Direct GPIO button interrupt (physical momentary switch)
  4. Software API: vagus.fire(reason) — for graceful shutdown

On any trigger: initiates two-stage sequence, then drives GPIO to cut relay.
"""

import time, logging, threading
from pathlib import Path
from enum import Enum
from typing import Optional

logger = logging.getLogger("VagusNerve")

# Hardware GPIO for relay control
VAGUS_RELAY_GPIO = 48         # BeagleBoard P9_14 → through 6N137 → relay coil
KILL_BUTTON_GPIO = 49         # BeagleBoard P9_16 → momentary switch input
RELAY_ACTIVE_LOW = True       # LOW = relay energised = body powered

SAFE_STOP_WINDOW_S  = 2.0     # seconds for software safe-stop before hard cut
WATCHDOG_TIMEOUT_S  = 10.0    # seconds of no heartbeat → auto-kill

# Safe posture joint targets (published during Stage 1)
SAFE_POSTURE = {
    "gait_mode":   "stand",
    "motor_scale": 0.0,
    "arm_l":  [-0.1, 0.1, 0.0, 0.15, 0.0, 0.0, 0.0],
    "arm_r":  [-0.1, -0.1, 0.0, 0.15, 0.0, 0.0, 0.0],
    "neck":    0.0,
    "hip_flex":  0.0, "knee_flex": 0.05, "ankle_df": -0.02,
}


class VagusState(Enum):
    NORMAL      = "normal"
    STAGE1      = "stage1_software_stop"   # safe-stop in progress
    STAGE2      = "stage2_hard_cut"        # relay opened, body dead
    RECOVERING  = "recovering"             # relay re-energised, software restart


class GPIOInterface:
    """Abstraction over BeagleBoard sysfs GPIO."""
    def __init__(self, pin: int, direction: str, sim_mode: bool = True):
        self.pin  = pin; self.sim_mode = sim_mode; self._value = 1  # default HIGH
        self._path = Path(f"/sys/class/gpio/gpio{pin}/value")
        if not sim_mode:
            try:
                export = Path("/sys/class/gpio/export")
                if not self._path.exists():
                    export.write_text(str(pin)); time.sleep(0.05)
                Path(f"/sys/class/gpio/gpio{pin}/direction").write_text(direction)
                if direction == "in":
                    Path(f"/sys/class/gpio/gpio{pin}/edge").write_text("falling")
            except Exception as e:
                logger.warning(f"GPIO{pin} init: {e} — sim mode"); self.sim_mode = True

    def write(self, v: int):
        self._value = int(bool(v))
        if not self.sim_mode:
            try: self._path.write_text(str(self._value))
            except Exception as e: logger.error(f"GPIO{self.pin} write: {e}")

    def read(self) -> int:
        if self.sim_mode: return self._value
        try: return int(self._path.read_text().strip())
        except: return 1


class VagusNerve:
    """
    Physical kill switch controller.
    Ensures hardware body power is cut within 2s of any kill trigger.
    """

    def __init__(self, bus=None, sim_mode: bool = True):
        self._bus       = bus
        self._state     = VagusState.NORMAL
        self._relay     = GPIOInterface(VAGUS_RELAY_GPIO, "out", sim_mode)
        self._button    = GPIOInterface(KILL_BUTTON_GPIO, "in",  sim_mode)
        self._sim_mode  = sim_mode
        self._kill_reason: Optional[str] = None
        self._stage1_t  = 0.0
        self._last_hb   = time.time()
        self._running   = False
        self._lock      = threading.Lock()
        self._callbacks = []   # called at Stage 2 (hard cut)

        # Energise relay immediately (normally-closed = body powered when relay active)
        self._energise()
        logger.info(f"VagusNerve initialised | relay energised | sim={sim_mode}")

    def register_callback(self, fn):
        """Register function to call at Stage 2 (hard cut). Use for logging/alerts."""
        self._callbacks.append(fn)

    def heartbeat(self):
        """Call periodically from watchdog to reset auto-kill timer."""
        with self._lock: self._last_hb = time.time()

    def fire(self, reason: str = "software_command"):
        """
        Initiate kill sequence.
        Stage 1: broadcast safe-stop to all nodes.
        Stage 2: hard cut after SAFE_STOP_WINDOW_S seconds.
        """
        with self._lock:
            if self._state != VagusState.NORMAL:
                return  # already firing
            self._state       = VagusState.STAGE1
            self._kill_reason = reason
            self._stage1_t    = time.time()

        logger.warning(f"VAGUS NERVE FIRING: {reason} | Stage 1 — software safe-stop")

        # Broadcast emergency to all nodes
        if self._bus:
            self._bus.publish(b"SYS_EMERGENCY", {
                "type":    "vagus_nerve",
                "reason":  reason,
                "stage":   1,
                "timeout_s": SAFE_STOP_WINDOW_S,
                "safe_posture": SAFE_POSTURE,
                "timestamp_ns": int(time.time_ns()),
            })

        # Stage 1 is handled in _monitor_loop

    def arm(self):
        """Re-arm after manual recovery. Re-energises relay."""
        with self._lock:
            if self._state == VagusState.STAGE2:
                self._energise()
                self._state = VagusState.RECOVERING
                logger.info("VagusNerve: relay re-energised — recovering")
                self._state = VagusState.NORMAL

    def _energise(self):
        """Energise relay → body powered (NC contact closed)."""
        self._relay.write(0 if RELAY_ACTIVE_LOW else 1)

    def _cut(self):
        """De-energise relay → body dead (NC contact opens)."""
        self._relay.write(1 if RELAY_ACTIVE_LOW else 0)

    def _monitor_loop(self):
        while self._running:
            time.sleep(0.05)
            with self._lock:
                state  = self._state
                stage1_elapsed = time.time() - self._stage1_t
                hb_age = time.time() - self._last_hb

            # Check physical button
            if self._button.read() == 0:   # active low
                self.fire("physical_button")

            # Watchdog timeout auto-kill
            if state == VagusState.NORMAL and hb_age > WATCHDOG_TIMEOUT_S:
                self.fire(f"watchdog_timeout_{hb_age:.0f}s")

            # Stage 1 → Stage 2 transition
            if state == VagusState.STAGE1 and stage1_elapsed >= SAFE_STOP_WINDOW_S:
                with self._lock: self._state = VagusState.STAGE2
                logger.warning("VAGUS NERVE: STAGE 2 — HARD CUT — body relay opened")
                self._cut()
                for cb in self._callbacks:
                    try: cb(self._kill_reason)
                    except Exception: pass
                if self._bus:
                    self._bus.publish(b"SYS_EMERGENCY", {
                        "type":    "vagus_nerve",
                        "reason":  self._kill_reason,
                        "stage":   2,
                        "relay_cut": True,
                        "timestamp_ns": int(time.time_ns()),
                    })

    def on_emergency_message(self, msg):
        """Subscribe to T.SYS_EMERGENCY and trigger if vagus_request."""
        p = msg.payload
        if p.get("type") == "vagus_request":
            self.fire(p.get("reason", "remote_request"))

    @property
    def state(self) -> VagusState: return self._state

    @property
    def is_body_powered(self) -> bool:
        return self._state not in (VagusState.STAGE2,)

    def start(self):
        self._running = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()

    def stop(self):
        self._running = False
        self._cut()   # safe: de-energise on process exit
