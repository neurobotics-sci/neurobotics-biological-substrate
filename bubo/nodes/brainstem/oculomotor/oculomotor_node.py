"""
bubo/nodes/brainstem/oculomotor/oculomotor_node.py — v50.0
Oculomotor nucleus: VOR + saccades + smooth pursuit + blink reflex.
Runs on spinal-arms Nano 2GB (co-located with arm servo controller)
or independently — the VOR loop is latency-critical.

Integrates:
  1. VOR (vestibulo-ocular reflex): IMU gyro → compensatory eye movement
  2. Saccades: SC priority map → ballistic gaze shifts (VOR suppressed)
  3. Smooth pursuit: MT optic flow → OKR slow-phase eye movement
  4. Blink reflex: SC proximity + audio startle → eyelid servo close
  5. Pupillary light reflex: V1 luminance → camera gain (iris analogue)
"""
import time, json, logging, threading
import numpy as np
from bubo.shared.bus.neural_bus import NeuralBus, T
from bubo.vor.vor_controller import VORController

logger = logging.getLogger("Oculomotor")


class SaccadeController:
    """
    Ballistic eye movement to acquired target.
    During saccade: VOR suppressed (VOR_SUPPRESS_MS window).
    After saccade: fixation hold (VOR maintains gaze on new target).
    """
    SACCADE_VEL_DPS = 400.0   # peak saccade velocity (biological: 400-700°/s)
    MIN_AMP_DEG     = 0.5      # ignore micro-saccades < 0.5°

    def __init__(self, vor: VORController, bus: NeuralBus):
        self._vor = vor; self._bus = bus
        self._active = False
        self._target_h = 0.0; self._target_v = 0.0
        self._t_start  = 0.0; self._duration = 0.0

    def command(self, h_deg: float, v_deg: float):
        """Initiate saccade to (h_deg, v_deg) from current fixation."""
        amp = float(np.hypot(h_deg, v_deg))
        if amp < self.MIN_AMP_DEG: return
        # Saccade duration: ~2ms/degree (Main Sequence, Bahill 1975)
        self._duration  = amp / self.SACCADE_VEL_DPS
        self._target_h  = h_deg; self._target_v = v_deg
        self._active    = True; self._t_start = time.time()
        self._vor.suppress(duration_ms=self._duration * 1000 + 30)  # +30ms margin
        logger.debug(f"Saccade: ({h_deg:.1f}°, {v_deg:.1f}°) dur={self._duration*1000:.1f}ms")

    def step(self) -> dict:
        if not self._active: return {}
        elapsed = time.time() - self._t_start
        if elapsed >= self._duration:
            self._active = False
            return {"saccade_complete": True, "h_deg": self._target_h, "v_deg": self._target_v}
        t_norm = elapsed / self._duration
        # Velocity profile: bell-shaped (main sequence)
        vel_profile = np.sin(t_norm * np.pi)
        h = self._target_h * t_norm
        v = self._target_v * t_norm
        return {"saccade_active": True, "h_deg": h, "v_deg": v, "vel_profile": float(vel_profile)}


class BlinkController:
    """
    Corneal blink reflex + voluntary blink rate.
    Triggers: SC proximity (depth < 5cm face region), loud audio, looming.
    Blink duration: ~150ms (biological).
    Blink rate: 10-20 blinks/min at rest, modulated by cognitive load.
    """
    BLINK_DURATION_MS = 150.0
    BLINK_INTERVAL_S  = 5.0   # spontaneous blink interval

    def __init__(self, bus: NeuralBus):
        self._bus = bus; self._blinking = False
        self._blink_until_ns = 0; self._t_last_blink = time.time()

    def trigger(self, reason="reflex"):
        self._blinking = True
        self._blink_until_ns = time.time_ns() + int(self.BLINK_DURATION_MS * 1e6)
        self._t_last_blink = time.time()
        self._bus.publish(T.REFLEX_BLINK, {
            "reason": reason, "duration_ms": self.BLINK_DURATION_MS,
            "timestamp_ns": time.time_ns()})

    def step(self) -> bool:
        """Returns True if eyes should be closed."""
        if self._blinking and time.time_ns() > self._blink_until_ns:
            self._blinking = False
        # Spontaneous blink
        if time.time() - self._t_last_blink > self.BLINK_INTERVAL_S:
            self.trigger("spontaneous")
        return self._blinking


class OculomotorNode:
    HZ = 200  # run at VOR rate

    def __init__(self, config: dict):
        self.name = "Oculomotor"
        self.bus  = NeuralBus(self.name, config["pub_port"], config["sub_endpoints"])
        self.vor  = VORController(self.bus)
        self.sac  = SaccadeController(self.vor, self.bus)
        self.blink = BlinkController(self.bus)

        self._gyro_rps    = [0.0, 0.0, 0.0]
        self._luminance   = 0.5
        self._looming     = False
        self._audio_loud  = False
        self._okr_h = self._okr_v = 0.0
        self._running = False
        self._lock = threading.Lock()

    def _on_vestibular(self, msg):
        g = msg.payload.get("gyro_rps", [0,0,0])
        with self._lock: self._gyro_rps = g
        self.vor.update_vestibular(g)

    def _on_sc_saccade(self, msg):
        """SC sends target pixel → convert to eye degrees."""
        target = msg.payload.get("target_px", [320, 240])
        cx, cy = 320, 240
        h_deg = float((target[0] - cx) / cx * 45.0)
        v_deg = float((cy - target[1]) / cy * 35.0)
        self.sac.command(h_deg, v_deg)

    def _on_mt(self, msg):
        """Optic flow → OKR contribution."""
        evs = msg.payload.get("motion_events", [])
        if evs:
            # Mean motion direction → smooth pursuit signal
            dx = float(np.mean([e.get("vel_pxf", 0) for e in evs]) / 50.0)
            with self._lock: self._okr_h = -dx * 10  # deg/s (opposite motion)
        # Blink on looming
        if msg.payload.get("looming_alert"):
            with self._lock: self._looming = True
            self.blink.trigger("looming")
        else:
            with self._lock: self._looming = False

    def _on_visual(self, msg):
        lum = msg.payload.get("colour", {}).get("luminance", 0.5)
        with self._lock: self._luminance = float(lum)
        # PLR: adjust camera gain (iris analogue)
        # gain_target inversely proportional to luminance
        cam_gain = float(np.clip(1.0 - 0.7 * lum, 0.1, 1.0))
        self.bus.publish(T.REFLEX_PLR, {
            "luminance": lum, "cam_gain_target": cam_gain,
            "timestamp_ns": time.time_ns()
        })

    def _on_audio(self, msg):
        rms = float(msg.payload.get("rms_db", -60))
        if rms > -20:  # loud sound → blink
            self.blink.trigger("acoustic_startle")
            self.bus.publish(T.REFLEX_BLINK, {
                "reason": "acoustic", "rms_db": rms, "timestamp_ns": time.time_ns()})

    def _on_vor_suppress(self, msg):
        dur = float(msg.payload.get("duration_ms", 120))
        self.vor.suppress(dur)

    def _loop(self):
        interval = 1.0 / self.HZ
        while self._running:
            t0 = time.time()
            with self._lock:
                okr_h = self._okr_h; okr_v = self._okr_v
            self.vor.update_optic_flow(okr_h, okr_v)
            sac_status = self.sac.step()
            eye_closed  = self.blink.step()
            if sac_status:
                self.bus.publish(T.SC_SACCADE, {
                    **sac_status, "source": "oculomotor", "timestamp_ns": time.time_ns()})
            if eye_closed:
                # Zero eye velocity commands during blink
                self.vor.suppress(10)
            time.sleep(max(0, interval - (time.time() - t0)))

    def start(self):
        self.bus.start()
        self.vor.start()
        self.bus.subscribe(T.VESTIBULAR,     self._on_vestibular)
        self.bus.subscribe(T.SC_SACCADE,     self._on_sc_saccade)
        self.bus.subscribe(T.VISUAL_MT,      self._on_mt)
        self.bus.subscribe(T.VISUAL_V1,      self._on_visual)
        self.bus.subscribe(T.AUDITORY_A1,    self._on_audio)
        self.bus.subscribe(T.VOR_SUPPRESS,   self._on_vor_suppress)
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info(f"{self.name} v50.0 | VOR({VORController.HZ}Hz) | saccades | blink | PLR")

    def stop(self):
        self.vor.stop(); self._running = False; self.bus.stop()


if __name__ == "__main__":
    with open("/etc/bubo/config.json") as f: cfg = json.load(f)["oculomotor"]
    n = OculomotorNode(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
