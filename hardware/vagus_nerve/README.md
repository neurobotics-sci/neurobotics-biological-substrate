# Vagus Nerve — Physical Kill Switch
## Bubo v5550 — IEC 60204-1 Category 1 E-Stop

### What it does
Physically cuts the 12V Galvanic Barrier power rail to all servo controllers
within 2 seconds of any kill trigger — regardless of software state.

The relay is **normally-closed and normally-energised**:
- When the BeagleBoard is running and healthy → relay coil energised → NC contact
  closed → 12V flows to Galvanic Barrier → servos have power → Bubo can move
- When kill is triggered → relay coil de-energised → NC contact opens →
  12V rail cut → Galvanic Barrier output drops → all servo PWM lines go high-Z →
  servos enter zero-torque mode → Bubo cannot move

This is **fail-safe**: any power loss to the BeagleBoard automatically cuts body power.

### Two-stage sequence (Category 1)
```
t=0s    Kill trigger (button / software / watchdog)
        ↓
        Stage 1: SYS_EMERGENCY broadcast → all nodes → rest posture command
                 Arms lower (500ms), knees flex (balance), head neutral
t=2.0s  Stage 2: Hardware relay coil de-energised → NC opens → 12V cut
                 REGARDLESS of software state
```

### Component List

| Item | Part | Qty | Cost |
|------|------|-----|------|
| Power relay | Tyco/TE V23057-B0002-A101 (12V coil, 10A, SPDT) | 1 | $4.50 |
| Relay socket | Tyco P/N 1415546-1 | 1 | $2.00 |
| Optoisolator | 6N137 (relay drive via brain-side logic) | 1 | $0.40 |
| Kill button | Apem IPR3SAD2 (red mushroom, NC+NO, 22mm) | 1 | $12.00 |
| 555 timer | NE556N (dual, for 2s delay + monostable) | 1 | $0.50 |
| NPN transistor | 2N2222A (relay coil driver, 150mA) | 1 | $0.20 |
| Resistors | 1kΩ, 10kΩ, 100kΩ (555 timing network) | 5 | $0.10 |
| Capacitors | 100μF, 10μF, 100nF | 3 | $0.30 |
| DIN rail mount | Relay + fuse holder for 12V rail | 1 | $8.00 |
| **Total** | | | **~$28** |

### Schematic

```
  Brain side (3.3V logic)             Body side (12V servo)
  ─────────────────────               ──────────────────────
  BeagleBone GPIO P9_14               12V PSU ──→ Relay COM
                │                     Relay NC ──→ Galvanic Barrier 12V IN
                ↓                     Relay NO ──→ (not connected)
            [6N137 LED]
                │                     Relay coil (+) ←─ 12V body
            6N137 out                 Relay coil (-) ←─ 2N2222A collector
                │                     2N2222A emitter → SGND
            [NE556 input]             2N2222A base ←── 6N137 out via 1kΩ
                │
         2-second delay
                │
           [2N2222A base]

  Kill button (physical):
  P9_16 GPIO ──→ [10kΩ pullup to 3.3V] ──→ Button ──→ GND
  Press = GPIO goes LOW = triggers same 556 network
```

### Wiring the 2-second delay (NE556)

```
  NE556 Timer 1 (monostable, 2s):
    VCC = 3.3V
    GND = DGND (brain side)
    TRIGGER (pin 6): pulled HIGH, goes LOW on kill signal
    OUTPUT  (pin 5): LOW normally, goes HIGH for 2s on trigger
    R1 = 180kΩ (timing), C1 = 10μF
    t = 1.1 × R1 × C1 = 1.1 × 180000 × 0.00001 = 1.98s ≈ 2s

  Output goes to 2N2222A base (via 1kΩ) → drives relay coil transistor
  HIGH output = transistor ON = relay energised = body powered
  LOW output (after 2s delay expires) = transistor OFF = relay drops = body dead
```

### Testing procedure

1. Power on BeagleBoard, verify relay clicks (energised)
2. Run: `python3 -c "from bubo.vagus.vagus_nerve import VagusNerve; v=VagusNerve(sim_mode=False); v.start(); input('Press enter to test kill...'); v.fire('test'); import time; time.sleep(3)"`
3. Measure 12V rail with multimeter during test:
   - t=0: 12.0V (normal)
   - t=0 to t=2s: 12.0V (Stage 1, software safe-stop)
   - t=2s: 0V (Stage 2, relay opened)
4. Verify servos drop to zero torque at t=2s
5. Re-arm: `v.arm()`, verify 12V returns
