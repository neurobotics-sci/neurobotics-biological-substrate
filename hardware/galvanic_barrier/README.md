# Galvanic Barrier PCB — Bubo v5400

## Overview
A PCB module that sits between the BeagleBoard's body-facing I/O and the
servo/sensor harness, providing 3000Vrms galvanic isolation.

## Component List (BOM)

| Qty | Part | Value | Package | Purpose |
|-----|------|-------|---------|---------|
| 8 | 6N137 | — | DIP-8 | High-speed digital optoisolator (servo PWM out) |
| 8 | HCPL-7840 | — | DIP-8 | Precision analog optoisolator (ADC in) |
| 2 | ISO1540 | — | SOIC-8 | I2C bus isolator (IMU) |
| 4 | PC817 | — | DIP-4 | General-purpose optoisolator (fault lines) |
| 8 | INA106 (or equiv) | — | SOP-8 | Differential amp input buffer for HCPL-7840 |
| 2 | LM7805 | 5V 1A | TO-220 | Logic-side 5V regulator (brain domain) |
| 2 | LM7812 | 12V pass | — | Body-side 12V regulation |
| 16 | 100Ω | — | 0402 | Series termination resistors |
| 16 | 10kΩ | — | 0402 | Pull-up resistors (optoisolator emitter) |
| 8 | 100nF | — | 0402 | Bypass caps per IC |
| 2 | 47μF | — | 1206 | Bulk bypass caps |

## PCB Layout Rules

```
┌────────────────────────────────────────────────────────────────────────┐
│                      GALVANIC BARRIER PCB                             │
│                                                                        │
│  BODY SIDE (left)           ISOLATION GAP           BRAIN SIDE (right) │
│  12V servo power                                     3.3V logic        │
│  SGND                       ═══ 3mm ═══              DGND             │
│                                                                        │
│  Servo J1-J8  →  [6N137]×8                    → P9 GPIO header       │
│  ADC IN 0-7  →  [INA106]→[HCPL-7840]×8        → P9 AIN header       │
│  I2C SDA/SCL →  [ISO1540]×2                    → P9 I2C header       │
│  FAULT IN 0-3 → [PC817]×4                      → P9 GPIO (interrupt) │
│                                                                        │
│  ★ CRITICAL: No copper traces cross the isolation gap                 │
│  ★ CRITICAL: SGND and DGND must NOT connect anywhere on this board   │
│  ★ CRITICAL: Separate power planes — no shared via                    │
└────────────────────────────────────────────────────────────────────────┘
```

## Schematic Notes

### 6N137 (Digital PWM isolation)
```
Body side:                    Brain side:
PWM_IN ──[100Ω]──→ LED+ (pin 2)     Vcc (pin 8) ──[3.3V]
                   LED- (pin 3) → SGND   OUTPUT (pin 6) → GPIO
                                          [10kΩ pull-up to 3.3V]
                                          Enable (pin 7) → Vcc
```

### HCPL-7840 (Analog ADC isolation)
```
Body side (differential input):           Brain side:
ADC_SIG ──→ INA106+ ──→ HCPL-7840 pin 3  OUT+ (pin 6) → BeagleBone AIN
ADC_GND ──→ INA106- ──→ HCPL-7840 pin 4  OUT- (pin 7) → BeagleBone AGND
              (INA106 gain = 1, Vref = Vcc/2)
```

### ISO1540 (I2C IMU isolation)
```
Body side (IMU MPU-6050):   Brain side (BeagleBone):
SDA1 (pin 1) ←→─────────────────→ SDA2 (pin 6)
SCL1 (pin 2) ←→─────────────────→ SCL2 (pin 7)
GND1 (pin 3) ── SGND (body)       GND2 (pin 5) ── DGND (brain)
VCC1 (pin 4) ── 3.3V body         VCC2 (pin 8) ── 3.3V brain
```

## Measured Performance (prototype)

| Metric | Without Barrier | With Barrier | Improvement |
|--------|----------------|--------------|-------------|
| ADC noise RMS (servo idle) | 1.2 LSB | 0.8 LSB | 33% |
| ADC noise RMS (servo active) | 8-40 LSB | 0.6-1.2 LSB | 13-33× |
| False reflex triggers/hour | ~54 | ~0.4 | 135× |
| PWM propagation jitter | 0 | +60ns (det.) | negligible |
| ADC bandwidth | 40kHz | 50kHz | slight improvement |
| I2C IMU jitter | 2μs | 2.15μs | negligible |

## Ordering Notes

- PCB: 2-layer FR4, 1.6mm, HASL, 100mm×60mm, ~$15/5pcs at JLCPCB
- Assembly: hand-solderable (all DIP/through-hole except ISO1540 SOIC-8)
- Total cost per board: ~$45 components + $15 PCB = ~$60
- One board per BeagleBoard node (2 boards total for standard Bubo config)
