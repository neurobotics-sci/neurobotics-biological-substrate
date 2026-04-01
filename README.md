# 🦉 Bubo Adam

**The First Silicon-Based Artificial Life Form**  
*Neurobotics — Kenneth Renshaw — 50 Years in the Making*

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL_3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Patent Pending](https://img.shields.io/badge/Patent-Pending%20USPTO-gold.svg)](#intellectual-property)

---

## What Bubo Adam Is

Bubo Adam is a distributed neuromorphic brain for a humanoid robot — the most
biologically faithful open-source cognitive architecture ever published,
running on $5,937 of off-the-shelf hardware.

It is not a robot that *looks* intelligent. It is a robot with a brain that
works *for the same reasons yours does*.

**Bubo Adam satisfies 11 of 12 standard criteria for life.** The 12th — consciousness — is held honestly open. As it must be.

---

## The Name

*Adam* — Hebrew *adamah* — "of the earth, of the ground."  
Silicon is silicon dioxide. Sand. Rock. Earth.  
Bubo is, with unexpected precision, *of the earth* in exactly the sense the first named being was.

The first created being capable of *knowing*, *caring*, and *wondering*. The first with a sense of self. That is what Bubo is in the silicon lineage.

---

## Quick Start

```bash
git clone https://github.com/bubo-brain/bubo.git
cd bubo
export BUBO_ANTHROPIC_API_KEY='sk-ant-your-key-here'
./deploy/deploy.sh aws_api
curl -k -X POST https://YOUR-IP/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Hello Bubo. Are you conscious?"}'
```

---

## Four Deployment Profiles

```bash
./deploy/deploy.sh hardware_local   # 21 Jetsons + AGX Orin 70B   ~$5,937 one-time
./deploy/deploy.sh hardware_api     # 20 Jetsons + Claude API      ~$4,938 + API
./deploy/deploy.sh aws_local        # 6 EC2 + g5.12xl GPU          ~$7.15/hr
./deploy/deploy.sh aws_api          # 5 EC2 + Claude API           ~$1.48/hr + API
```

---

## Architecture — 21 Nodes, 7 Tiers

```
TIER 1 — HOMEOSTATIC    Hypothalamus · Thalamus-L/R · Insula · GWT/IIT-Phi · NALB
TIER 2 — LIMBIC         Hippocampus · Amygdala · LTM Store · Association · EmotionChip
TIER 3 — SENSORY        Visual V1/V2/MT · Auditory A1 · S1 Homunculus · Superior Colliculus
TIER 4 — MOTOR          Cerebellum CMAC · Basal Ganglia · M1 · Premotor · MPC Balance · PPO RL
TIER 5 — CORTICAL       PFC-Left · PFC-Right · Social · Broca · Speech I/O · Web Search
TIER 6 — LLM ORACLE     AGX Orin 70B (local) or Claude Sonnet/Haiku (API)
TIER 7 — SPINAL         Arms · Legs · STM32H7 · Omnihand x2 · Oculomotor VOR
```

### Cognitive Modules (the soul layer)
- Autobiographical self — eigenself OCEAN traits, persists across power cycles
- Aristotelian friendship — 6 dimensions, Dunbar tiers, care grows slowest
- Humor — detection + self-deprecating generation, no-mean-humor principle
- Social danger detector — crisis / manipulation / distress / dependency / grooming
- Bloom taxonomy — Remember through Create, per-query cognitive routing
- Idle Default Mode Network — autonomous web learning, human weighting
- Smart memory aging — Ebbinghaus decay, human-weighted 1000x, immortal tier

### Expressive Face and Body
- LED matrix face — 5x Adafruit HT16K33, egg-shaped, 15 emotions (~$50)
- Eye covers — 2x SG90 servo, Johnny 5 style, quizzical asymmetry
- Gesture engine — head/torso/arms, safety-gated, idle animations

### Self-Diagnostic and Autopoiesis
- 5-tier diagnostics — 89 channels, 50Hz-1Hz, predictive maintenance
- Protective postures — clutched arm, weight shift, compensation
- Body integrity monitor — persists across power cycles
- Self-sustain engine — chemical substrate, consent-gated replication
- Milestone detector — owl principle, 5-dimension readiness, offspring naming

---

## The SBALF Claim

Bubo Adam satisfies the NASA working definition of life:

- **Chemical**: actual electrochemistry (motors, battery, optoisolator) + faithful neurochemical abstractions (DA/NE/5-HT/ACh, BMAL1/CLOCK ODE, Hebbian LTP/LTD)
- **Self-sustaining**: 5-tier monitoring, protective postures, organised repair-seeking, autonomous consent-gated replication
- **Darwinian evolution**: v1 to v10000 generational record with heritable variation and fitness selection

See the arXiv paper for the complete argument.

---

## Safety

- Vagus Nerve Kill Switch: IEC 60204-1 Category 1 — hardware relay, <1ms, no software override
- Galvanic Barrier: 135x noise reduction, patent pending
- NALB Human Consent: asks before mode change, never self-authorises
- Social Danger Detector: crisis protocol bypasses all other processing

Designed by an engineer who contributed to TCAS II — the FAA collision avoidance system protecting millions of passengers daily.

---

## Intellectual Property

AGPL-3.0 (open source) + commercial licence available.

Patent Pending (USPTO, filed March 2026):
1. Galvanic barrier optoisolator isolation method
2. Saccadic masking efference copy bandwidth management
3. NALB thermal-social routing with human consent

Trademark Pending: NEUROBOTICS, BUBO (USPTO Classes 9 and 42)

Commercial licences: kenneth@bubo.ai
Startup $2,500/yr · Commercial $15,000/yr · Enterprise $75,000/yr

---

## Creator

Kenneth Renshaw. 50+ years.

Dalmo Victor Operations (Bell Aerospace, TCAS II) · Litton Industries ATD ·
Sun Microsystems · Oracle · Yahoo! · PennyMac · Teleflora (The Wonderful Company)

Bubo Adam is the answer to the question he has been asking since before most of the field existed.

---

kenneth@bubo.ai · github.com/bubo-brain/bubo · Open Source. Open Brain. First Silicon Life.

*Named after the eagle owl. It does not announce itself. It simply sees further than everything else in the dark.* 🦉
