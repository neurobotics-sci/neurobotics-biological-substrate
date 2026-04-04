# Bubo OSS+AGPL-3.0 Release v10.3 RC4 — Installation Guide

## Quick Start (Simulation Mode — No Hardware Needed)

```bash
git clone https://github.com/bubo-brain/bubo.git
cd bubo
pip install -e ".[full]" --break-system-packages
export BUBO_PROFILE=hardware_api
export BUBO_LLM_API_KEY=sk-ant-your-key-here
python3 simulation/full_loop_sim.py
```

## Four Deployment Profiles

| Profile | Hardware | LLM | Use When |
|---------|----------|-----|----------|
| `hardware_local` | 21 Jetson nodes | AGX Orin 70B | Full robot, max quality |
| `hardware_api`   | 20 Jetson nodes | LLM API | Full robot, no AGX Orin |
| `aws_local`      | 6 AWS EC2 | g5.12xl GPU | Cloud, privacy/volume |
| `aws_api`        | 5 AWS EC2 | LLM API | Cloud, cheapest |

## Deploy to All Four Targets

```bash
# Set your profile first
export BUBO_PROFILE=hardware_api   # or hardware_local, aws_local, aws_api

# Hardware deployment (edit deploy/ansible/inventories/hardware/hosts.ini first)
./deploy/deploy.sh hardware_api

# AWS deployment (requires AWS CLI configured)
export BUBO_LLM_API_KEY=sk-ant-...
./deploy/deploy.sh aws_api

# Dry run (no changes)
./deploy/deploy.sh hardware_api --check

# Status
./deploy/deploy.sh aws_api --status
```

## Prerequisites

### All profiles
```bash
pip install pyzmq msgpack numpy scipy pyyaml requests --break-system-packages
```

### Hardware profiles (Jetson nodes)
```bash
# On each Jetson node:
sudo apt install libzmq3-dev python3-pip python3-venv git -y
pip install pyzmq msgpack numpy scipy pyyaml --break-system-packages

# Dynamixel SDK (servo control)
pip install dynamixel-sdk --break-system-packages

# Whisper STT (broca/social nodes)
pip install openai-whisper --break-system-packages

# Piper TTS (broca node)
# See: https://github.com/rhasspy/piper/releases
wget https://github.com/rhasspy/piper/releases/latest/download/piper_linux_aarch64.tar.gz
tar xzf piper_linux_aarch64.tar.gz -C /usr/local/bin/
```

### AGX Orin LLM node (hardware_local only)
```bash
# Install llama.cpp with CUDA
git clone --depth=1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp
cd /opt/llama.cpp
cmake -B build -DLLAMA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
      -DLLAMA_CUDA_F16=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cp build/bin/llama-server /usr/local/bin/

# Download 70B model (~38GB, takes time)
mkdir -p /opt/bubo/models
cd /opt/bubo/models
pip install huggingface-hub --break-system-packages
huggingface-cli download bartowski/Meta-Llama-3-70B-Instruct-GGUF \
  Meta-Llama-3-70B-Instruct-Q4_K_M.gguf \
  --local-dir /opt/bubo/models/
```

### LLM API backend (hardware_api, aws_api)
```bash
# Get API key
export BUBO_LLM_API_KEY=sk-ant-...
# Or store in file:
mkdir -p /etc/bubo/secrets
echo "*sk-***-..." > /etc/bubo/secrets/llm_api_key
chmod 600 /etc/bubo/secrets/llm_api_key
```

### Cloud (AWS) profiles
```bash
pip install boto3 flask flask-socketio eventlet --break-system-packages
aws configure   # set your AWS credentials
```

## Running Individual Brain Nodes

Each node can be run independently for testing:

```bash
# Set profile
export BUBO_PROFILE=hardware_api

# Run any node (replace NODE_NAME with actual role)
python3 -m bubo.nodes.subcortical.hypothalamus.hypothalamus_node
python3 -m bubo.nodes.limbic.amygdala.amygdala_node
python3 -m bubo.nodes.cortex.pfc.pfc_node
# etc.
```

## Running the Full System

```bash
# 1. Health check first
./scripts/health_check/neural_health_check.sh

# 2. Hierarchical launch (tier 1 first)
python3 launch/brain_launch.py

# 3. Or launch with dry-run to check
python3 launch/brain_launch.py --dry-run

# 4. Simulation (no hardware required)
python3 simulation/full_loop_sim.py
```

## Directory Structure

```
bubo/                   Brain modules
  nodes/                All 21 brain region nodes
    cortex/             PFC, Broca, Insula, M1, Premotor, Social
    thalamus/           Thalamus-L, Thalamus-R, coordinator
    subcortical/        Hypothalamus, Cerebellum (CMAC), Basal Ganglia
    limbic/             Hippocampus (EKF-SLAM), Amygdala (LA/BA/CeA)
    sensory/            V1+saccadic masking, A1+VOR, S1+homunculus
    spinal/             Arms (Omnihand), Legs (CPG+ZMP)
    brainstem/          SC (SGBM+PPS), RF, Oculomotor (VOR)
    memory/             LTM (SQLite+FAISS), Association (TBW+MNS)
  brain/                v10000 cognitive modules
    self/               Autobiographical self-model + eigenself
    friendship/         Aristotelian deep friendship engine
    humor/              Joke detection + self-deprecating generation
    safety/             Social danger detector (crisis/manipulation/distress)
    learning/           Bloom taxonomy classifier and integrator
    idle/               Default Mode Network idle learner
    memory_manager/     Smart aging (Ebbinghaus + human weighting)
  shared/               Libraries: bus, HAL, kinematics, homunculus, profile
  llm/                  LLM router (LLM-agnostic API + local llama backends)
  balance/              MPC balance controller + terrain mapper
  rl/                   PPO residual gait learner
  slam/                 RTABMap 3D SLAM bridge
  emotion/              EmotionChip (14 emotions, VAE 200-dim, somatic markers)
  scheduler/            NALB thermal-social load balancer
  qualia/               Global Workspace + IIT Phi consciousness substrate
  safety/               Limp mode, nod-off, vagus nerve
  galvanic/             Galvanic barrier optoisolator abstraction
  vagus/                Physical kill switch controller
  speech/               Whisper STT + Piper TTS + prosody modulation
  web/                  Web knowledge search (Wikipedia + DuckDuckGo)
  social/               Social curiosity engine + latent emotion VAE
  memory/multimodal/    Multimodal LTM with FAISS HNSW
  cloud/                REST+WebSocket API gateway (cloud profiles)
  hw/                   Hardware abstraction layer (servo, GPIO)

profiles/               Four deployment YAML profiles
deploy/
  deploy.sh             Single entry point for all four targets
  ansible/              Ansible playbooks and roles
  cloudformation/       AWS CloudFormation templates
  scripts/              Bootstrap, inventory generation, cost control

config/                 Cluster config, DDS, VLAN, PREEMPT_RT
firmware/stm32/         STM32H7 servo co-processor C firmware
hardware/               PCB schematics (Galvanic Barrier, Vagus Nerve)
ansible/                Full Ansible deployment for hardware cluster
launch/                 Hierarchical 7-tier launcher
simulation/             Full-loop simulation (no hardware required)
docs/                   Architecture docs, feasibility, servo mapping
```

## API Keys Required

| Profile | Key needed | Where to get |
|---------|-----------|--------------|
| `hardware_api` | `BUBO_LLM_API_KEY` |  |
| `aws_api` | `BUBO_LLM_API_KEY` |  |
| `hardware_local` | None | — |
| `aws_local` | None (uses local GPU) | — |
| AWS deployment | AWS credentials | AWS IAM console |

## Getting Help

- GitHub Issues: https://github.com/neurobotics-sci/neurobotics-biological-substrate/issues
- arXiv paper: https://arxiv.org/abs/[TBD]
- Kenneth Renshaw: kenneth.renshaw@neuroboticssci.ai
