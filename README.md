# SBALF OSS Edition - Bubo Mark XVII Chassis Revision
# ===================================================

NOTE:
=====

Developed under DISA STIG-compliant Perforce; GitHub carries public OSS drops only.
This represents the released codebase for the Bubo Mark XVI Hardware version.
For more information contact kenneth.renshaw@neuroboticssci.ai or info@neuroboticssci.ai.

Sections:
=========

1. Introduction
2. License
3. Getting Started - Ansible deployment
4. Getting Started - Thalamocortical Loop Test ( "poke" and "move" test )
5. A Potpourri of Thoughts Regarding the Bubo Architecture

Current Neurobotics LLC Research & Development work is completely withheld,
except for a couple of hints for the astute reader *wink*.

Section 1. Introduction
=======================

* Tier 1 Reflex Systems
* Tier 2 Homeostatic Systems
* Tier 3 Limbic Systems      - Multimodal Long Term Memory IP Held
* Tier 4 Sensory Integration - Edge Sensory IP Held
* Tier 5 Motor Execution

## Excluded

* Tier 6 Planning
* Tier 7 Executive Cognition
* VSCC Proprietary Extensions # Vector Symbolic Cognitive Coprocessor - FPGA/RISC-V

Section 2. License
==================

AGPLv3

Commercial licensing available separately.

For more information contact us at info@neuroboticssci.ai

https://github.com/neurobotics-sci/neurobotics-biological-substrate/blob/main/LICENSE.md

Section 3. Getting Started - Ansible codebase deployment
========================================================

Read about neuroscience *grin*

At the least to get this codebase up and running you will need a solid understanding of Linux, some Ansible knowledge, and some Python/PIP knowledge.

The easiest path is to load up all the functional modules onto one x86-based node and run in a simulated manner for testing and development and then deploy to the edge Jetsons or whatever you choose instead. Our future directions is not NVIDIA-centric I assure you *grin*

Setup your ansible packages, your SSH user keychain and deploy it via ssh-copy-id to <user>@localhost if that is what you are doing, and then dive into the deep end here and see how the water is!

git clone https://github.com/neurobotics-sci/neurobotics-biological-substrate.git

Edit ansible/inventory/hosts.ini according to your deployment topology and node/arch variants.
At the very least change 192.168.10.112 to match your machine. It might work just that easily :)

ansible -i ansible/inventory/hosts.ini all -m ping                           # Check topology
ansible-playbook -i ansible/inventory/hosts.ini ansible/site_hardware.yml -C # Check mode
ansible-playbook -i ansible/inventory/hosts.ini ansible/site_hardware.yml    # For real

Here is an example topology ping check followed by a dry run:

(base) [kenneth@n150-bubo neurobotics-biological-substrate-PULL_TEST]$ ansible -i ansible/inventory/hosts.ini all -m ping

hippocampus | SUCCESS =>
    changed: false
    ping: pong
amygdala | SUCCESS =>
    changed: false
    ping: pong
somatosensory | SUCCESS =>
    changed: false
    ping: pong
m1 | SUCCESS =>
    changed: false
    ping: pong
cerebellum | SUCCESS =>
    changed: false
    ping: pong
basal-ganglia | SUCCESS =>
    changed: false
    ping: pong
cingulate | SUCCESS =>
    changed: false
    ping: pong
circadian_clock | SUCCESS =>
    changed: false
    ping: pong
dopaminergic_sys | SUCCESS =>
    changed: false
    ping: pong
ans_autonomic | SUCCESS =>
    changed: false
    ping: pong
vagus_nerve | SUCCESS =>
    changed: false
    ping: pong
hypothalamus | SUCCESS =>
    changed: false
    ping: pong
insula | SUCCESS =>
    changed: false
    ping: pong
thalamus-l | SUCCESS =>
    changed: false
    ping: pong
thalamus-r | SUCCESS =>
    changed: false
    ping: pong
dist-thalami | SUCCESS =>
    changed: false
    ping: pong
reticular | SUCCESS =>
    changed: false
    ping: pong

(base) [kenneth@n150-bubo neurobotics-biological-substrate-PULL_TEST]$ ansible-playbook -i ansible/inventory/hosts.ini ansible/site_hardware.yml -C|grep -E "TASK|PLAY"

PLAY [Brain — Bootstrap all hardware nodes] ************************************
TASK [Gathering Facts] *********************************************************
TASK [common : Update apt cache (Bypass Python 3.6 trap)] **********************
TASK [common : Install system packages (Bypass Python 3.6 trap)] ***************
TASK [common : Create brain user] **********************************************
TASK [common : Create directory structure] *************************************
TASK [common : Push SBALF Codebase from Perforce Workspace to Edge Nodes] ******
TASK [common : Fix SBALF Codebase Permissions (Post-Rsync)] ********************
TASK [common : Ensure brain user owns the pushed SBALF codebase] ***************
TASK [common : Create Python virtualenv] ***************************************
TASK [common : Upgrade core Python build tools] ********************************
TASK [common : Install Python dependencies (Architecture Specific)] ************
TASK [common : Install Brain package (editable)] *******************************
TASK [common : Write node identity] ********************************************
TASK [common : Write BRAIN_PROFILE to environment] *****************************
TASK [common : Generate self-signed TLS certificate] ***************************
TASK [common : 🗣️ Deploy TTS Models (Speech/Hearing Tier)] **********************
PLAY [Brain — Homeostatic tier (hypothalamus, thalami, insula)] ****************
TASK [Gathering Facts] *********************************************************
TASK [brain_node : Install node-specific Python packages] **********************
TASK [brain_node : Generate cluster config for this node] **********************
TASK [brain_node : Create systemd service for brain node] **********************
TASK [brain_node : Enable and start brain node] ********************************
TASK [brain_node : Wait for node heartbeat] ************************************
PLAY [Brain — Limbic tier (amygdala, hippocampus)] *****************************
TASK [Gathering Facts] *********************************************************
TASK [brain_node : Install node-specific Python packages] **********************
TASK [brain_node : Generate cluster config for this node] **********************
TASK [brain_node : Create systemd service for brain node] **********************
TASK [brain_node : Enable and start brain node] ********************************
TASK [brain_node : Wait for node heartbeat] ************************************
PLAY [Brain — Motor tier (cerebellum, BG, M1, premotor)] ***********************
TASK [Gathering Facts] *********************************************************
TASK [brain_node : Install node-specific Python packages] **********************
TASK [brain_node : Generate cluster config for this node] **********************
TASK [brain_node : Create systemd service for brain node] **********************
TASK [brain_node : Enable and start brain node] ********************************
TASK [brain_node : Wait for node heartbeat] ************************************
PLAY [Brain — Cortical tier (PFC-L, PFC-R)] ************************************
TASK [Gathering Facts] *********************************************************
TASK [brain_node : Install node-specific Python packages] **********************
TASK [brain_node : Generate cluster config for this node] **********************
TASK [brain_node : Create systemd service for brain node] **********************
TASK [brain_node : Enable and start brain node] ********************************
TASK [brain_node : Wait for node heartbeat] ************************************
PLAY [Brain — Spinal tier (arms, legs)] ****************************************
PLAY [Brain — Verify deployment] ***********************************************
TASK [Gathering Facts] *********************************************************
TASK [Check Python and ZMQ] ****************************************************
TASK [Deployment summary] ******************************************************
PLAY RECAP *********************************************************************

Check the deployed modules under /etc/systemd/system/brain-*.service via systemctl and debug the python
launch commands on your systems. I'd suggest running the command in the service file directly on the
command line, debug your issues, then move them under systemd management. Good luck! :)

Section 4. Getting Started - Thalamocortical Loop Test ( "poke" and "move" demo )
=================================================================================

This is a specific demo and test case to demonstrate the foundational hardware and autonomic layer for the Bubo robotics architecture.

Tiers 1-5: Kinematics, Sensor Fusion, Safety, Sim2Real, and the Physical Substrate.

🧠 Architecture Overview: The 5-Tier Sensory Baseline

This repository code references and implements the code needed for a Sterile Afferent Pathway, demonstrating how high-frequency physical stimuli are transformed into observable cortical perceptions within the Bubo framework.

The Data Flow Pipeline
======================

The system operates across multiple layers of the Bubo abstraction, ensuring total decoupling between the peripheral stimulus and the telemetry output.

    Level 0: Peripheral Stimulus (tools/thalamic_probe.py)

        Role: Emulates a spinal afferent nerve.

        Mechanism: Injects randomized pressure packets (0.70 N to 0.95 N) at 2 Hz directly into the VPL (Ventral Posterior Lateral) relay.

    Level 1: Cortical Processing (bubo/nodes/sensory/s1_node.py)

        Role: The Somatosensory Cortex (S1).

        Mechanism: Polls the relay at 10 Hz, applying cortical persistence to the input. It wraps raw data into a NeuralMessage envelope, injecting real-time neuromodulatory state (DA, 5HT, NE) before broadcasting.

    Level 2: Telemetry Bridge (telemetry/bubo_bridge.py)

        Role: The Translator.

        Mechanism: A ZeroMQ-to-HTTP bridge that scrapes the high-speed cortical broadcast and exposes it as a Prometheus-compatible gauge. It effectively bridges the gap between asynchronous robotics and synchronous observability.

Key Technical Patterns

    NeuralMessage Wrapping: All data is encapsulated with metadata including source, vlan, and neuromod levels.

    Temporal Decoupling: The stimulus (2 Hz) and the cortical broadcast (10 Hz) operate on independent clocks, simulating biological sensory sampling.

    Observability: Integrated with Prometheus/Grafana for real-time "Neural Sparklines."

For a more complete explanation of this demo including a video of me starting the modules in sequence and showing the outputs live on-screen is available at our website knowledge portal here:

https://neuroboticssci.ai/droid-demo/

To checkout the code used for this demo please use the following commands:

git clone https://github.com/neurobotics-sci/neurobotics-biological-substrate.git
git checkout 49208e405d07d1ec6c0f6f96273264f963a3b38e

cd neurobotics-sci

Part of what I do is teach people to fish not just hand them one, especially in advanced topics where AI overuse has tended to make people intellectually lazy *wink*. To that end I will point you at the files referenced in the demo and I'm sure through a little ora et labora ( pray and study ) you can understand what it is doing, how to do it yourself, and why exactly this is part of the paradigm shift that is Neurobotics.

neurobotics-biological-substrate/tools/thalamic_probe.py: # Equivalent of a stimulus probe inserted into the thalamocortical pathway

"""
Bubo Thalamic Probe v1.1
Injects simulated spinal sensory afferents directly into the VPL_RELAY.
Used for testing S1 Cortical responsiveness in a sterile baseline.
"""

import logging
import zmq
import json
import time
import sys

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

def run_probe():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    
    # We BIND here so the Thalamus (the subscriber) can connect to us.
    # Port 5602 is the standard Spinal-to-Thalamus sensory gateway.
    try:
        sock.bind("tcp://*:5633")
        print(f"[*] Thalamic Probe Active | Binding to Port 5633")
        print(f"[*] Emulating Spinal Afferent Pathways... (Ctrl+C to stop)")
    except zmq.error.ZMQError as e:
        print(f"[!] Critical Port Error: {e}")
        print("[!] Tip: Run 'fuser -k 5602/tcp' to clear zombies.")
        sys.exit(1)

    # Handshake buffer
    time.sleep(2)

    count = 0
    import random

    try:
        while True:
            count += 1
            # Randomly pick which arm to stimulate
            target_region = random.choice(["left_forearm", "right_forearm"])
            current_pressure = round(random.uniform(0.70, 0.95), 2)
        
            tickle = {
                "source": "spinal_probe_v1",
                "type": "SENSORY_AFFERENT",
                "data": {
                    "region": target_region,
                    "pressure": current_pressure,
                    "salience": 1.0,
                    "is_unexpected": True,
                    "timestamp_ns": time.time_ns()
                }
            }
        
            payload = json.dumps(tickle).encode()
            sock.send_multipart([b"VPL_RELAY", payload])
        
            if count % 10 == 0:
                print(f"[>] Pulse {count} delivered to {target_region}. Pressure: {current_pressure} N")
                
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n[*] Retracting probe... Cleaning up sockets.")
    finally:
        sock.close()
        ctx.term()

if __name__ == "__main__":
    run_probe()

neurobotics-biological-substrate/tools/wiretap.py: # Equivalent of a physical brain probe inserted into the thalamocortical region of the brain

import zmq

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
# Connect to the exact port the probe is broadcasting on
sock.connect("tcp://127.0.0.1:5633")
# Subscribe to EVERYTHING
sock.setsockopt(zmq.SUBSCRIBE, b"")

print("[*] Wiretap active on Port 5633. Waiting for packets...")
while True:
    topic, payload = sock.recv_multipart()
    print(f"\n[RECEIVED] Topic: {topic.decode()}")
    print(f"[PAYLOAD]  {payload.decode()}")

Section 5. A Potpourri of Thoughts Regarding the Bubo Architecture



#######
# EOF #
#######
