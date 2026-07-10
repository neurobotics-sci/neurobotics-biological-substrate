Thalamocortical Loop Test ( "poke" and "move" demo )
====================================================

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
