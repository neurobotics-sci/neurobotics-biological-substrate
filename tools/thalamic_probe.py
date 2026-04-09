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
