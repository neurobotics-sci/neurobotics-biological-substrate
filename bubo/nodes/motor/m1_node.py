"""bubo/nodes/motor/m1_node.py — v1.0 M1 Primary Motor Cortex"""
import zmq
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("M1_MotorCortex")

def run_m1():
    ctx = zmq.Context()
    
    # 1. The Cortico-Cortical Listener (Listening to S1)
    sub_sock = ctx.socket(zmq.SUB)
    sub_sock.setsockopt(zmq.CONFLATE, 1)  # Only keep the freshest thought!
    sub_sock.connect("tcp://127.0.0.1:5632") # Tapping into S1's broadcast
    sub_sock.setsockopt(zmq.SUBSCRIBE, b"TOUCH_SA1")
    
    # 2. The Pyramidal Tract Emitter (Descending to Spine)
    pub_sock = ctx.socket(zmq.PUB)
    pub_sock.bind("tcp://*:5634") # Port 5634 is our new Descending Spinal line
    
    logger.info("[*] M1 Primary Motor Cortex Online")
    logger.info("[*] Listening to S1 (5632) -> Projecting to Spinal Cord (5634)")
    
    try:
        while True:
            # Wait for sensory data to arrive from S1
            topic, payload = sub_sock.recv_multipart()
            data = json.loads(payload.decode())
            
            # Unpack S1's neural envelope
            inner = data.get("payload", data)
            zone = inner.get("zone_id")
            pressure = inner.get("pressure_N", 0.0)
            
            # --- VOLUNTARY MOTOR LOGIC ---
            # If the left arm gets poked hard enough (> 0.85N), we move it away.
            if zone == "arm_L_lower" and float(pressure) > 0.85:
                logger.info(f"[!] Salient stimulus on {zone} ({pressure}N). Formulating motor intent.")
                
                motor_cmd = {
                    "source": "M1_MotorCortex",
                    "target": "Ventral_Horn_C5_T1", # Biological routing to the Brachial Plexus
                    "payload": {
                        "actuator_group": "arm_L_flexors",
                        "force_vector": 0.8, # 80% contraction force
                        "duration_ms": 250
                    },
                    # Note: Action requires Dopamine (DA) and Acetylcholine (ACh)
                    "neuromod": {"DA": 0.8, "ACh": 0.9}, 
                    "timestamp_ns": time.time_ns()
                }
                
                # Fire the command down the Pyramidal Tract
                pub_sock.send_multipart([b"MOTOR_DESCENDING", json.dumps(motor_cmd).encode()])
                logger.info("[>] Pyramidal tract fired: arm_L_flexors contracted.")
                
                # Neural refractory period (we don't want to spasm)
                time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("\n[*] Shutting down M1.")
    finally:
        sub_sock.close()
        pub_sock.close()
        ctx.term()

if __name__ == "__main__":
    run_m1()
