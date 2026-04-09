"""bubo/nodes/sensory/somatosensory — v10.17 S1"""
import time,json,logging,threading,numpy as np
from bubo.shared.bus.neural_bus import NeuralBus,T

# Add this line to force stdout printing!
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

logger=logging.getLogger("S1_Somatosensory")

ZONE_IDS=["hand_R_index","hand_R_thumb","hand_L_index","foot_R_sole","foot_L_sole",
           "face_lips","trunk_chest_R","arm_R_lower","arm_L_lower","leg_R_thigh"]

class S1Node:
    POLL_HZ=10
    def __init__(self,config):
        self.name="S1_Somatosensory"
        self.bus=NeuralBus(self.name, config["pub_port"], config.get("sub_endpoints", []))
        self._sero=0.5
        self._running=False
        
        # --- THE DIRECT NEURAL IMPLANT ---
        import zmq
        self.ctx = zmq.Context()
        self.bypass_sock = self.ctx.socket(zmq.SUB)
        self.bypass_sock.connect("tcp://127.0.0.1:5633")
        self.bypass_sock.setsockopt(zmq.SUBSCRIBE, b"VPL_RELAY")
    def _on_neuromod(self,msg):
        if "5HT" in msg.payload: self._sero=float(msg.payload["5HT"])
    def _on_debug_input(self, *args):
        # By using *args, we are completely immune to Signature/Type Errors
        logger.info(f"!!! CORTICAL ACTIVATION !!! S1 intercept: {args}")
    def _poll(self):
        import time
        import zmq
        import json
        iv = 1.0 / self.POLL_HZ
        while self._running:
            t0 = time.time()
            
            # --- READ FROM THE IMPLANT ---
            try:
                topic, payload = self.bypass_sock.recv_multipart(flags=zmq.NOBLOCK)
                data = json.loads(payload.decode())
                self.last_real_input = data.get("data", data)
                logger.info(f"!!! CORTICAL ACTIVATION !!! S1 felt: {self.last_real_input}")
            except zmq.Again:
                pass # No new data this 10Hz cycle
            
            # 1. Determine Input (Real or Dream)
            if hasattr(self, 'last_real_input'):
                inner_data = self.last_real_input.get('data', self.last_real_input)
                pressure = float(inner_data.get('pressure', 0.85))
                
                # Extract the region sent by the probe
                incoming_region = inner_data.get('region', 'left_forearm')
                
                # The Cortical Homunculus Translation Map
                region_map = {
                    "left_forearm": "arm_L_lower",
                    "right_forearm": "arm_R_lower"
                }
                
                # Dynamically set the target zone (with a safe fallback)
                target_zones = [region_map.get(incoming_region, "unknown_limb")] 
                
            else:
                pressure = None
                target_zones = ["area_3b"] # fallback

            # 2. Process and Publish
            for zone_id in target_zones:
                p = pressure if pressure is not None else 0.0
                
                # We can safely use T here because NeuralBus handles outbound properly
                self.bus.publish(b"TOUCH_SA1", {
                    "zone_id": zone_id, 
                    "pressure_N": p, 
                    "is_nociceptive": False
                })

            # 3. Rest cycle
            time.sleep(max(0, iv - (time.time() - t0)))
    def start(self):
        self.bus.start() 
        # ADD THE 'b' HERE! This perfectly matches the probe's byte-string.
        self.bus.subscribe(b"VPL_RELAY", self._on_debug_input)
        self._running = True
        threading.Thread(target=self._poll, daemon=True).start()
        logger.info(f"{self.name} v10.17 | Online | Listening for VPL_RELAY")
    def stop(self): self._running=False;self.bus.stop()

if __name__=="__main__":
    with open("config/cluster_config.json") as f: cfg=json.load(f)["somatosensory"]
    n=S1Node(cfg);n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
