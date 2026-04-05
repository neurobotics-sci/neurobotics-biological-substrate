"""bubo/nodes/sensory/somatosensory — v10.17 S1"""
import time,json,logging,threading,numpy as np
from bubo.shared.bus.neural_bus import NeuralBus,T
logger=logging.getLogger("S1_Somatosensory")

ZONE_IDS=["hand_R_index","hand_R_thumb","hand_L_index","foot_R_sole","foot_L_sole",
           "face_lips","trunk_chest_R","arm_R_lower","arm_L_lower","leg_R_thigh"]

class S1Node:
    POLL_HZ=100
    def __init__(self,config):
        self.name="S1_Somatosensory";self.bus=NeuralBus(self.name,config["pub_port"],config["sub_endpoints"])
        self._sero=0.5;self._running=False
    def _on_neuromod(self,msg):
        if "5HT" in msg.payload: self._sero=float(msg.payload["5HT"])
    def _poll(self):
        iv=1.0/self.POLL_HZ
        while self._running:
            t0=time.time();t=time.time()
            for zone_id in ZONE_IDS:
                pressure=float(np.clip(np.random.exponential(0.05),0,15))
                temp=float(np.clip(np.random.normal(34.0,0.5),5,70))
                is_noci=temp>45.0 or pressure>8.0
                if is_noci:
                    adelta=float(np.clip((temp-45)/10,0,1)) if temp>45 else float(np.clip((pressure-8)/4,0,1))
                    c_rate=float(np.clip((temp-45)/5,0,1)) if temp>45 else float(np.clip((pressure-8)/2,0,1))
                    gate_level=float(1.0-0.5*self._sero)
                    topic=T.NOCI_HEAT if temp>45 else (T.NOCI_COLD if temp<10 else T.NOCI_MECH)
                    self.bus.publish(topic,{"zone_id":zone_id,"intensity":adelta*gate_level,"temperature_C":temp,
                        "pressure_N":pressure,"adelta_rate":adelta,"c_rate":c_rate,
                        "features":[adelta,temp/60,c_rate,pressure/10],"is_nociceptive":True,"source":"S1"})
                else:
                    self.bus.publish(T.TOUCH_SA1,{"zone_id":zone_id,"pressure_N":pressure,"temperature_C":temp,
                        "features":[pressure/10,temp/60],"is_nociceptive":False,"area_3b":float(np.tanh(pressure/4))})
            time.sleep(max(0,iv-(time.time()-t0)))
    def start(self):
        self.bus.start();self.bus.subscribe(T.SERO_RAPHE,self._on_neuromod)
        self._running=True;threading.Thread(target=self._poll,daemon=True).start()
        logger.info(f"{self.name} v10.17 | {len(ZONE_IDS)} zones | TRPV1 | gate-control | {self.POLL_HZ}Hz")
    def stop(self): self._running=False;self.bus.stop()

if __name__=="__main__":
    with open("/etc/bubo/config.json") as f: cfg=json.load(f)["somatosensory"]
    n=S1Node(cfg);n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
