"""bubo/nodes/brainstem/reticular_formation — v10.17"""
import time,json,logging,threading,numpy as np
from bubo.shared.bus.neural_bus import NeuralBus,T
logger=logging.getLogger("ReticularFormation")

class ReticuloSpinalNode:
    def __init__(self,config):
        self.name="ReticularFormation";self.bus=NeuralBus(self.name,config.get("pub_port",5660),config.get("sub_endpoints",[]))
        self._arousal=0.3;self._running=False
    def _on_cea(self,msg):
        fear=float(msg.payload.get("cea_activation",0))
        if fear>0.5: self._arousal=min(1.0,self._arousal+fear*0.5);self.bus.publish(T.RF_AROUSAL,{"arousal":self._arousal,"state":"threat"})
        a=msg.payload.get("action","")
        if any(w in a for w in ["forward","walk"]): self.bus.publish(T.MLR_LOCO,{"drive":0.6,"action":a})
    def start(self):
        self._running=True;logger.info(f"{self.name} v10.17 | arousal | MLR")
    def stop(self): self._running=False;self.bus.stop()

if __name__=="__main__":
    with open("/etc/bubo/config.json") as f: cfg=json.load(f).get("reticular_formation",{"pub_port":5660,"sub_endpoints":[]})
    n=ReticuloSpinalNode(cfg);n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
