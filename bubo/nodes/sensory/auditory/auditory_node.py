from bubo.saccadic_masking.saccadic_masking import SaccadicMaskingController
"""bubo/nodes/sensory/auditory — v10.17 A1+Vestibular"""
import time,json,logging,threading,math,numpy as np
from bubo.shared.bus.neural_bus import NeuralBus,T
logger=logging.getLogger("AuditoryVestibular")
try: import smbus2;HAS_IMU=True
except: HAS_IMU=False

class CompFilter:
    TAU=5.0;VOR_GAIN=0.95
    def __init__(self): self._roll=0.0;self._pitch=0.0;self._yaw=0.0;self._t=time.time()
    def update(self,accel_g,gyro_deg_s):
        t=time.time();dt=max(t-self._t,0.001);self._t=t
        gx,gy,gz=[math.radians(v) for v in gyro_deg_s];ax,ay,az=accel_g
        alpha=self.TAU/(self.TAU+dt)
        norm=math.sqrt(ax**2+ay**2+az**2)
        if abs(norm-1.0)<0.3: ra=math.degrees(math.atan2(ay,az));pa=math.degrees(math.atan2(-ax,math.sqrt(ay**2+az**2)))
        else: ra=math.degrees(self._roll);pa=math.degrees(self._pitch)
        self._roll=alpha*(self._roll+gx*dt)+(1-alpha)*math.radians(ra)
        self._pitch=alpha*(self._pitch+gy*dt)+(1-alpha)*math.radians(pa)
        self._yaw=(self._yaw+gz*dt)
        g_vec=np.array([-math.sin(self._pitch),math.sin(self._roll)*math.cos(self._pitch),-math.cos(self._roll)*math.cos(self._pitch)])
        lin=(np.array([ax,ay,az])-g_vec)*9.81
        return {"roll_deg":float(math.degrees(self._roll)),"pitch_deg":float(math.degrees(self._pitch)),"yaw_deg":float(math.degrees(self._yaw)),
                "gyro_rps":[gx,gy,gz],"accel_g":list(accel_g),"linear_accel_g":(lin/9.81).tolist(),
                "vor_horiz_deg":float(-self.VOR_GAIN*gz*dt*180/math.pi),"jerk_mag":float(np.linalg.norm(lin)/9.81),
                "perturbation":float(np.linalg.norm(lin)/9.81)>0.15,"features":[float(math.degrees(self._roll))/45,float(math.degrees(self._pitch))/45,float(gz),0.0]}

class AuditoryVestibularNode:
    HZ=50
    def __init__(self,config):
        self.name="AuditoryVestibular";self.bus=NeuralBus(self.name,config["pub_port"],config["sub_endpoints"])
        self.filt=CompFilter();self.masking=SaccadicMaskingController('auditory');self._running=False
    def _read_imu(self):
        t=time.time()
        return [0.02*np.sin(t*2.1),0.01*np.sin(t*1.7),1.0+0.005*np.sin(t*3.3)],[1.0*np.sin(t*1.3),0.5*np.cos(t*0.9),0.3*np.sin(t*2.0)]
    def _loop(self):
        iv=1.0/self.HZ
        while self._running:
            t0=time.time();accel,gyro=self._read_imu();vest=self.filt.update(accel,gyro)
            self.bus.publish(T.VESTIBULAR,{**vest,"gyro":gyro,"accel":accel})
            t=time.time();pitch=float(math.radians(0))*100
            if self.masking.should_publish(T.AUDITORY_A1):
              self.bus.publish(T.AUDITORY_A1,{"pitch_hz":150+50*np.sin(t*0.5),"band_energy_mean":0.3,"rms_db":-20.0,
                "features":[0.3,0.3,(0)/90.0],"localisation":{"azimuth_deg":0.0,"itd_us":0.0,"ild_db":0.0}})
            time.sleep(max(0,iv-(time.time()-t0)))
    def start(self):
        self.bus.start()
        self.bus.subscribe(T.SC_SACCADE,lambda m:self.masking.trigger_suppression(m.payload))
        self._running=True;threading.Thread(target=self._loop,daemon=True).start()
        logger.info(f"{self.name} v10.17 | CompFilter(tau={CompFilter.TAU}s) | {self.HZ}Hz")
    def stop(self): self._running=False;self.bus.stop()

if __name__=="__main__":
    with open("/etc/bubo/config.json") as f: cfg=json.load(f)["auditory_vestibular"]
    n=AuditoryVestibularNode(cfg);n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
