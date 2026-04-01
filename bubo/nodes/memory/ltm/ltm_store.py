"""
bubo/nodes/memory/ltm/ltm_store.py — v11.14
Long-Term Memory: SQLite WAL, saliency scoring, multimodal binding, NREM3 consolidation.
"""
import time, json, logging, threading, sqlite3, numpy as np
from collections import deque
from pathlib import Path
from bubo.shared.bus.neural_bus import NeuralBus, T

logger = logging.getLogger("LTM")
DB_PATH = Path("/opt/bubo/data/bubo_ltm.db")


def saliency(ne, valence, retrievals, binding, cortisol):
    return float(np.clip(
        0.35*ne + 0.25*abs(valence) + 0.20*float(np.log1p(retrievals))
        + 0.10*binding + 0.10*cortisol, 0, 1))


class Binder:
    DIM = 64
    def __init__(self):
        rng = np.random.default_rng(42)
        self._Wv = rng.standard_normal((self.DIM,32))*0.1
        self._Wa = rng.standard_normal((self.DIM,16))*0.1
        self._Ws = rng.standard_normal((self.DIM,8))*0.1
    def bind(self, vf=None, af=None, sf=None):
        parts = []
        if vf and len(vf): parts.append(np.tanh(self._Wv@np.resize(np.array(vf,dtype=float),32)))
        if af and len(af): parts.append(np.tanh(self._Wa@np.resize(np.array(af,dtype=float),16)))
        if sf and len(sf): parts.append(np.tanh(self._Ws@np.resize(np.array(sf,dtype=float),8)))
        if not parts: return np.zeros(self.DIM), 0.0
        j = np.mean(parts,axis=0); j /= (np.linalg.norm(j)+1e-8)
        s = float(np.mean([np.dot(parts[i]/(np.linalg.norm(parts[i])+1e-8),
                                   parts[k]/(np.linalg.norm(parts[k])+1e-8))
                           for i in range(len(parts)) for k in range(i+1,len(parts))])) if len(parts)>=2 else 0.3
        return j, s


class LTMDB:
    def __init__(self, path=DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._c = sqlite3.connect(str(path), check_same_thread=False)
        self._c.execute("PRAGMA journal_mode=WAL"); self._c.execute("PRAGMA synchronous=NORMAL")
        self._c.execute("""CREATE TABLE IF NOT EXISTS ltm(
            id INTEGER PRIMARY KEY, trace_id TEXT UNIQUE, ts_ns INTEGER,
            saliency REAL, valence REAL, ne REAL, cortisol REAL,
            retrievals INTEGER DEFAULT 0, binding REAL, phase TEXT,
            slam_x REAL, slam_y REAL, emb BLOB, vf BLOB, af BLOB, desc TEXT)""")
        self._c.execute("CREATE INDEX IF NOT EXISTS idx_s ON ltm(saliency DESC)")
        self._c.commit()

    def insert(self, ep, emb, vf=None, af=None):
        sp = ep.get("slam_pose",[0,0])
        try:
            self._c.execute("""INSERT OR REPLACE INTO ltm
                (trace_id,ts_ns,saliency,valence,ne,cortisol,retrievals,binding,phase,
                 slam_x,slam_y,emb,vf,af,desc) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                ep.get("trace_id",f"ltm_{time.time_ns()}"),
                ep.get("timestamp_ns",time.time_ns()),
                ep.get("saliency",0.0), ep.get("valence",0.0), ep.get("ne",0.2),
                ep.get("cortisol",0.15), ep.get("retrieval_count",0),
                ep.get("binding_strength",0.0), ep.get("sleep_phase","wake"),
                float(sp[0]) if len(sp)>0 else 0.0, float(sp[1]) if len(sp)>1 else 0.0,
                emb.astype(np.float32).tobytes(),
                np.array(vf or [],dtype=np.float32).tobytes(),
                np.array(af or [],dtype=np.float32).tobytes(),
                ep.get("description",""),
            ))
            self._c.commit()
        except Exception as e: logger.error(f"LTM insert: {e}")

    def query(self, q_emb, k=5):
        rows = self._c.execute("SELECT trace_id,saliency,valence,emb FROM ltm ORDER BY saliency DESC LIMIT 300").fetchall()
        if not rows: return []
        qn = q_emb/(np.linalg.norm(q_emb)+1e-8); scored=[]
        for r in rows:
            e=np.frombuffer(r[3],dtype=np.float32)
            if len(e)!=64: continue
            scored.append({"trace_id":r[0],"saliency":r[1],"valence":r[2],
                           "similarity":float(np.dot(e/(np.linalg.norm(e)+1e-8),qn))})
        scored.sort(key=lambda x:-x["similarity"])
        for item in scored[:k]:
            self._c.execute("UPDATE ltm SET retrievals=retrievals+1 WHERE trace_id=?",(item["trace_id"],))
        self._c.commit(); return scored[:k]

    def prune(self, thr=0.15):
        c=self._c.execute("DELETE FROM ltm WHERE saliency<?", (thr,)); self._c.commit(); return c.rowcount
    def top(self, n=20):
        return [{"trace_id":r[0],"saliency":r[1],"valence":r[2],"slam_xy":[r[3],r[4]]}
                for r in self._c.execute("SELECT trace_id,saliency,valence,slam_x,slam_y FROM ltm ORDER BY saliency DESC LIMIT ?",(n,)).fetchall()]
    def stats(self):
        c=self._c.execute("SELECT COUNT(*),AVG(saliency) FROM ltm").fetchone()
        return {"total":c[0],"avg_saliency":c[1] or 0.0}


class LTMNode:
    def __init__(self, config):
        self.name="LTM_Store"
        self.bus=NeuralBus(self.name,config["pub_port"],config["sub_endpoints"])
        self.db=LTMDB(); self.binder=Binder()
        self._q=deque(maxlen=500); self._vfc={}; self._afc={}
        self._running=False

    def _on_cons(self,msg):
        for ep in msg.payload.get("episodes",[]): self._q.append(ep)
    def _on_v(self,msg):
        f=msg.payload.get("features",[])
        if f: self._vfc[round(time.time(),1)]=f
    def _on_a(self,msg):
        f=msg.payload.get("features",[])
        if f: self._afc[round(time.time(),1)]=f
    def _on_recall(self,msg):
        cue=msg.payload.get("cue",[])
        if not cue: return
        qe,_=self.binder.bind(vf=cue[:32])
        res=self.db.query(qe,k=5)
        self.bus.publish(T.HIPPO_RECALL,{"source":"LTM","results":res,"top":self.db.top(5),"timestamp_ns":time.time_ns()})
    def _on_prune(self,msg):
        n=self.db.prune(float(msg.payload.get("threshold",0.15)))
        logger.info(f"LTM pruned {n}")

    def _worker(self):
        while self._running:
            if not self._q: time.sleep(0.2); continue
            ep=self._q.popleft(); now=round(time.time(),1)
            vf=self._vfc.get(now) or self._vfc.get(round(now-0.1,1))
            af=self._afc.get(now) or self._afc.get(round(now-0.1,1))
            emb,strength=self.binder.bind(vf=vf,af=af)
            ne=ep.get("ne",0.2); emo=ep.get("emotion_tag",{})
            ep["saliency"]=saliency(ne,emo.get("valence",0),ep.get("retrieval_count",0),strength,ep.get("cortisol",0.15))
            ep["binding_strength"]=strength; ep["timestamp_ns"]=time.time_ns()
            self.db.insert(ep,emb,vf,af)

    def _stats(self):
        while self._running:
            time.sleep(60); s=self.db.stats()
            self.bus.publish(T.LTM_STATS,{**s,"timestamp_ns":time.time_ns()})

    def start(self):
        self.bus.start()
        self.bus.subscribe(T.LTM_CONSOLIDATE,self._on_cons)
        self.bus.subscribe(T.VISUAL_V1,self._on_v); self.bus.subscribe(T.AUDITORY_A1,self._on_a)
        self.bus.subscribe(T.HIPPO_RECALL,self._on_recall); self.bus.subscribe(T.LTM_PRUNE,self._on_prune)
        self._running=True
        threading.Thread(target=self._worker,daemon=True).start()
        threading.Thread(target=self._stats,daemon=True).start()
        logger.info(f"{self.name} v11.14 | SQLite WAL | {self.db.stats()['total']} eps")
    def stop(self): self._running=False; self.bus.stop()

if __name__=="__main__":
    with open("/etc/bubo/config.json") as f: cfg=json.load(f)["ltm_store"]
    n=LTMNode(cfg); n.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: n.stop()
