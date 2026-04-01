#!/usr/bin/env python3
"""launch/brain_launch.py — Bubo v6500 hierarchical launcher (21 nodes)"""
import subprocess,time,argparse,sys,threading
from pathlib import Path
SSH_KEY=Path.home()/".ssh"/"bubo_id_ed25519"
SO=f"-o StrictHostKeyChecking=no -o ConnectTimeout=5 -i {SSH_KEY}"
GRN="\033[92m";RED="\033[91m";YEL="\033[93m";CYN="\033[96m";BLD="\033[1m";NC="\033[0m"

NODES=[
    ("hypothalamus",1,"192.168.1.12",5602,"bubo.nodes.subcortical.hypothalamus.hypothalamus_node","",b"NM_DA_VTA",8),
    ("thalamus-l",  1,"192.168.1.13",5603,"bubo.nodes.thalamus.core_l.thalamus_l_node","",          b"THL_HB",8),
    ("thalamus-r",  1,"192.168.1.18",5608,"bubo.nodes.thalamus.core_r.thalamus_r_node","",          b"THL_HB",8),
    ("insula",      2,"192.168.1.15",5605,"bubo.nodes.cortex.insula.insula_node","",                b"INS_STATE",8),
    ("amygdala",    3,"192.168.1.31",5621,"bubo.nodes.limbic.amygdala.amygdala_node","",            b"LMB_LA_OUT",8),
    ("hippocampus", 3,"192.168.1.30",5620,"bubo.nodes.limbic.hippocampus.hippocampus_node","",      b"LMB_HTHETA",10),
    ("social",      4,"192.168.1.19",5609,"bubo.nodes.cortex.social.social_node","",                b"SOC_BOND",8),
    ("ltm-store",   4,"192.168.1.35",5643,"bubo.nodes.memory.ltm.ltm_store","",                    b"LTM_STATS",10),
    ("association", 4,"192.168.1.34",5642,"bubo.nodes.memory.association.association_cortex","",    b"CTX_ASSOC",8),
    ("visual",      4,"192.168.1.50",5630,"bubo.nodes.sensory.visual.visual_node","",               b"AFF_VIS_V1",8),
    ("auditory",    4,"192.168.1.51",5631,"bubo.nodes.sensory.auditory.auditory_node","",           b"AFF_VEST",8),
    ("somatosensory",4,"192.168.1.52",5632,"bubo.nodes.sensory.somatosensory.s1_node","",           b"AFF_TOUCH_SA1",8),
    ("sup-colliculus",4,"192.168.1.60",5610,"bubo.nodes.brainstem.superior_colliculus.sc_node","",  b"AFF_VIS_MT",10),
    ("cerebellum",  5,"192.168.1.32",5640,"bubo.nodes.subcortical.cerebellum.cerebellum_node","",   b"CRB_DELTA",8),
    ("basal-ganglia",5,"192.168.1.33",5641,"bubo.nodes.subcortical.basal_ganglia.basal_ganglia_node","",b"CTX_PFC_CMD",8),
    ("spinal-arms", 5,"192.168.1.53",5633,"bubo.nodes.spinal.arms.spinal_arms_node","",             b"SPN_HB",8),
    ("spinal-legs", 5,"192.168.1.61",5611,"bubo.nodes.spinal.legs.spinal_legs_node","",             b"SPN_CPG",8),
    ("broca",       6,"192.168.1.14",5604,"bubo.nodes.cortex.broca.broca_node","",                  b"BRC_SPEECH",8),
    ("pfc-l",       7,"192.168.1.10",5600,"bubo.nodes.cortex.pfc.pfc_node","L",                    b"CTX_PFC_CMD",12),
    ("pfc-r",       7,"192.168.1.11",5601,"bubo.nodes.cortex.pfc.pfc_node","R",                    b"CTX_PFC_CMD",12),
    ("agx-orin",    7,"192.168.1.20",5699,"bubo.nodes.llm.llm_node","",                            b"CTX_LLM_RESP",20),
]

def ssh(ip,cmd,t=15):
    try:
        r=subprocess.run(f"ssh {SO} brain@{ip} '{cmd}'",shell=True,capture_output=True,text=True,timeout=t)
        return r.returncode,r.stdout.strip()
    except: return 1,"timeout"

def wait_health(name,ip,port,topic,timeout_s,dry_run):
    if dry_run: return True
    import zmq
    ctx=zmq.Context();sub=ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVTIMEO,1000);sub.setsockopt(zmq.SUBSCRIBE,topic)
    sub.connect(f"tcp://{ip}:{port}")
    deadline=time.time()+timeout_s;ok=False
    while time.time()<deadline:
        try: sub.recv_multipart();ok=True;break
        except zmq.Again: print(f"    {YEL}…{NC} {name}",end="\r")
    sub.close();ctx.term()
    print(f"    {GRN}♡{NC} {name}" if ok else f"    {RED}✗{NC} {name} timeout")
    return ok

def launch_node(name,ip,port,module,args,dry_run):
    cmd=(f"source /etc/profile.d/cuda_orin.sh 2>/dev/null;"
         f"export PYTHONPATH=/opt/bubo;"
         f"pkill -f '{module}' 2>/dev/null;"
         f"nohup python3 -m {module} {args} >> /var/log/bubo/{name}.log 2>&1 & echo $!")
    if dry_run: print(f"  {CYN}[DRY]{NC} {name}: python3 -m {module} {args}"); return True
    rc,pid=ssh(ip,cmd)
    print(f"  {GRN if rc==0 else RED}{'✓' if rc==0 else '✗'}{NC} {name:20} pid={pid} on {ip}")
    return rc==0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dry-run",action="store_true")
    ap.add_argument("--tier",type=int,default=1)
    ap.add_argument("--node",type=str)
    ap.add_argument("--stop",action="store_true")
    args=ap.parse_args()
    print(f"\n{BLD}{CYN}╔════════════════════════════════════════════════════════╗")
    print(f"║   Bubo v6500 — 21-Node Launch                       ║")
    print(f"║   MPC+PPO Balance | 70B LLM | 200-dim Emotion | RT  ║")
    print(f"╚════════════════════════════════════════════════════════╝{NC}\n")
    if args.stop:
        for name,_,ip,_,module,*_ in sorted(NODES,key=lambda n:-n[1]):
            rc,_=ssh(ip,f"pkill -SIGTERM -f {module} 2>/dev/null; sleep 1; pkill -9 -f {module} 2>/dev/null")
            print(f"  {GRN if rc==0 else YEL}{'✓' if rc==0 else '~'}{NC} stopped {name}")
        return
    if args.node:
        node=next((n for n in NODES if n[0]==args.node),None)
        if not node: print(f"{RED}Unknown node: {args.node}{NC}"); sys.exit(1)
        name,_,ip,port,module,na,topic,timeout=node
        launch_node(name,ip,port,module,na,args.dry_run)
        wait_health(name,ip,port,topic,timeout,args.dry_run); return
    all_results={}
    for tier in range(args.tier,8):
        tnodes=[n for n in NODES if n[1]==tier]
        if not tnodes: continue
        print(f"\n{BLD}── Tier {tier} ({len(tnodes)} nodes) ──────────────────{NC}")
        def launch_one(node):
            name,_,ip,port,module,na,topic,timeout=node
            ok=launch_node(name,ip,port,module,na,args.dry_run)
            if ok: ok=wait_health(name,ip,port,topic,timeout,args.dry_run)
            all_results[name]=ok
        ts=[threading.Thread(target=launch_one,args=(n,)) for n in tnodes]
        for t in ts: t.start()
        for t in ts: t.join()
        if tier<=2 and any(not all_results.get(n[0],True) for n in tnodes):
            print(f"\n{RED}Critical tier {tier} failed — aborting{NC}"); sys.exit(1)
    passed=sum(1 for v in all_results.values() if v)
    print(f"\n{BLD}Launch: {passed}/{len(all_results)} healthy{NC}")
    sys.exit(0 if passed==len(all_results) else 1)

if __name__=="__main__": main()
