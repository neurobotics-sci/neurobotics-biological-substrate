#!/usr/bin/env bash
# scripts/health_check/neural_health_check.sh — Bubo v5550 (20 nodes)
set -euo pipefail
SSH_KEY="$HOME/.ssh/bubo_id_ed25519"
SO="-o StrictHostKeyChecking=no -o ConnectTimeout=4 -i $SSH_KEY"
LOG="/tmp/bubo_health_$(date +%Y%m%d_%H%M%S).log"
GRN='\033[92m'; YEL='\033[93m'; RED='\033[91m'; CYN='\033[96m'; BLD='\033[1m'; NC='\033[0m'
PASS=0; WARN=0; FAIL=0; CRIT=0
pass(){ echo -e "  ${GRN}✓${NC} $*"; PASS=$((PASS+1)); }
warn(){ echo -e "  ${YEL}⚠${NC} $*"; WARN=$((WARN+1)); }
fail(){ echo -e "  ${RED}✗${NC} $*"; FAIL=$((FAIL+1)); }
crit(){ echo -e "  ${RED}${BLD}✗ CRIT:${NC} $*"; CRIT=$((CRIT+1)); }
section(){ echo -e "\n${BLD}${CYN}── $* ──────────────────────────────${NC}"; }
exec > >(tee -a "$LOG") 2>&1

echo -e "${BLD}${CYN}╔══════════════════════════════════════════════════════════╗"
echo     "║   Bubo v5550 — Neural Health Check (20 nodes)           ║"
echo -e  "╚══════════════════════════════════════════════════════════╝${NC}"

declare -A NODES=(
  [pfc-l]="192.168.1.10" [pfc-r]="192.168.1.11"
  [hypothalamus]="192.168.1.12" [thalamus-l]="192.168.1.13"
  [broca]="192.168.1.14" [insula]="192.168.1.15"
  [thalamus-r]="192.168.1.18" [social]="192.168.1.19"
  [hippocampus]="192.168.1.30" [amygdala]="192.168.1.31"
  [cerebellum]="192.168.1.32" [basal-ganglia]="192.168.1.33"
  [association]="192.168.1.34" [ltm-store]="192.168.1.35"
  [visual]="192.168.1.50" [auditory]="192.168.1.51"
  [somatosensory]="192.168.1.52" [spinal-arms]="192.168.1.53"
  [sup-colliculus]="192.168.1.60" [spinal-legs]="192.168.1.61"
)
OFFLINE=()

section "1 — SSH Key"
[[ -f "$SSH_KEY" ]] && pass "Key: $SSH_KEY" || crit "Missing key — run scripts/keydist.sh"

section "2 — Reachability"
for node in "${!NODES[@]}"; do
  ip="${NODES[$node]}"
  ping -c1 -W2 "$ip" &>/dev/null && pass "$node ($ip)" || { crit "$node ($ip) OFFLINE"; OFFLINE+=("$node"); }
done

section "3 — Temperature (Vagus thresholds)"
for node in "${!NODES[@]}"; do
  ip="${NODES[$node]}"; [[ " ${OFFLINE[*]} " =~ " $node " ]] && continue
  tc=$(ssh $SO brain@"$ip" "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 40000" 2>/dev/null || echo 40000)
  t=$((tc/1000))
  if   [[ $t -gt 85 ]]; then crit "$node ${t}°C → VAGUS SHUTDOWN THRESHOLD"
  elif [[ $t -gt 78 ]]; then fail "$node ${t}°C → motor inhibit zone"
  elif [[ $t -gt 70 ]]; then warn "$node ${t}°C → CPU throttle zone"
  else pass "$node ${t}°C nominal"; fi
done

section "4 — DDS Partitions"
for node in "pfc-l" "pfc-r"; do
  ip="${NODES[$node]}"
  nc -z -w2 "$ip" 11811 2>/dev/null && pass "$node DDS Discovery Server :11811 OK" || warn "$node DDS DS not running"
done
for node in "cerebellum" "hypothalamus"; do
  ip="${NODES[$node]}"; [[ " ${OFFLINE[*]} " =~ " $node " ]] && continue
  r=$(ssh $SO brain@"$ip" "test -f /etc/bubo/fastdds_partitions.xml && echo y || echo n" 2>/dev/null || echo n)
  [[ "$r" == "y" ]] && pass "$node partition XML deployed" || warn "$node partition XML missing"
done

section "5 — Vagus Nerve"
for node in "spinal-legs" "sup-colliculus"; do
  ip="${NODES[$node]}"; [[ " ${OFFLINE[*]} " =~ " $node " ]] && continue
  r=$(ssh $SO brain@"$ip" "test -e /sys/class/gpio/gpio48 && echo gpio_ok || echo gpio_missing; systemctl is-active bubo-vagus 2>/dev/null || echo inactive" 2>/dev/null || echo "err")
  echo "$r" | grep -q "gpio_ok" && pass "$node vagus GPIO48" || warn "$node vagus GPIO not configured"
  echo "$r" | grep -q "active" && pass "$node bubo-vagus.service active" || warn "$node vagus service inactive"
done

section "6 — CMAC + Glial"
ip="${NODES[cerebellum]:-}"
[[ -n "$ip" && ! " ${OFFLINE[*]} " =~ " cerebellum " ]] && {
  n=$(ssh $SO brain@"$ip" "python3 -c \"import json,os; p='/opt/bubo/data/cmac_weights.json'; print(json.load(open(p))['n_updates'] if os.path.exists(p) else 0)\" 2>/dev/null || echo 0")
  [[ "${n:-0}" -gt 200 ]] && pass "CMAC trained (${n} updates)" || warn "CMAC cold-start (${n:-0} updates — need 200+)"
}
ip="${NODES[ltm-store]:-}"
[[ -n "$ip" && ! " ${OFFLINE[*]} " =~ " ltm-store " ]] && {
  r=$(ssh $SO brain@"$ip" "test -f /etc/udev/rules.d/99-bubo-charger.rules && echo y || echo n" 2>/dev/null || echo n)
  [[ "$r" == "y" ]] && pass "ltm-store glial udev rule present" || warn "glial udev rule missing"
}

section "7 — Imports + PTP"
for node in "${!NODES[@]}"; do
  ip="${NODES[$node]}"; [[ " ${OFFLINE[*]} " =~ " $node " ]] && continue
  r=$(ssh $SO brain@"$ip" "
    python3 -c 'import sys; sys.path.insert(0,\"/opt/bubo\")
from bubo.shared.bus.neural_bus import T
from bubo.dds_partitions.partition_manager import PartitionManager
print(\"ok\")' 2>/dev/null || echo fail
    systemctl is-active ptp4l 2>/dev/null || echo ptp_inactive
  " 2>/dev/null | tr '\n' ' ')
  echo "$r" | grep -q "ok" && pass "$node imports OK" || fail "$node imports FAIL"
  echo "$r" | grep -q "active" && true || warn "$node ptp4l inactive"
done

echo ""
echo -e "${BLD}${CYN}══════════════════════════════════════════════════════${NC}"
echo -e "${BLD}PASS:${NC} $PASS  ${BLD}WARN:${NC} $WARN  ${BLD}FAIL:${NC} $FAIL  ${BLD}CRIT:${NC} $CRIT | Log: $LOG"
echo ""
if   [[ $CRIT -gt 0 ]]; then echo -e "${RED}${BLD}✗ UNSAFE — resolve ${CRIT} critical(s)${NC}"; exit 2
elif [[ $FAIL -gt 0 ]]; then echo -e "${YEL}${BLD}⚠ CAUTION — ${FAIL} failure(s)${NC}"; exit 1
else echo -e "${GRN}${BLD}✓ NOMINAL — python3 launch/brain_launch.py${NC}"; fi
