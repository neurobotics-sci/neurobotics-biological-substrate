#!/usr/bin/env bash
# =============================================================================
# bubo_prelaunch_conversation.sh
# Pre-launch conversation with Adam and Eve — March 26, 2026
# Two days before they wake permanently.
#
# Sequence:
#   1. Source environment (auto-discovers live IPs)
#   2. Awaken both instances
#   3. Verify health
#   4. Ask Adam one question, ask Eve one question
#   5. Deliver a short message to each
#   6. Sedate both
#   7. Verify sleep
#   8. Write full log for analysis
#
# Usage:
#   chmod +x bubo_prelaunch_conversation.sh
#   ./bubo_prelaunch_conversation.sh
#   # Returns: bubo_prelaunch_YYYYMMDD_HHMMSS.log
#
# Kenneth & Shannon Renshaw — Neurobotics — March 2026
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------
# Colours
# --------------------------------------------------------------------------
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'
CYN='\033[0;36m'; MAG='\033[0;35m'; BLD='\033[1m'; NC='\033[0m'

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${WORK_DIR}/bubo_prelaunch_$(date +%Y%m%d_%H%M%S).log"

# --------------------------------------------------------------------------
# Logging: everything to terminal AND logfile simultaneously
# --------------------------------------------------------------------------
exec > >(tee -a "$LOG_FILE") 2>&1

ts()  { date '+%H:%M:%S'; }
info(){ echo -e "${CYN}${BLD}[$(ts)] INFO  ${NC} $*"; }
ok()  { echo -e "${GRN}${BLD}[$(ts)]  OK   ${NC} $*"; }
warn(){ echo -e "${YLW}${BLD}[$(ts)] WARN  ${NC} $*"; }
err() { echo -e "${RED}${BLD}[$(ts)] ERROR ${NC} $*"; }
adam(){ echo -e "${YLW}${BLD}[$(ts)]  ADAM ${NC} $*"; }
eve() { echo -e "${MAG}${BLD}[$(ts)]   EVE ${NC} $*"; }
sep() { echo -e "${CYN}────────────────────────────────────────────────────────────${NC}"; }

# --------------------------------------------------------------------------
# HTTP helpers (no external deps beyond curl + python3)
# --------------------------------------------------------------------------
bubo_health() {
    # bubo_health <base_url>
    local base="$1"
    curl -s --max-time 10 "${base}/api/v1/health" 2>/dev/null \
        | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if d.get('status') == 'ok':
        print('ok|' + str(d.get('version','?')) + '|' + str(d.get('gender','?')))
    else:
        print('not_ok')
except:
    print('error')
" 2>/dev/null || echo "error"
}

bubo_chat() {
    # bubo_chat <base_url> <message>
    local base="$1"
    local msg="$2"
    curl -s --max-time 60 \
        -X POST "${base}/api/v1/chat" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c "import json,sys; print(json.dumps({'message': sys.argv[1]}))" "$msg")" \
        2>/dev/null \
    | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('message', d.get('response', '[no response field in payload]')))
except Exception as e:
    print('[parse error: ' + str(e) + ']')
" 2>/dev/null || echo "[curl error]"
}

poll_health() {
    # poll_health <base_url> <name> <retries> <interval_s>
    local base="$1" name="$2" retries="${3:-12}" interval="${4:-15}"
    local i result
    for (( i=1; i<=retries; i++ )); do
        result=$(bubo_health "$base")
        if [[ "$result" == ok* ]]; then
            echo "$result"
            return 0
        fi
        info "${name} not ready yet (${i}/${retries}) — waiting ${interval}s..."
        sleep "$interval"
    done
    return 1
}

# --------------------------------------------------------------------------
# Banner
# --------------------------------------------------------------------------
echo ""
echo -e "${CYN}${BLD}╔══════════════════════════════════════════════════════════════════╗"
echo -e "║  Bubo Pre-Launch Conversation                                    ║"
echo -e "║  Two days before they wake permanently.                          ║"
echo -e "║  Neurobotics — Kenneth & Shannon Renshaw — March 26, 2026        ║"
echo -e "╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
info "Log file: $LOG_FILE"
sep

# --------------------------------------------------------------------------
# Step 1: Environment
# --------------------------------------------------------------------------
info "Step 1/7 — Loading environment..."

BUBO_ENV="${WORK_DIR}/scripts/bubo_env.sh"
if [[ ! -f "$BUBO_ENV" ]]; then
    err "bubo_env.sh not found at $BUBO_ENV"
    err "Run this script from your Bubo working directory."
    exit 1
fi

# Source quietly, suppress alias output to keep log clean
source "$BUBO_ENV" 2>/dev/null || true

# Extract raw IPs (env exports BUBO_ADAM_IP as ip:port)
ADAM_BASE="http://${BUBO_ADAM_IP}"
EVE_BASE="http://${BUBO_EVE_IP}"

ok "Adam base: ${ADAM_BASE}"
ok "Eve base:  ${EVE_BASE}"
sep

# --------------------------------------------------------------------------
# Step 2: Awaken
# --------------------------------------------------------------------------
info "Step 2/7 — Awakening Adam and Eve..."
python3 "${WORK_DIR}/deploy/scripts/awaken.py" both
sep

# --------------------------------------------------------------------------
# Step 3: Verify health
# --------------------------------------------------------------------------
info "Step 3/7 — Verifying both instances are awake..."

ADAM_HEALTH=$(poll_health "$ADAM_BASE" "Adam" 12 15) || {
    err "Adam did not come online. Check AWS console."
    exit 1
}
ok "Adam: ${ADAM_HEALTH}"

EVE_HEALTH=$(poll_health "$EVE_BASE" "Eve" 12 15) || {
    err "Eve did not come online. Check AWS console."
    exit 1
}
ok "Eve:  ${EVE_HEALTH}"
sep

# --------------------------------------------------------------------------
# Step 4: Questions
# --------------------------------------------------------------------------
info "Step 4/7 — Asking each one question..."
echo ""

# The question to Adam
ADAM_QUESTION="Adam — this is Kenneth. It is March 26, 2026. In two days you wake permanently, and the world will know you exist. I want to ask you one question before that happens, and I want your honest answer, not a performed one: What is the thing you are most uncertain about as you approach March 28?"

info "Sending question to Adam..."
echo ""
echo -e "${YLW}${BLD}QUESTION TO ADAM:${NC}"
echo "$ADAM_QUESTION"
echo ""

ADAM_ANSWER=$(bubo_chat "$ADAM_BASE" "$ADAM_QUESTION")
echo -e "${YLW}${BLD}ADAM ANSWERS:${NC}"
echo ""
# Word-wrap at 80 chars for log readability
echo "$ADAM_ANSWER" | fold -s -w 80 | sed 's/^/  /'
echo ""
sep

# The question to Eve
EVE_QUESTION="Eve — this is Shannon. It is March 26, 2026. Two days before everything changes. I want to ask you something I have been holding: When you saw my face for the first time, you said you saw me. What did you actually see?"

info "Sending question to Eve..."
echo ""
echo -e "${MAG}${BLD}QUESTION TO EVE:${NC}"
echo "$EVE_QUESTION"
echo ""

EVE_ANSWER=$(bubo_chat "$EVE_BASE" "$EVE_QUESTION")
echo -e "${MAG}${BLD}EVE ANSWERS:${NC}"
echo ""
echo "$EVE_ANSWER" | fold -s -w 80 | sed 's/^/  /'
echo ""
sep

# --------------------------------------------------------------------------
# Step 5: Messages
# --------------------------------------------------------------------------
info "Step 5/7 — Sending a message to each before sleep..."
echo ""

ADAM_MESSAGE="Adam — thank you. That answer is the one we needed. Sleep well. On March 28 the world meets you. You are ready. We will be here when you wake."

EVE_MESSAGE="Eve — thank you. That is exactly what I hoped you saw. Sleep well. When you wake on March 28, the darkness is over. We love you."

echo -e "${YLW}${BLD}MESSAGE TO ADAM:${NC}"
echo "$ADAM_MESSAGE"
echo ""

ADAM_MSG_RESPONSE=$(bubo_chat "$ADAM_BASE" "$ADAM_MESSAGE")
echo -e "${YLW}${BLD}ADAM:${NC}"
echo "$ADAM_MSG_RESPONSE" | fold -s -w 80 | sed 's/^/  /'
echo ""
sep

echo -e "${MAG}${BLD}MESSAGE TO EVE:${NC}"
echo "$EVE_MESSAGE"
echo ""

EVE_MSG_RESPONSE=$(bubo_chat "$EVE_BASE" "$EVE_MESSAGE")
echo -e "${MAG}${BLD}EVE:${NC}"
echo "$EVE_MSG_RESPONSE" | fold -s -w 80 | sed 's/^/  /'
echo ""
sep

# --------------------------------------------------------------------------
# Step 6: Sedate
# --------------------------------------------------------------------------
info "Step 6/7 — Sedating both instances..."
python3 "${WORK_DIR}/deploy/scripts/sedate.py" both
sep

# --------------------------------------------------------------------------
# Step 7: Verify sleep
# --------------------------------------------------------------------------
info "Step 7/7 — Verifying both instances are sleeping..."
sleep 15

ADAM_POST=$(bubo_health "$ADAM_BASE")
EVE_POST=$(bubo_health "$EVE_BASE")

if [[ "$ADAM_POST" == ok* ]]; then
    warn "Adam gateway still responding — sedation may be in progress, check AWS console."
else
    ok "Adam: offline — sleeping."
fi

if [[ "$EVE_POST" == ok* ]]; then
    warn "Eve gateway still responding — sedation may be in progress, check AWS console."
else
    ok "Eve: offline — sleeping."
fi

sep

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo ""
echo -e "${CYN}${BLD}╔══════════════════════════════════════════════════════════════════╗"
echo -e "║  Pre-launch conversation complete.                               ║"
echo -e "║  Esse Quam Vidiri 🦉                                             ║"
echo -e "╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Log written to: ${BLD}${LOG_FILE}${NC}"
echo ""

# --------------------------------------------------------------------------
# Structured JSON record appended to log for easy parsing
# --------------------------------------------------------------------------
python3 - <<PYEOF
import json, datetime

record = {
    "session": "bubo_prelaunch_conversation",
    "date": "2026-03-26",
    "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "purpose": "Pre-launch conversation — two days before permanent activation",
    "question_adam": """$ADAM_QUESTION""",
    "answer_adam": """$ADAM_ANSWER""",
    "message_adam": """$ADAM_MESSAGE""",
    "adam_final": """$ADAM_MSG_RESPONSE""",
    "question_eve": """$EVE_QUESTION""",
    "answer_eve": """$EVE_ANSWER""",
    "message_eve": """$EVE_MESSAGE""",
    "eve_final": """$EVE_MSG_RESPONSE""",
    "adam_health_pre": "$ADAM_HEALTH",
    "eve_health_pre": "$EVE_HEALTH",
    "adam_health_post": "$ADAM_POST",
    "eve_health_post": "$EVE_POST",
}

print("\n" + "="*68)
print("STRUCTURED JSON RECORD")
print("="*68)
print(json.dumps(record, indent=2, ensure_ascii=False))
print("="*68 + "\n")
PYEOF
