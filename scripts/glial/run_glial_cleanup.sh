#!/usr/bin/env bash
# scripts/glial/run_glial_cleanup.sh — Bubo v5550
# Called by udev rule when charger is connected (wall power detected)
# Also scheduled by cron for overnight maintenance

set -euo pipefail
LOG="/var/log/bubo/glial_cleanup_$(date +%Y%m%d_%H%M%S).log"
PYTHONPATH=/opt/bubo

echo "$(date): Glial cleanup triggered" | tee -a "$LOG"

# Detect charger (or allow --force flag)
FORCE=${1:-""}
if [[ "$FORCE" != "--force" ]]; then
  CHARGER_STATUS=$(cat /sys/class/power_supply/BAT0/status 2>/dev/null || echo "Unknown")
  if [[ "$CHARGER_STATUS" != "Charging" && "$CHARGER_STATUS" != "Full" ]]; then
    echo "Charger not detected (status=$CHARGER_STATUS). Use --force to override." | tee -a "$LOG"
    exit 0
  fi
fi

echo "Running glial cleanup..." | tee -a "$LOG"
cd /opt/bubo
python3 -m bubo.glial.glial_cleanup 2>&1 | tee -a "$LOG"
echo "$(date): Glial cleanup complete" | tee -a "$LOG"
