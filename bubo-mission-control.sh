#!/bin/bash

LOG_DIR="/tmp/bubo_logs"

# Define the nodes we want to observe
# Format: "Window Title:LogFileName"
NODES=(
    "THALAMUS_GATING:thalamus_l_node.log"
    "SOMATOSENSORY_S1:s1_node.log"
    "CEREBELLUM_CMAC:cerebellum_node.log"
    "BASAL_GANGLIA:basal_ganglia_node.log"
    "SPINAL_AFFERENT:spinal_arms_node.log"
)

echo "🛰️  Deploying Bubo Mission Control..."

for i in "${!NODES[@]}"; do
    IFS=":" read -r TITLE LOG <<< "${NODES[$i]}"
    LOG_PATH="$LOG_DIR/$LOG"

    # Create the log file if it doesn't exist yet so tail doesn't crash
    touch "$LOG_PATH"

    # Launch xterm
    # -T: Window Title
    # -geometry: WxH+X+Y (This offsets them slightly so they don't stack perfectly)
    xterm -T "$TITLE" -geometry 100x25+$((i*50))+$((i*50)) -e "tail -f $LOG_PATH" &
done

echo "✅ All sectors online. Close xterm windows manually to dismiss."
