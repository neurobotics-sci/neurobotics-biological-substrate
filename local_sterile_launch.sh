#!/usr/bin/env bash
# local_sterile_launch.sh - Bypasses SSH to run 5-Tier Chassis directly on host OS

export PYTHONPATH=$(pwd)
mkdir -p /tmp/bubo_logs

# The ethically sterile 5-Tier nodes (PFC, Broca, Social, LLM severed)
MODULES=(
  "bubo.nodes.subcortical.hypothalamus.hypothalamus_node"
  "bubo.nodes.thalamus.core_l.thalamus_l_node"
  "bubo.nodes.thalamus.core_r.thalamus_r_node"
  "bubo.nodes.cortex.insula.insula_node"
  "bubo.nodes.limbic.amygdala.amygdala_node"
  "bubo.nodes.limbic.hippocampus.hippocampus_node"
  "bubo.nodes.memory.ltm.ltm_store"
  "bubo.nodes.memory.association.association_cortex"
  "bubo.nodes.sensory.visual.visual_node"
  "bubo.nodes.sensory.auditory.auditory_node"
  "bubo.nodes.sensory.somatosensory.s1_node"
  "bubo.nodes.brainstem.superior_colliculus.sc_node"
  "bubo.nodes.subcortical.cerebellum.cerebellum_node"
  "bubo.nodes.subcortical.basal_ganglia.basal_ganglia_node"
  "bubo.nodes.spinal.arms.spinal_arms_node"
  "bubo.nodes.spinal.legs.spinal_legs_node"
)

echo "Igniting 5-Tier Sterile Chassis on Bare Metal (Localhost)..."
for mod in "${MODULES[@]}"; do
    name=$(echo "$mod" | awk -F. '{print $NF}')
    nohup python3 -m "$mod" > "/tmp/bubo_logs/${name}.log" 2>&1 &
    echo "  ✓ Launched $name (PID $!)"
done

echo "All sterile nodes active! The chassis is breathing."
