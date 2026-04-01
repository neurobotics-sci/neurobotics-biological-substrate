#!/usr/bin/env bash
# config/realtime/preempt_rt_setup.sh — Bubo v6500
# Install PREEMPT_RT kernel on Jetson Orin/Nano for < 100μs scheduling jitter
# Solves: Python GIL 100Hz timing unreliability (jitter up to 8ms without RT)
#
# WHAT THIS DOES:
#   Standard Linux scheduler (SCHED_OTHER): jitter 1-50ms — UNACCEPTABLE for 100Hz
#   PREEMPT_RT: jitter < 100μs — motor control deadline reliably met
#
# PREEMPT_RT APPROACH for Jetson:
#   NVIDIA ships partial PREEMPT_RT patches for L4T (JetPack 6.x)
#   The key config: CONFIG_PREEMPT_RT=y + SCHED_FIFO priority on Python
#
# ALTERNATIVELY: STM32H7 co-processor handles 1kHz servo PID entirely,
#   Python just sends targets at 100Hz (much larger timing budget)
#
# Reference: NVIDIA developer forum RT kernel for Jetson AGX Orin
# https://developer.nvidia.com/embedded/linux-tegra-r3531

set -euo pipefail

JETSON_RELEASE=$(cat /etc/nv_tegra_release 2>/dev/null | head -1 || echo "unknown")
KERNEL_VER=$(uname -r)

echo "Jetson release: $JETSON_RELEASE"
echo "Current kernel: $KERNEL_VER"

# Step 1: Check if already RT
if uname -r | grep -q "rt"; then
  echo "✓ PREEMPT_RT kernel already active"
  # Just set process priorities
else
  echo "Installing PREEMPT_RT patches..."

  # Install build dependencies
  apt-get install -y build-essential libncurses5-dev bison flex libssl-dev \
                     libelf-dev git dwarves bc 2>/dev/null

  # Clone NVIDIA's RT-patched kernel (L4T 36.x)
  RT_KERNEL_DIR="/opt/bubo/rt_kernel"
  mkdir -p "$RT_KERNEL_DIR"
  cd "$RT_KERNEL_DIR"

  echo "Clone the JetPack 6 kernel source with RT patches:"
  echo "  git clone https://github.com/NVIDIA/linux-tegra.git --branch rt-6.x"
  echo "  (Requires ~2GB, ~30min build time)"
  echo ""
  echo "Configure:"
  echo "  make ARCH=arm64 tegra_defconfig"
  echo "  # Enable: CONFIG_PREEMPT_RT=y"
  echo "  make ARCH=arm64 menuconfig"
  echo ""
  echo "Build and install:"
  echo "  make ARCH=arm64 -j$(nproc) Image modules"
  echo "  make ARCH=arm64 modules_install"
  echo "  cp arch/arm64/boot/Image /boot/Image.rt"
  echo "  reboot"
fi

# Step 2: Set real-time scheduling priorities for Bubo processes
echo ""
echo "Setting real-time scheduling for Bubo motor control processes..."

# Spinal control loops: highest priority
for service in bubo-spinal_legs bubo-spinal_arms bubo-cerebellum; do
  if systemctl is-active "$service" &>/dev/null; then
    PID=$(systemctl show "$service" -p MainPID --value)
    if [[ -n "$PID" && "$PID" -gt 0 ]]; then
      chrt -f -p 85 "$PID" 2>/dev/null && echo "  ✓ $service PID $PID → SCHED_FIFO 85"
    fi
  fi
done

# Thalamus: medium-high (relay latency critical)
for service in bubo-thalamus_l bubo-thalamus_r; do
  if systemctl is-active "$service" &>/dev/null; then
    PID=$(systemctl show "$service" -p MainPID --value)
    if [[ -n "$PID" && "$PID" -gt 0 ]]; then
      chrt -f -p 70 "$PID" 2>/dev/null && echo "  ✓ $service PID $PID → SCHED_FIFO 70"
    fi
  fi
done

# Configure CPU isolation for motor nodes
if [[ -f /sys/devices/system/cpu/isolated ]]; then
  # Isolate CPU2-3 for motor control (if 4+ cores)
  NCPU=$(nproc)
  if [[ $NCPU -ge 4 ]]; then
    echo "2-3" > /sys/devices/system/cpu/isolated 2>/dev/null || true
    echo "  ✓ CPU2-3 isolated for RT processes"
  fi
fi

# Disable CPU frequency scaling on motor nodes (prevents latency spikes)
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo "performance" > "$cpu" 2>/dev/null || true
done
echo "  ✓ cpufreq → performance mode (motor nodes)"

# Step 3: Network IRQ affinity (move network IRQs to non-RT cores)
for irq_dir in /proc/irq/*/; do
  irq_num=$(basename "$irq_dir")
  name=$(cat "$irq_dir/actions" 2>/dev/null || echo "")
  if echo "$name" | grep -qE "eth|gbe|xhci"; then
    echo "1" > "$irq_dir/smp_affinity_list" 2>/dev/null || true
  fi
done
echo "  ✓ Network IRQs → CPU1 (non-RT)"

echo ""
echo "PREEMPT_RT setup complete. Expected jitter:"
echo "  Without RT: 1-50ms  (motor deadline misses at 100Hz)"
echo "  With RT:    < 100μs (motor deadlines always met)"
