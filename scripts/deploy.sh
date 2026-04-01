#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY="$HOME/.ssh/bubo_id_ed25519"
[[ ! -f "$KEY" ]] && { echo "SSH key missing"; exit 1; }
echo "=== Bubo v50.0 Deploy (22 nodes) ==="
ansible-playbook -i "$REPO/ansible/inventory/hosts.ini" "$REPO/ansible/site.yml" "$@" \
  2>&1 | tee "/tmp/bubo_deploy_$(date +%Y%m%d_%H%M%S).log"
