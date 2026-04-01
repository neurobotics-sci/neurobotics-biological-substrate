#!/usr/bin/env bash
KEY="$HOME/.ssh/bubo_id_ed25519"
IPS=(192.168.1.{10,11,12,13,14,15,16,17,18,19,30,31,32,33,34,35,50,51,52,53,60,61})
[[ ! -f "$KEY" ]] && ssh-keygen -t ed25519 -f "$KEY" -N "" -C "bubo_v50"
for ip in "${IPS[@]}"; do
  echo -n "  brain@$ip ... "
  ssh-copy-id -i "${KEY}.pub" -o StrictHostKeyChecking=no brain@"$ip" 2>/dev/null && echo OK || echo FAIL
done
