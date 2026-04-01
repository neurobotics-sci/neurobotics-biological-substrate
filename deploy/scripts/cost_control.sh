#!/usr/bin/env bash
# deploy/scripts/cost_control.sh — Bubo Adam & Eve
# Start, stop, restart, or teardown a Bubo CloudFormation stack.
#
# Usage:
#   ./deploy/scripts/cost_control.sh start   bubo-prod
#   ./deploy/scripts/cost_control.sh stop    bubo-prod
#   ./deploy/scripts/cost_control.sh restart bubo-prod
#   ./deploy/scripts/cost_control.sh status  bubo-prod
#   ./deploy/scripts/cost_control.sh ip      bubo-prod
#   ./deploy/scripts/cost_control.sh teardown bubo-prod
#
# Stopped instances cost $0/hr (EBS storage only, ~$0.02/day).
# Restarting a stopped stack takes ~2-3 minutes.
# Tearing down requires a full redeploy (~35-45 min) to restart.
#
# Bubo Adam stack name:  bubo-prod
# Bubo Eve  stack name:  bubo-eve-prod

set -euo pipefail

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'
CYN='\033[0;36m'; BLD='\033[1m'; NC='\033[0m'

ACTION="${1:-}"
ENV_NAME="${2:-bubo-prod}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

usage() {
  echo -e "${BLD}Usage:${NC} $0 {start|stop|restart|status|ip|teardown} [env-name]"
  echo ""
  echo "  start    ENV   — Start all stopped EC2 instances in stack"
  echo "  stop     ENV   — Stop all running EC2 instances (saves cost)"
  echo "  restart  ENV   — Stop then start"
  echo "  status   ENV   — Show instance states and IPs"
  echo "  ip       ENV   — Print gateway public IP only"
  echo "  teardown ENV   — Delete CloudFormation stack (full redeploy needed)"
  echo ""
  echo "  ENV defaults to 'bubo-prod' (Adam). Use 'bubo-eve-prod' for Eve."
  exit 1
}

[[ -z "$ACTION" ]] && usage

get_instances() {
  aws ec2 describe-instances \
    --filters \
      "Name=tag:Project,Values=Bubo" \
      "Name=tag:BuboEnvironment,Values=${ENV_NAME}" \
    --query "Reservations[*].Instances[*].{id:InstanceId,role:Tags[?Key=='BuboRole']|[0].Value,state:State.Name,ip:PublicIpAddress}" \
    --output json \
    --region "${REGION}" 2>/dev/null \
  | python3 -c "
import json,sys
data=json.load(sys.stdin)
flat=[i for g in data for i in g]
for i in flat: print(i['id'],i.get('role','?'),i.get('state','?'),i.get('ip') or '-')
"
}

get_instance_ids() {
  local state_filter="${1:-}"
  aws ec2 describe-instances \
    --filters \
      "Name=tag:Project,Values=Bubo" \
      "Name=tag:BuboEnvironment,Values=${ENV_NAME}" \
      ${state_filter:+"Name=instance-state-name,Values=${state_filter}"} \
    --query "Reservations[*].Instances[*].InstanceId" \
    --output text \
    --region "${REGION}" 2>/dev/null
}

get_gateway_ip() {
  aws cloudformation describe-stacks \
    --stack-name "${ENV_NAME}" \
    --query "Stacks[0].Outputs[?OutputKey=='GatewayPublicIP'].OutputValue" \
    --output text \
    --region "${REGION}" 2>/dev/null
}

do_start() {
  echo -e "${CYN}Starting instances in ${ENV_NAME}...${NC}"
  IDS=$(get_instance_ids "stopped")
  if [[ -z "$IDS" ]]; then
    echo -e "${YLW}  No stopped instances found (already running?). Checking status...${NC}"
    do_status; return
  fi
  aws ec2 start-instances --instance-ids $IDS --region "${REGION}" > /dev/null
  echo "  Waiting for instances to reach running state (~2-3 min)..."
  aws ec2 wait instance-running --instance-ids $IDS --region "${REGION}"
  echo -e "${GRN}  Done. Instances started.${NC}"
  sleep 5   # give gateway a moment to bind
  do_status
  echo ""
  IP=$(get_gateway_ip)
  if [[ -n "$IP" && "$IP" != "None" ]]; then
    echo -e "${BLD}Gateway IP: ${GRN}${IP}${NC}"
    echo -e "  Health: ${CYN}curl -k https://${IP}/api/v1/health${NC}"
    echo ""
    echo "  Don't forget to update your environment:"
    if [[ "$ENV_NAME" == *"eve"* ]]; then
        echo "    export BUBO_EVE_IP='${IP}'"
    else
        echo "    export BUBO_ADAM_IP='${IP}'"
    fi
    (( 0 )) && [[ "$ENV_NAME" == *"eve"* ]] && echo "    export BUBO_EVE_IP='${IP}'"
  fi
}

do_stop() {
  echo -e "${CYN}Stopping instances in ${ENV_NAME}...${NC}"
  IDS=$(get_instance_ids "running")
  if [[ -z "$IDS" ]]; then
    echo -e "${YLW}  No running instances found (already stopped?).${NC}"
    return
  fi
  aws ec2 stop-instances --instance-ids $IDS --region "${REGION}" > /dev/null
  echo "  Waiting for instances to stop..."
  aws ec2 wait instance-stopped --instance-ids $IDS --region "${REGION}"
  echo -e "${GRN}  Done. Instances stopped. Cost: ~\$0.02/day (EBS only).${NC}"
}

do_status() {
  echo -e "${BLD}Stack: ${ENV_NAME} | Region: ${REGION}${NC}"
  printf "  %-22s %-12s %-12s %-18s\n" "Instance ID" "Role" "State" "Public IP"
  printf "  %-22s %-12s %-12s %-18s\n" "----------" "----" "-----" "---------"
  get_instances | while read id role state ip; do
    case "$state" in
      running) COLOR="${GRN}" ;;
      stopped) COLOR="${YLW}" ;;
      *)       COLOR="${RED}" ;;
    esac
    printf "  %-22s %-12s ${COLOR}%-12s${NC} %-18s\n" "$id" "$role" "$state" "$ip"
  done
}

do_ip() {
  IP=$(get_gateway_ip)
  if [[ -n "$IP" && "$IP" != "None" ]]; then
    echo "$IP"
  else
    # Fall back to live EC2 query
    get_instances | grep gateway | awk '{print $4}'
  fi
}

do_teardown() {
  echo -e "${RED}${BLD}WARNING: This will DELETE the CloudFormation stack '${ENV_NAME}'.${NC}"
  echo -e "${RED}         A full redeploy (~35-45 min) will be required to restart.${NC}"
  echo -e "${RED}         Persistent LTM data on EFS will also be deleted.${NC}"
  echo ""
  read -p "  Type 'yes' to confirm teardown of ${ENV_NAME}: " confirm
  [[ "$confirm" != "yes" ]] && { echo "Aborted."; exit 0; }
  echo "  Deleting stack ${ENV_NAME}..."
  aws cloudformation delete-stack --stack-name "${ENV_NAME}" --region "${REGION}"
  echo "  Waiting for deletion to complete..."
  aws cloudformation wait stack-delete-complete --stack-name "${ENV_NAME}" --region "${REGION}"
  echo -e "${GRN}  Stack deleted.${NC}"
}

case "$ACTION" in
  start)    do_start    ;;
  stop)     do_stop     ;;
  restart)  do_stop && sleep 3 && do_start ;;
  status)   do_status   ;;
  ip)       do_ip       ;;
  teardown) do_teardown ;;
  *)        usage       ;;
esac
