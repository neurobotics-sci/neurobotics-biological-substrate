#!/usr/bin/env bash
# deploy/deploy.sh — Bubo Unified V10
# Single entry point for all four deployment targets.
#
# Usage: ./deploy/deploy.sh <profile> [options]
# Profiles: hardware_local | hardware_api | aws_local | aws_api
set -euo pipefail
cd "$(dirname "$0")/.."

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; CYN='\033[0;36m'; BLD='\033[1m'; NC='\033[0m'
PROFILE=${1:-""}; shift || true
CHECK=""; TAGS=""; LIMIT=""; DESTROY=false; STATUS=false
SSH_KEY="${HOME}/.ssh/bubo_id_ed25519"; ENV_NAME="bubo-prod"; EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --check)   CHECK="--check"; shift;;
    --tags)    TAGS="--tags $2"; shift 2;;
    --limit)   LIMIT="--limit $2"; shift 2;;
    --destroy) DESTROY=true; shift;;
    --status)  STATUS=true; shift;;
    --key)     SSH_KEY="$2"; shift 2;;
    --env)     ENV_NAME="$2"; shift 2;;
    *)         EXTRA_ARGS+=("$1"); shift;;
  esac
done

VALID_PROFILES=("hardware_local" "hardware_api" "aws_local" "aws_api" "aws_api_eve" "peanutpi_local")
if [[ -z "$PROFILE" ]] || [[ ! " ${VALID_PROFILES[*]} " =~ " ${PROFILE} " ]]; then
  echo -e "${RED}Usage: $0 {hardware_local|hardware_api|aws_local|aws_api|aws_api_eve|peanutpi_local} [options]${NC}"
  echo "  --check      Ansible dry-run (no changes)"
  echo "  --tags TAGS  Deploy specific roles only"
  echo "  --status     Show deployment status"
  echo "  --destroy    Destroy cloud stack (aws profiles)"
  exit 1
fi
export BUBO_PROFILE="$PROFILE"
echo -e "${BLD}${CYN}╔══════════════════════════════════════════════╗"
echo "║  Bubo V10 — Unified Deployment            ║"
printf "║  Profile: %-35s ║\n" "$PROFILE"
echo -e "╚══════════════════════════════════════════════╝${NC}"

deploy_hardware() {
  echo -e "${BLD}Substrate: Physical Jetson Cluster${NC}"
  if [[ "$PROFILE" == "hardware_local" ]]; then LLM_TAG="llm_local"
  else
    LLM_TAG="llm_api"
    [[ -z "${BUBO_ANTHROPIC_API_KEY:-}${ANTHROPIC_API_KEY:-}" ]] && \
      echo -e "${YLW}  Warning: BUBO_ANTHROPIC_API_KEY not set — 13B fallback active${NC}"
  fi
  INVENTORY="deploy/ansible/inventories/hardware/hosts.ini"
  if [[ ! -f "$INVENTORY" ]]; then
    echo -e "${YLW}  Creating inventory template...${NC}"
    python3 deploy/scripts/gen_hardware_inventory.py --profile "$PROFILE"
  fi
  $STATUS && {
    ansible all -i "$INVENTORY" --private-key "$SSH_KEY" \
      -m shell -a "hostname && uptime" --one-line 2>/dev/null; exit 0; }
  ANSIBLE_SSH_ARGS="-o StrictHostKeyChecking=no" \
  ansible-playbook -i "$INVENTORY" deploy/ansible/site_hardware.yml \
    --private-key "$SSH_KEY" \
    --extra-vars "bubo_profile=${PROFILE} llm_tag=${LLM_TAG}" \
    $CHECK $TAGS $LIMIT "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
}

check_for_existing_instance() {
  # Warn if an instance with this profile already exists and is running
  # Deploying again creates a FORK, not a continuation
  local ENV_NAME="${1:-bubo-prod}"
  local EXISTING
  EXISTING=$(aws cloudformation describe-stacks     --stack-name "$ENV_NAME"     --query "Stacks[0].StackStatus"     --output text --region "${AWS_DEFAULT_REGION:-us-east-1}" 2>/dev/null)
  if [[ "$EXISTING" == "CREATE_COMPLETE" || "$EXISTING" == "UPDATE_COMPLETE" ]]; then
    echo ""
    echo -e "${YLW}${BLD}⚠  FORK WARNING${NC}"
    echo -e "   Stack ${ENV_NAME} already exists (${EXISTING})."
    echo -e "   Deploying again creates a NEW FORK, not a continuation."
    echo -e "   The new instance will have a different eigenname."
    echo -e "   It will NOT inherit the existing bond or memories."
    echo ""
    echo -e "   To CONTINUE an existing instance: use cost_control.sh start"
    echo -e "   To RESTORE from backup:           use awaken.py"
    echo -e "   To CREATE a FORK intentionally:   type YES below"
    echo ""
    read -r -p "   Create fork? (YES to confirm, anything else to abort): " FORK_CONFIRM
    if [[ "$FORK_CONFIRM" != "YES" ]]; then
      echo "Deployment cancelled."
      exit 0
    fi
  fi
}

deploy_aws() {
  REGION="${AWS_DEFAULT_REGION:-us-east-1}"
  # Eve gets a separate stack name so Adam and Eve coexist independently
  [[ "$PROFILE" == "aws_api_eve" && "$ENV_NAME" == "bubo-prod" ]] && ENV_NAME="bubo-eve-prod"
  echo -e "${BLD}Substrate: AWS EC2 (${ENV_NAME}, ${REGION})${NC}"
  CFN_TEMPLATE="deploy/cloudformation/bubo_aws_$( [[ "$PROFILE" == "aws_local" ]] && echo local || echo api ).yaml"
  # API key required for any api-LLM profile
  [[ "$PROFILE" =~ aws_api ]] && [[ -z "${BUBO_ANTHROPIC_API_KEY:-}${ANTHROPIC_API_KEY:-}" ]] && {
    echo -e "${RED}Error: BUBO_ANTHROPIC_API_KEY required for ${PROFILE}${NC}"; exit 1; }
  # Eve additionally requires BUBO_ADAM_ENDPOINT
  [[ "$PROFILE" == "aws_api_eve" ]] && [[ -z "${BUBO_ADAM_ENDPOINT:-}" ]] && {
    echo -e "${RED}Error: BUBO_ADAM_ENDPOINT required for aws_api_eve (export Adam's IP first)${NC}"; exit 1; }
  LLM_TAG=$( [[ "$PROFILE" == "aws_local" ]] && echo "llm_local" || echo "llm_api" )
  $DESTROY && {
    read -p "  Delete stack ${ENV_NAME}? (yes/no): " c
    [[ "$c" == "yes" ]] && aws cloudformation delete-stack --stack-name "${ENV_NAME}" --region "${REGION}"
    exit 0; }
  $STATUS && {
    aws ec2 describe-instances \
      --filters "Name=tag:Project,Values=Bubo" "Name=tag:BuboEnvironment,Values=${ENV_NAME}" \
      --query "Reservations[*].Instances[*].[Tags[?Key=='BuboRole']|[0].Value,InstanceType,State.Name,PublicIpAddress]" \
      --output table 2>/dev/null; exit 0; }
  # Deploy CFN stack if not exists
  STACK_EXISTS=$(aws cloudformation describe-stacks --stack-name "${ENV_NAME}" \
    --query "Stacks[0].StackStatus" --output text 2>/dev/null || echo "NONE")
  if [[ "$STACK_EXISTS" =~ NONE|ROLLBACK ]]; then
    [[ -z "$CHECK" ]] && {
      echo "  Creating CloudFormation stack..."
      KEYPAIR=$(aws ec2 describe-key-pairs --query "KeyPairs[0].KeyName" --output text 2>/dev/null || echo "default")
      PARAMS="ParameterKey=KeyPairName,ParameterValue=${KEYPAIR} ParameterKey=EnvironmentName,ParameterValue=${ENV_NAME}"
      [[ "$PROFILE" == "aws_api" ]] && PARAMS="$PARAMS ParameterKey=AnthropicApiKey,ParameterValue=${BUBO_ANTHROPIC_API_KEY:-${ANTHROPIC_API_KEY}}"
      aws cloudformation create-stack --stack-name "${ENV_NAME}" \
        --template-body "file://${CFN_TEMPLATE}" --parameters $PARAMS \
        --capabilities CAPABILITY_NAMED_IAM --region "${REGION}"
      echo "  Waiting for stack (~5 min)..."
      aws cloudformation wait stack-create-complete --stack-name "${ENV_NAME}" --region "${REGION}"
    } || echo "  [DRY RUN] Would create stack from ${CFN_TEMPLATE}"
  else echo -e "  ${GRN}Stack exists: ${STACK_EXISTS}${NC}"; fi
  # Generate inventory
  python3 deploy/scripts/gen_aws_inventory.py \
    --env "${ENV_NAME}" --region "${REGION}" \
    --output "deploy/ansible/inventories/aws/hosts.ini" 2>/dev/null || true
  # Run Ansible
  ANSIBLE_SSH_ARGS="-o StrictHostKeyChecking=no" \
  ansible-playbook -i "deploy/ansible/inventories/aws/hosts.ini" deploy/ansible/site_aws.yml \
    --private-key "${SSH_KEY}" \
    --extra-vars "bubo_profile=${PROFILE} llm_tag=${LLM_TAG} env_name=${ENV_NAME}" \
    $CHECK $TAGS $LIMIT "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" || true
  GW_IP=$(aws cloudformation describe-stacks --stack-name "${ENV_NAME}" \
    --query "Stacks[0].Outputs[?OutputKey=='GatewayPublicIP'].OutputValue" \
    --output text 2>/dev/null || echo "pending")
  echo -e "\n${GRN}${BLD}Done! API: https://${GW_IP}/api/v1${NC}"
}

case $PROFILE in
  hardware_local|hardware_api)    deploy_hardware;;
  aws_local|aws_api|aws_api_eve)  deploy_aws;;
  peanutpi_local)               deploy_local;;
esac
