#!/usr/bin/env bash
# deploy/scripts/bootstrap.sh — Bubo Unified V10
# Universal EC2 bootstrap: sets node identity and triggers Ansible.
set -euo pipefail

BUBO_NODE=${BUBO_NODE:-unknown}
BUBO_ENV=${BUBO_ENV:-prod}
BUBO_PROFILE=${BUBO_PROFILE:-aws_api}
BUBO_GPU=${BUBO_GPU:-false}
BUBO_LLM_BACKEND=${BUBO_LLM_BACKEND:-anthropic}

echo "[Bubo Bootstrap] node=$BUBO_NODE profile=$BUBO_PROFILE llm=$BUBO_LLM_BACKEND"

apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git curl wget \
    amazon-ssm-agent amazon-cloudwatch-agent

systemctl enable --now amazon-ssm-agent 2>/dev/null || true

mkdir -p /etc/bubo /var/log/bubo /opt/bubo
cat > /etc/bubo/identity << EOF
BUBO_ROLE=$BUBO_NODE
BUBO_PROFILE=$BUBO_PROFILE
BUBO_VERSION=9000
BUBO_LLM_BACKEND=$BUBO_LLM_BACKEND
BUBO_GPU=$BUBO_GPU
EOF

cat > /etc/bubo/env << EOF
BUBO_PROFILE=$BUBO_PROFILE
BUBO_LLM_BACKEND=$BUBO_LLM_BACKEND
BUBO_VERSION=9000
EOF

# Install Ansible
pip3 install ansible boto3 --break-system-packages -q

# Clone repo and run bootstrap
git clone --depth=1 https://github.com/bubo-brain/bubo.git /opt/bubo/src 2>/dev/null || \
  (cd /opt/bubo/src && git pull --ff-only 2>/dev/null || true)

cd /opt/bubo/src
ansible-playbook -i "localhost," -c local deploy/ansible/site_aws.yml \
  --tags bootstrap \
  -e "bubo_profile=$BUBO_PROFILE bubo_role=$BUBO_NODE llm_tag=${BUBO_LLM_BACKEND/anthropic/llm_api}" \
  >> /var/log/bubo/bootstrap.log 2>&1 || true

# Signal CloudFormation
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' http://169.254.169.254/latest/api/token)" http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || echo "us-east-1")
aws cloudformation signal-resource \
  --stack-name "bubo-${BUBO_ENV}" \
  --logical-resource-id "$(echo ${BUBO_NODE} | sed 's/.*/\u&/')Instance" \
  --unique-id "$INSTANCE_ID" --status SUCCESS \
  --region "$REGION" 2>/dev/null || true

echo "[Bubo Bootstrap] Complete: $BUBO_NODE ($BUBO_PROFILE)"
