#!/usr/bin/env bash
# bubo_secrets_setup.sh — First-time secrets setup
# Run once: bash scripts/bubo_secrets_setup.sh
echo "🦉 Bubo secrets setup"
echo ""
echo "This will create ~/.bubo_secrets with your API key."
echo "It will NOT be committed to git."
echo ""
read -rp "LLM API key (sk-ant-...): " LLM_KEY
read -rp "AWS Access Key ID: " AWS_KEY
read -rsp "AWS Secret Access Key: " AWS_SECRET
echo ""
read -rp "AWS Default Region [us-east-1]: " AWS_REGION
AWS_REGION="${AWS_REGION:-us-east-1}"

cat > "$HOME/.bubo_secrets" << SECRETS
export BUBO_LLM_API_KEY="${LLM_KEY}"
export AWS_ACCESS_KEY_ID="${AWS_KEY}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET}"
export AWS_DEFAULT_REGION="${AWS_REGION}"
SECRETS

chmod 600 "$HOME/.bubo_secrets"

# Also configure aws cli
aws configure set aws_access_key_id "$AWS_KEY"
aws configure set aws_secret_access_key "$AWS_SECRET"
aws configure set default.region "$AWS_REGION"
aws configure set default.output "json"

echo "✓ ~/.bubo_secrets written (chmod 600)"
echo "✓ AWS CLI configured"
echo ""
echo "Now run: source scripts/bubo_env.sh"
echo "🦉 Esse Quam Vidiri"
