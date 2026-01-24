#!/bin/bash
# =============================================================================
# Step 6: Compute Log Probabilities (SLURM)
# =============================================================================
# This script computes oracle log probabilities for conversations.
# Requires GPU access via SLURM.
#
# Usage: sbatch 06-compute-log-probs.slurm.sh
# =============================================================================

#SBATCH --job-name=stargate-logprobs
#SBATCH --output=logs/06-logprobs-%j.out
#SBATCH --error=logs/06-logprobs-%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

set -euo pipefail

# Clear any pre-existing environment variables that might interfere
# (These will be properly set by config.env below)
unset DATA_ROOT PROMPT_PATH PERSONAS_PATH GOLD_PATH SIMULATION_PATH LOGPROBS_PATH WINRATE_PATH VERSION 2>/dev/null || true

# Detect PROJECT_ROOT automatically
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    # SLURM job: user should submit from project root
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    # Local run: derive from script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
export PROJECT_ROOT

source "${PROJECT_ROOT}/experiments/star-gate/eval-scripts/config.env"

# Create log directory
mkdir -p "${PROJECT_ROOT}/experiments/star-gate/eval-scripts/logs"

echo "=============================================="
echo "Step 6: Compute Log Probabilities (Test Split)"
echo "=============================================="
echo "Model: ${CUSTOM_MODEL_ID}"
echo "=============================================="

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

# Create output directories
mkdir -p "${LOGPROBS_PATH}/${VERSION}/qa-experimental/${CUSTOM_MODEL_NAME}"

# Navigate to script directory
cd "${PROJECT_ROOT}/experiments/star-gate/log-probs"

echo "Computing log probabilities..."
python likelihood-qa-experimental.py \
    model=${CUSTOM_MODEL_NAME} \
    qa_model=${CUSTOM_MODEL_NAME} \
    condition=qa-experimental \
    split=test

echo ""
echo "Done! Log probabilities saved to: ${LOGPROBS_PATH}/${VERSION}/qa-experimental/${CUSTOM_MODEL_NAME}/"
echo "Next step: ./07-filter-log-probs.sh"
