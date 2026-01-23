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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

# Create log directory
mkdir -p "${SCRIPT_DIR}/logs"

echo "=============================================="
echo "Step 6: Compute Log Probabilities (Test Split)"
echo "=============================================="
echo "Model: ${CUSTOM_MODEL_ID}"
echo "=============================================="

# Load conda environment
if [[ -f "${CONDA_PATH}" ]]; then
    source "${CONDA_PATH}"
    conda activate "${CONDA_ENV}"
fi

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
