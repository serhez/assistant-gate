#!/bin/bash
# =============================================================================
# Step 8: Generate Final Responses for Win Rate Evaluation (SLURM)
# =============================================================================
# This script generates final responses from your model for pairwise evaluation.
# Requires GPU access via SLURM.
#
# Usage: sbatch 08-generate-responses.slurm.sh
# =============================================================================

#SBATCH --job-name=stargate-responses
#SBATCH --output=logs/08-responses-%j.out
#SBATCH --error=logs/08-responses-%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

# Create log directory
mkdir -p "${SCRIPT_DIR}/logs"

echo "=============================================="
echo "Step 8: Generate Responses (Test Split)"
echo "=============================================="
echo "Model: ${CUSTOM_MODEL_ID}"
echo "=============================================="

# Load conda environment
if [[ -f "${CONDA_PATH}" ]]; then
    source "${CONDA_PATH}"
    conda activate "${CONDA_ENV}"
fi

# Create output directories
mkdir -p "${WINRATE_PATH}/${VERSION}/${CUSTOM_MODEL_NAME}"

# Navigate to script directory
cd "${PROJECT_ROOT}/experiments/star-gate/response-win-rates-randomized-zero-shot"

echo "Generating responses from custom model..."
python get-responses.py \
    answer_model=${CUSTOM_MODEL_NAME} \
    qa_model=${CUSTOM_MODEL_NAME} \
    split=test

# Also generate baseline responses for comparison
echo ""
echo "Generating baseline responses..."
python get-responses.py \
    answer_model=baseline \
    qa_model=baseline \
    split=test

echo ""
echo "Done! Responses saved to: ${WINRATE_PATH}/${VERSION}/"
echo "Next step: ./09-get-ratings.sh"
