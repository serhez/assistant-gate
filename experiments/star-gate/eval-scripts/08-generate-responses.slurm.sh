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
echo "Step 8: Generate Responses (Test Split)"
echo "=============================================="
echo "Model: ${CUSTOM_MODEL_ID}"
echo "=============================================="

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

# Create output directories
mkdir -p "${WINRATE_PATH}/${VERSION}/${CUSTOM_MODEL_NAME}"

# Navigate to script directory
cd "${PROJECT_ROOT}/experiments/star-gate/response-win-rates-randomized-zero-shot"

echo "Generating responses from ${CUSTOM_MODEL_NAME}..."
python get-responses.py \
    answer_model=${CUSTOM_MODEL_NAME} \
    qa_model=${CUSTOM_MODEL_NAME} \
    split=test

echo ""
echo "Done! Responses saved to: ${WINRATE_PATH}/${VERSION}/${CUSTOM_MODEL_NAME}/"
echo ""
echo "To evaluate another model, update CUSTOM_MODEL_ID and CUSTOM_MODEL_NAME in config.env"
echo "and re-run: ./00-setup-custom-model-configs.sh && sbatch 08-generate-responses.slurm.sh"
echo ""
echo "Once you have responses for all models you want to compare, run: ./09-get-ratings.sh"
