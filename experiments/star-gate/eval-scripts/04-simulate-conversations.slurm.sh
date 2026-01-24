#!/bin/bash
# =============================================================================
# Step 4: Simulate Multi-Turn Conversations (SLURM)
# =============================================================================
# This script simulates multi-turn QA conversations using your custom model.
# Requires GPU access via SLURM.
#
# Usage: sbatch 04-simulate-conversations.slurm.sh
#
# Before running:
#   1. Edit config.env with your model and cluster settings
#   2. Create custom model configs (see 04a-setup-custom-model-configs.sh)
# =============================================================================

#SBATCH --job-name=stargate-simulate
#SBATCH --output=logs/04-simulate-%j.out
#SBATCH --error=logs/04-simulate-%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32

# These are set dynamically from config.env, but provide defaults for SLURM
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
echo "Step 4: Simulate Conversations (Test Split)"
echo "=============================================="
echo "QA Model: ${CUSTOM_MODEL_ID}"
echo "QA Model Name: ${CUSTOM_MODEL_NAME}"
echo "Human Model Backend: ${HUMAN_MODEL_BACKEND}"
echo "Human Model ID: ${HUMAN_MODEL_ID}"
echo "GPUs: ${NUM_GPUS}"
echo "=============================================="

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

# Create output directories
mkdir -p "${SIMULATION_PATH}/${VERSION}/${CUSTOM_MODEL_NAME}"

# Navigate to script directory
cd "${PROJECT_ROOT}/experiments/star-gate/simulate-conversations"

# Run conversation simulation for each turn
for ((turn=1; turn<=MAX_TURNS; turn++)); do
    echo ""
    echo "=== Turn ${turn} of ${MAX_TURNS} ==="

    # Generate QA (questions from model)
    echo "Generating questions (turn ${turn})..."
    python generate-qa.py \
        qa_model=${CUSTOM_MODEL_NAME} \
        human_model=${CUSTOM_MODEL_NAME} \
        split=test \
        turn="t${turn}"

    # Generate human responses (using configured HUMAN_MODEL_BACKEND)
    echo "Generating human responses (turn ${turn})..."
    python generate-human.py \
        qa_model=${CUSTOM_MODEL_NAME} \
        human_model=${HUMAN_MODEL_BACKEND} \
        split=test \
        turn="t${turn}"
done

echo ""
echo "Done! Conversations saved to: ${SIMULATION_PATH}/${VERSION}/${CUSTOM_MODEL_NAME}/"
echo "Next step: ./05-pool-conversations.sh"
