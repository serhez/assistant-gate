#!/bin/bash
# =============================================================================
# Step 7: Filter Top-K Conversations by Log Probability
# =============================================================================
# This script filters conversations to keep top-k by oracle likelihood.
# Runs locally (no GPU required).
#
# Usage: ./07-filter-log-probs.sh
# =============================================================================

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

echo "=============================================="
echo "Step 7: Filter Log Probabilities (Test Split)"
echo "=============================================="
echo "Top-K: ${TOP_K}"
echo "=============================================="

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

# Ensure output directory exists (should already exist from step 06)
mkdir -p "${LOGPROBS_PATH}/${VERSION}/qa-experimental/${CUSTOM_MODEL_NAME}"

# Navigate to script directory
cd "${PROJECT_ROOT}/experiments/star-gate/log-probs"

echo "Filtering top-${TOP_K} conversations..."
python filter.py \
    model=${CUSTOM_MODEL_NAME} \
    qa_model=${CUSTOM_MODEL_NAME} \
    condition=qa-experimental \
    split=test \
    k=${TOP_K}

echo ""
echo "Done! Filtered conversations at: ${LOGPROBS_PATH}/${VERSION}/qa-experimental/${CUSTOM_MODEL_NAME}/"
echo "Next step: ./08-generate-responses.slurm.sh (requires GPU)"
