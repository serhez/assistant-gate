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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

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
