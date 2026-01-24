#!/bin/bash
# =============================================================================
# Step 5: Pool Conversations
# =============================================================================
# This script pools the simulated conversations.
# Runs locally (no GPU required).
#
# Usage: ./05-pool-conversations.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

echo "=============================================="
echo "Step 5: Pool Conversations (Test Split)"
echo "=============================================="

# Ensure output directory exists (should already exist from step 04)
mkdir -p "${SIMULATION_PATH}/${VERSION}/${CUSTOM_MODEL_NAME}"

# Navigate to script directory
cd "${PROJECT_ROOT}/experiments/star-gate/simulate-conversations"

echo "Pooling conversations for ${CUSTOM_MODEL_NAME}..."
python pool-conversations.py \
    qa_model=${CUSTOM_MODEL_NAME} \
    split=test \
    k=${TOP_K}

echo ""
echo "Done! Pooled conversations at: ${SIMULATION_PATH}/${VERSION}/${CUSTOM_MODEL_NAME}/"
echo "Next step: ./06-compute-log-probs.slurm.sh (requires GPU)"
