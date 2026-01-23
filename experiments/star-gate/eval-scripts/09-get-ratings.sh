#!/bin/bash
# =============================================================================
# Step 9: Get Win Rate Ratings
# =============================================================================
# This script uses OpenRouter (DeepSeek) to evaluate pairwise response quality.
# Runs locally (no GPU required, uses API).
#
# Usage: ./09-get-ratings.sh
#
# Required: OPENROUTER_API_KEY environment variable
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

echo "=============================================="
echo "Step 9: Get Win Rate Ratings (Test Split)"
echo "=============================================="
echo "Comparing: baseline vs ${CUSTOM_MODEL_NAME}"
echo "Rating model: ${RATING_MODEL}"
echo "=============================================="

# Verify API key
if [[ -z "${OPENROUTER_API_KEY:-}" || "${OPENROUTER_API_KEY}" == "your-openrouter-api-key" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set in config.env"
    exit 1
fi

# Navigate to script directory
cd "${PROJECT_ROOT}/experiments/star-gate/response-win-rates-randomized-zero-shot"

echo "Computing win rates: baseline vs ${CUSTOM_MODEL_NAME}..."
python get-ratings.py \
    rating_model=openrouter \
    qa_model=baseline \
    qa_model_2=${CUSTOM_MODEL_NAME} \
    split=test

echo ""
echo "=============================================="
echo "EVALUATION COMPLETE!"
echo "=============================================="
echo ""
echo "Results saved to: ${WINRATE_PATH}/${VERSION}/baseline_${CUSTOM_MODEL_NAME}/"
echo ""
echo "To analyze results, look for win rate JSON files in the output directory."
