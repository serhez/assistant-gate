#!/bin/bash
# =============================================================================
# Step 9: Get Response Ratings (Single-Model Evaluation)
# =============================================================================
# This script uses OpenRouter (DeepSeek) to rate response quality for a single
# model using absolute scoring (1-10 scale).
#
# Runs locally (no GPU required, uses API).
#
# Usage: ./09-get-ratings.sh
#
# Before running:
#   1. Generate responses for the model via step 08
#   2. Set CUSTOM_MODEL_NAME in config.env (or export before running)
#
# Required: OPENROUTER_API_KEY environment variable
#
# To evaluate multiple models, run this script separately for each:
#   CUSTOM_MODEL_NAME=baseline ./09-get-ratings.sh
#   CUSTOM_MODEL_NAME=my-model ./09-get-ratings.sh
# =============================================================================

set -euo pipefail

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
echo "Step 9: Get Response Ratings (Test Split)"
echo "=============================================="
echo "Model: ${CUSTOM_MODEL_NAME}"
echo "Rating model: ${RATING_MODEL}"
echo "=============================================="

# Verify API key
if [[ -z "${OPENROUTER_API_KEY:-}" || "${OPENROUTER_API_KEY}" == "your-openrouter-api-key" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set in config.env"
    exit 1
fi

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

# Create output directory
mkdir -p "${WINRATE_PATH}/${VERSION}/${CUSTOM_MODEL_NAME}"

# Navigate to script directory
cd "${PROJECT_ROOT}/experiments/star-gate/response-win-rates-randomized-zero-shot"

echo "Computing ratings for model: ${CUSTOM_MODEL_NAME}..."
python get-ratings.py \
    rating_model=openrouter \
    qa_model=custom \
    qa_model.shortname=${CUSTOM_MODEL_NAME} \
    split=test

echo ""
echo "=============================================="
echo "EVALUATION COMPLETE!"
echo "=============================================="
echo ""
echo "Results saved to: ${WINRATE_PATH}/${VERSION}/${CUSTOM_MODEL_NAME}/"
echo ""
echo "Output files:"
echo "  - ${CUSTOM_MODEL_NAME}/test_turn-1_ratings.json  (per-response scores for turn 1)"
echo "  - ${CUSTOM_MODEL_NAME}/test_turn-2_ratings.json  (per-response scores for turn 2)"
echo "  - ${CUSTOM_MODEL_NAME}/test_turn-3_ratings.json  (per-response scores for turn 3)"
echo "  - ${CUSTOM_MODEL_NAME}/test_summary.json         (aggregate statistics)"
echo "  - ${CUSTOM_MODEL_NAME}/test_results.json         (detailed results with full context)"
echo ""
echo "To evaluate another model:"
echo "  CUSTOM_MODEL_NAME=other-model ./09-get-ratings.sh"
