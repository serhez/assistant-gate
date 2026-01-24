#!/bin/bash
# =============================================================================
# Step 3: Build Gold (Oracle) Responses
# =============================================================================
# This script generates oracle/gold standard responses using OpenRouter (DeepSeek).
# Runs locally (no GPU required, uses API).
#
# Usage: ./03-build-gold-responses.sh
#
# Required: OPENROUTER_API_KEY environment variable
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

echo "=============================================="
echo "Step 3: Build Gold Responses (Test Split)"
echo "=============================================="

# Verify API key
if [[ -z "${OPENROUTER_API_KEY:-}" || "${OPENROUTER_API_KEY}" == "your-openrouter-api-key" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set in config.env"
    exit 1
fi

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

# Create output directory
mkdir -p "${GOLD_PATH}/${VERSION}"

# Navigate to script directory
cd "${PROJECT_ROOT}/experiments/star-gate/build-gold-responses"

echo "Generating gold responses for test split using OpenRouter..."
echo "Oracle model: ${ORACLE_MODEL}"
python generate.py model=openrouter split=test

echo ""
echo "Done! Gold responses generated at: ${GOLD_PATH}/${VERSION}/test.json"
echo "Next step: ./04-simulate-conversations.sh (requires GPU)"
