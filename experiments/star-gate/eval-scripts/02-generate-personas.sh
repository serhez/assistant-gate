#!/bin/bash
# =============================================================================
# Step 2: Generate Personas for Test Split
# =============================================================================
# This script generates diverse user personas using OpenRouter (DeepSeek).
# Runs locally (no GPU required, uses API).
#
# Usage: ./02-generate-personas.sh
#
# Required: OPENROUTER_API_KEY environment variable
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
echo "Step 2: Generate Personas (Test Split)"
echo "=============================================="

# Verify API key
if [[ -z "${OPENROUTER_API_KEY:-}" || "${OPENROUTER_API_KEY}" == "your-openrouter-api-key" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set in config.env"
    exit 1
fi

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

# Create output directory
mkdir -p "${PERSONAS_PATH}/${VERSION}"

# Navigate to script directory
cd "${PROJECT_ROOT}/experiments/star-gate/persona-generation"

echo "Generating personas for test split using OpenRouter..."
python generate-personas-test.py model=openrouter split=test

echo ""
echo "Done! Personas generated at:"
echo "  - ${PERSONAS_PATH}/${VERSION}/test.json"
echo "  - ${PERSONAS_PATH}/${VERSION}/test_NAMES.json"
echo "Next step: ./03-build-gold-responses.sh"
