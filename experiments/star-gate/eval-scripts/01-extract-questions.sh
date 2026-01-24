#!/bin/bash
# =============================================================================
# Step 1: Extract Questions from Source Dataset
# =============================================================================
# This script extracts initial tasks/prompts from the source dataset.
# Runs locally (no GPU required).
#
# Usage: ./01-extract-questions.sh
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
echo "Step 1: Extract Questions"
echo "=============================================="

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

# Note: paths.py now reads from environment variables (set via config.env).
# No need to overwrite it - just ensure the env vars are exported.

# Create output directories
mkdir -p "${PROMPT_PATH}/${VERSION}"

# Navigate to script directory and run
cd "${PROJECT_ROOT}/experiments/star-gate/instruct-questions"
python extract-questions.py

echo ""
echo "Done! Questions extracted to: ${PROMPT_PATH}/${VERSION}/"
echo "Next step: ./02-generate-personas.sh"
