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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

echo "=============================================="
echo "Step 1: Extract Questions"
echo "=============================================="

# Update paths.py with your data paths
echo "Updating paths.py..."
cat > "${PROJECT_ROOT}/src/paths.py" << EOF
## FILEPATHS
SIMULATION_PATH = '${SIMULATION_PATH}'
LOGPROBS_PATH = '${LOGPROBS_PATH}'
PERSONAS_PATH = '${PERSONAS_PATH}'
GOLD_PATH = '${GOLD_PATH}'
PROMPT_PATH = '${PROMPT_PATH}'
SFT_DATA_PATH = '${DATA_ROOT}/sft-data'
CONTENT_VIOLATIONS_PATH = '${DATA_ROOT}/content-violations'
FIGURES_PATH = '${DATA_ROOT}/figures'
WINRATE_PATH = '${WINRATE_PATH}'
MODELRESPONSE_PATH = '${DATA_ROOT}/model-responses'
SPECIFICITY_PATH = '${DATA_ROOT}/specificity-ratings/'
VERSION = 'star-1'
VERSION_2 = 'star-2'
VERSION_AG = 'star-1-ag'
VERSION_2_BSFT = '${VERSION}'
VERSION_2_MISTRAL_ABLATION = 'star-2-mistral-ablation'
VERSION_1_ESFT = 'star-1-esft'
VERSION_3_QSFT = 'star-3-qsft'
VERSION_3_MISTRAL_ABLATION = 'star-3-mistral-ablation'
VERSION_1_MISTRAL_ABLATION = 'star-1-mistral-ablation'
VERSION_2_GEMMA_ABLATION = 'star-2-gemma-ablation'
LLAMA_VERSION = 'star-gate-llama3'
EOF

# Create output directories
mkdir -p "${PROMPT_PATH}/${VERSION}"

# Navigate to script directory and run
cd "${PROJECT_ROOT}/experiments/star-gate/instruct-questions"
python extract-questions.py

echo ""
echo "Done! Questions extracted to: ${PROMPT_PATH}/${VERSION}/"
echo "Next step: ./02-generate-personas.sh"
