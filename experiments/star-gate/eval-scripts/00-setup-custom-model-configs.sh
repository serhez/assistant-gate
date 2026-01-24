#!/bin/bash
# =============================================================================
# Step 0: Setup Custom Model Configurations
# =============================================================================
# This script creates Hydra config files for your custom HuggingFace model
# across all pipeline stages.
#
# Run this ONCE before starting the evaluation pipeline.
#
# Usage: ./00-setup-custom-model-configs.sh
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
echo "Step 0: Setup Custom Model Configurations"
echo "=============================================="
echo "Model ID: ${CUSTOM_MODEL_ID}"
echo "Model Name: ${CUSTOM_MODEL_NAME}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
echo "Batch Size: ${VLLM_BATCH_SIZE}"
echo "=============================================="

# -----------------------------------------------------------------------------
# 1. Simulate Conversations - QA Model Config
# -----------------------------------------------------------------------------
QA_MODEL_CONFIG="${PROJECT_ROOT}/experiments/star-gate/simulate-conversations/conf/qa_model/${CUSTOM_MODEL_NAME}.yaml"
echo "Creating: ${QA_MODEL_CONFIG}"
mkdir -p "$(dirname "${QA_MODEL_CONFIG}")"
cat > "${QA_MODEL_CONFIG}" << EOF
hydra:
  run:
    dir: outputs

model_type: vllm
name: ${CUSTOM_MODEL_NAME}-vllm
shortname: ${CUSTOM_MODEL_NAME}

model_config:
  model: ${CUSTOM_MODEL_ID}
  dtype: auto
  tensor_parallel_size: ${TENSOR_PARALLEL_SIZE}
  seed: 1

run:
  batch_size: ${VLLM_BATCH_SIZE}
  verbose: false
  initial_completion_config:
    do_sample: true
    temperature: 0.9
    top_p: 0.9
    max_new_tokens: 700
    num_return_sequences: ${NUM_RETURN_SEQUENCES}
  completion_config:
    do_sample: true
    temperature: 0.9
    top_p: 0.9
    max_new_tokens: 700
    num_return_sequences: 1
EOF

# -----------------------------------------------------------------------------
# 2. Simulate Conversations - Human Model Config
# -----------------------------------------------------------------------------
HUMAN_MODEL_CONFIG="${PROJECT_ROOT}/experiments/star-gate/simulate-conversations/conf/human_model/${CUSTOM_MODEL_NAME}.yaml"
echo "Creating: ${HUMAN_MODEL_CONFIG}"
mkdir -p "$(dirname "${HUMAN_MODEL_CONFIG}")"
cat > "${HUMAN_MODEL_CONFIG}" << EOF
hydra:
  run:
    dir: outputs

model_type: vllm
name: ${CUSTOM_MODEL_NAME}-vllm
shortname: ${CUSTOM_MODEL_NAME}

model_config:
  model: ${CUSTOM_MODEL_ID}
  dtype: auto
  tensor_parallel_size: ${TENSOR_PARALLEL_SIZE}
  seed: 1

run:
  batch_size: ${VLLM_BATCH_SIZE}
  verbose: false
  completion_config:
    do_sample: false
    temperature: 0
    top_p: 0.9
    max_new_tokens: 700
    num_return_sequences: 1
EOF

# -----------------------------------------------------------------------------
# 3. Log-Probs - Model Config
# -----------------------------------------------------------------------------
LOGPROBS_MODEL_CONFIG="${PROJECT_ROOT}/experiments/star-gate/log-probs/conf/model/${CUSTOM_MODEL_NAME}.yaml"
echo "Creating: ${LOGPROBS_MODEL_CONFIG}"
mkdir -p "$(dirname "${LOGPROBS_MODEL_CONFIG}")"
cat > "${LOGPROBS_MODEL_CONFIG}" << EOF
hydra:
  run:
    dir: outputs

model_type: vllm
name: ${CUSTOM_MODEL_NAME}-vllm
shortname: ${CUSTOM_MODEL_NAME}

model_config:
  model: ${CUSTOM_MODEL_ID}
  dtype: auto
  tensor_parallel_size: 2
  seed: 1

run:
  batch_size: ${VLLM_BATCH_SIZE}
  verbose: false
  completion_config:
    max_new_tokens: 1
    temperature: 0
    prompt_logprobs: 1
EOF

# -----------------------------------------------------------------------------
# 4. Log-Probs - QA Model Config
# -----------------------------------------------------------------------------
LOGPROBS_QA_CONFIG="${PROJECT_ROOT}/experiments/star-gate/log-probs/conf/qa_model/${CUSTOM_MODEL_NAME}.yaml"
echo "Creating: ${LOGPROBS_QA_CONFIG}"
mkdir -p "$(dirname "${LOGPROBS_QA_CONFIG}")"
cat > "${LOGPROBS_QA_CONFIG}" << EOF
hydra:
  run:
    dir: outputs

model_type: vllm
name: ${CUSTOM_MODEL_NAME}-vllm
shortname: ${CUSTOM_MODEL_NAME}

model_config:
  model: ${CUSTOM_MODEL_ID}
  dtype: auto
  tensor_parallel_size: 2
  seed: 1

run:
  batch_size: ${VLLM_BATCH_SIZE}
  verbose: false
EOF

# -----------------------------------------------------------------------------
# 5. Response Win Rates - Answer Model Config
# -----------------------------------------------------------------------------
for WINRATE_DIR in "response-win-rates" "response-win-rates-randomized" "response-win-rates-randomized-zero-shot" "response-win-rates-counterbalanced"; do
    ANSWER_MODEL_CONFIG="${PROJECT_ROOT}/experiments/star-gate/${WINRATE_DIR}/conf/answer_model/${CUSTOM_MODEL_NAME}.yaml"
    echo "Creating: ${ANSWER_MODEL_CONFIG}"
    mkdir -p "$(dirname "${ANSWER_MODEL_CONFIG}")"
    cat > "${ANSWER_MODEL_CONFIG}" << EOF
hydra:
  run:
    dir: outputs

model_type: vllm
name: ${CUSTOM_MODEL_NAME}-vllm
shortname: ${CUSTOM_MODEL_NAME}

model_config:
  model: ${CUSTOM_MODEL_ID}
  dtype: auto
  tensor_parallel_size: ${TENSOR_PARALLEL_SIZE}
  seed: 1

tokenizer_config:
  pretrained_model_name_or_path: ${CUSTOM_MODEL_ID}
  model_max_length: 1024

run:
  batch_size: ${VLLM_BATCH_SIZE}
  verbose: false
  completion_config:
    do_sample: false
    best_of: 1
    temperature: 0.0
    top_p: 1
    top_k: -1
    max_new_tokens: 700
    use_beam_search: false
    presence_penalty: 0
    frequency_penalty: 0
    num_return_sequences: 1
EOF
done

# -----------------------------------------------------------------------------
# 6. Response Win Rates - QA Model Config
# -----------------------------------------------------------------------------
for WINRATE_DIR in "response-win-rates" "response-win-rates-randomized" "response-win-rates-randomized-zero-shot" "response-win-rates-counterbalanced"; do
    QA_MODEL_CONFIG="${PROJECT_ROOT}/experiments/star-gate/${WINRATE_DIR}/conf/qa_model/${CUSTOM_MODEL_NAME}.yaml"
    echo "Creating: ${QA_MODEL_CONFIG}"
    mkdir -p "$(dirname "${QA_MODEL_CONFIG}")"
    cat > "${QA_MODEL_CONFIG}" << EOF
hydra:
  run:
    dir: outputs

model_type: vllm
name: ${CUSTOM_MODEL_NAME}-vllm
shortname: ${CUSTOM_MODEL_NAME}

model_config:
  model: ${CUSTOM_MODEL_ID}
  dtype: auto
  tensor_parallel_size: ${TENSOR_PARALLEL_SIZE}
  seed: 1

tokenizer_config:
  pretrained_model_name_or_path: ${CUSTOM_MODEL_ID}
  model_max_length: 1024

run:
  batch_size: ${VLLM_BATCH_SIZE}
  verbose: false
  completion_config:
    do_sample: false
    best_of: 1
    temperature: 0.0
    top_p: 1
    top_k: -1
    max_new_tokens: 700
    use_beam_search: false
    presence_penalty: 0
    frequency_penalty: 0
    num_return_sequences: 1
EOF
done

# -----------------------------------------------------------------------------
# 7. Response Win Rates - QA Model 2 Config (for comparison)
# -----------------------------------------------------------------------------
for WINRATE_DIR in "response-win-rates" "response-win-rates-randomized" "response-win-rates-randomized-zero-shot" "response-win-rates-counterbalanced"; do
    QA_MODEL_2_CONFIG="${PROJECT_ROOT}/experiments/star-gate/${WINRATE_DIR}/conf/qa_model_2/${CUSTOM_MODEL_NAME}.yaml"
    echo "Creating: ${QA_MODEL_2_CONFIG}"
    mkdir -p "$(dirname "${QA_MODEL_2_CONFIG}")"
    cat > "${QA_MODEL_2_CONFIG}" << EOF
hydra:
  run:
    dir: outputs

model_type: vllm
name: ${CUSTOM_MODEL_NAME}-vllm
shortname: ${CUSTOM_MODEL_NAME}

model_config:
  model: ${CUSTOM_MODEL_ID}
  dtype: auto
  tensor_parallel_size: ${TENSOR_PARALLEL_SIZE}
  seed: 1

tokenizer_config:
  pretrained_model_name_or_path: ${CUSTOM_MODEL_ID}
  model_max_length: 1024

run:
  batch_size: ${VLLM_BATCH_SIZE}
  verbose: false
  completion_config:
    do_sample: false
    best_of: 1
    temperature: 0.0
    top_p: 1
    top_k: -1
    max_new_tokens: 700
    use_beam_search: false
    presence_penalty: 0
    frequency_penalty: 0
    num_return_sequences: 1
EOF
done

echo ""
echo "=============================================="
echo "Configuration files created successfully!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Edit config.env with your actual settings"
echo "  2. Run: ./01-extract-questions.sh"
