# STaR-GATE Evaluation Scripts

This directory contains scripts to run the STaR-GATE evaluation pipeline on the **test set only**, using:

- **Oracle Model**: DeepSeek V3.2 via OpenRouter
- **Rating Model**: DeepSeek V3.2 via OpenRouter
- **Evaluated Model**: Your custom HuggingFace model via vLLM

## Prerequisites

1. **Python environment** with the project installed:
   ```bash
   pip install -e .
   pip install flash-attn --no-build-isolation
   pip install -r requirements.txt
   ```

2. **API Keys**:
   - `OPENROUTER_API_KEY` - For DeepSeek oracle/rating calls
   - `HF_TOKEN` - If your model is private/gated on HuggingFace

3. **GPU Access** - For steps 4, 6, and 8 (vLLM inference)

## Quick Start

### 1. Configure Settings

Edit `config.env` with your settings:

```bash
# Your custom model
export CUSTOM_MODEL_ID="your-username/your-model-name"
export CUSTOM_MODEL_NAME="mymodel"

# API keys
export OPENROUTER_API_KEY="your-key"
export HF_TOKEN="your-token"

# Data paths
export DATA_ROOT="/path/to/your/data"

# GPU settings (for SLURM)
export NUM_GPUS=4
export TENSOR_PARALLEL_SIZE=4
export SLURM_PARTITION="gpu"
```

### 2. Run the Pipeline

```bash
# Step 0: Create model configs (run once)
./00-setup-custom-model-configs.sh

# Step 1-3: Data preparation (no GPU)
./01-extract-questions.sh
./02-generate-personas.sh
./03-build-gold-responses.sh

# Step 4: Simulate conversations (GPU - SLURM)
sbatch 04-simulate-conversations.slurm.sh

# Step 5: Pool conversations (no GPU)
./05-pool-conversations.sh

# Step 6: Compute log probabilities (GPU - SLURM)
sbatch 06-compute-log-probs.slurm.sh

# Step 7: Filter top-k conversations (no GPU)
./07-filter-log-probs.sh

# Step 8: Generate final responses (GPU - SLURM)
sbatch 08-generate-responses.slurm.sh

# Step 9: Get win rate ratings (no GPU)
./09-get-ratings.sh
```

## Pipeline Overview

| Step | Script | GPU | Description |
|------|--------|-----|-------------|
| 0 | `00-setup-custom-model-configs.sh` | No | Create Hydra configs for your model |
| 1 | `01-extract-questions.sh` | No | Extract tasks from source dataset |
| 2 | `02-generate-personas.sh` | No | Generate personas (OpenRouter API) |
| 3 | `03-build-gold-responses.sh` | No | Generate oracle responses (OpenRouter API) |
| 4 | `04-simulate-conversations.slurm.sh` | **Yes** | Simulate multi-turn conversations (vLLM) |
| 5 | `05-pool-conversations.sh` | No | Pool conversation variants |
| 6 | `06-compute-log-probs.slurm.sh` | **Yes** | Compute oracle log probabilities (vLLM) |
| 7 | `07-filter-log-probs.sh` | No | Filter top-k conversations |
| 8 | `08-generate-responses.slurm.sh` | **Yes** | Generate final responses (vLLM) |
| 9 | `09-get-ratings.sh` | No | Compute win rates (OpenRouter API) |

## Running Without SLURM

If you have direct GPU access (no SLURM), you can run the `.slurm.sh` scripts directly:

```bash
# Instead of: sbatch 04-simulate-conversations.slurm.sh
# Run:
bash 04-simulate-conversations.slurm.sh
```

The SLURM headers will be ignored when running with `bash`.

## Customizing SLURM Settings

Edit `config.env` to customize SLURM:

```bash
export SLURM_ACCOUNT="your-account"
export SLURM_PARTITION="your-partition"
export SLURM_NODE="-w your-node"  # Optional: specific node
```

Then update the SLURM scripts' headers, or override at submission:

```bash
sbatch --partition=my-partition --gres=gpu:8 04-simulate-conversations.slurm.sh
```

## Output Locations

All outputs are saved under `${DATA_ROOT}/${VERSION}/`:

- **Prompts**: `${PROMPT_PATH}/${VERSION}/test.json`
- **Personas**: `${PERSONAS_PATH}/${VERSION}/test.json`
- **Gold Responses**: `${GOLD_PATH}/${VERSION}/test.json`
- **Simulated Conversations**: `${SIMULATION_PATH}/${VERSION}/${CUSTOM_MODEL_NAME}/`
- **Log Probabilities**: `${LOGPROBS_PATH}/${VERSION}/qa-experimental/${CUSTOM_MODEL_NAME}/`
- **Win Rates**: `${WINRATE_PATH}/${VERSION}/baseline_${CUSTOM_MODEL_NAME}/`

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size in `config.env`:
```bash
export VLLM_BATCH_SIZE=50  # Default: 100
```

### Model Loading Issues
Ensure `HF_TOKEN` is set for private/gated models:
```bash
export HF_TOKEN="hf_your_token"
```

### OpenRouter API Errors
Check your API key and rate limits:
```bash
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer ${OPENROUTER_API_KEY}"
```

## Files

```
eval-scripts/
├── README.md                           # This file
├── config.env                          # Configuration (edit this!)
├── 00-setup-custom-model-configs.sh    # Create model configs
├── 01-extract-questions.sh             # Extract tasks
├── 02-generate-personas.sh             # Generate personas
├── 03-build-gold-responses.sh          # Generate oracle responses
├── 04-simulate-conversations.slurm.sh  # Simulate conversations (GPU)
├── 05-pool-conversations.sh            # Pool conversations
├── 06-compute-log-probs.slurm.sh       # Compute log probs (GPU)
├── 07-filter-log-probs.sh              # Filter top-k
├── 08-generate-responses.slurm.sh      # Generate responses (GPU)
├── 09-get-ratings.sh                   # Get win rates
└── logs/                               # SLURM job logs
```
