import os

## FILEPATHS
# These can be overridden by environment variables (set in config.env)
# Falls back to original hardcoded paths for backwards compatibility

_DEFAULT_DATA_ROOT = os.environ.get('DATA_ROOT', '/scr/andukuri/assistant-gate-hgx')

SIMULATION_PATH = os.environ.get('SIMULATION_PATH', f'{_DEFAULT_DATA_ROOT}/simulated-conversations')
LOGPROBS_PATH = os.environ.get('LOGPROBS_PATH', f'{_DEFAULT_DATA_ROOT}/log-probs')
PERSONAS_PATH = os.environ.get('PERSONAS_PATH', f'{_DEFAULT_DATA_ROOT}/personas')
GOLD_PATH = os.environ.get('GOLD_PATH', f'{_DEFAULT_DATA_ROOT}/gold-responses')
PROMPT_PATH = os.environ.get('PROMPT_PATH', f'{_DEFAULT_DATA_ROOT}/prompts')
SFT_DATA_PATH = os.environ.get('SFT_DATA_PATH', f'{_DEFAULT_DATA_ROOT}/sft-data')
CONTENT_VIOLATIONS_PATH = os.environ.get('CONTENT_VIOLATIONS_PATH', f'{_DEFAULT_DATA_ROOT}/content-violations')
FIGURES_PATH = os.environ.get('FIGURES_PATH', f'{_DEFAULT_DATA_ROOT}/figures')
WINRATE_PATH = os.environ.get('WINRATE_PATH', f'{_DEFAULT_DATA_ROOT}/win-rates')
MODELRESPONSE_PATH = os.environ.get('MODELRESPONSE_PATH', f'{_DEFAULT_DATA_ROOT}/model-responses')
SPECIFICITY_PATH = os.environ.get('SPECIFICITY_PATH', f'{_DEFAULT_DATA_ROOT}/specificity-ratings/')
MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', f'{_DEFAULT_DATA_ROOT}/pretrained_models')

# Version tags
VERSION = os.environ.get('VERSION', 'star-1')
VERSION_2 = 'star-2'
VERSION_AG = 'star-1-ag'
VERSION_2_BSFT = os.environ.get('VERSION', 'star-2-bsft')
VERSION_2_MISTRAL_ABLATION = 'star-2-mistral-ablation'
VERSION_1_ESFT = 'star-1-esft'
VERSION_3_QSFT = 'star-3-qsft'
VERSION_3_MISTRAL_ABLATION = 'star-3-mistral-ablation'
VERSION_1_MISTRAL_ABLATION = 'star-1-mistral-ablation'
VERSION_2_GEMMA_ABLATION = 'star-2-gemma-ablation'
LLAMA_VERSION = 'star-gate-llama3'
