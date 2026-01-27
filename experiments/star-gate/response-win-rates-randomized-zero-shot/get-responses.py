import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import argparse
import fire
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import logging
import torch
import random
import re
import time
import os
import gc
import signal
from collections import defaultdict
from datasets import load_dataset, Dataset
from vllm.distributed.parallel_state import destroy_model_parallel
from transformers import AutoTokenizer

from paths import *
from utils import *


# import models
from AG.models.huggingface.hf_inference_model import HFInferenceModel
from AG.models.openai.azure import AsyncAzureChatLLM
from AG.models.openai.gpt4 import GPT4Agent
from AG.models.vllm_models.inference_model import VLLMInferenceModel


# logging
logging.basicConfig(level=logging.INFO)

# Conversation separators - must match simulate-conversations scripts
TURN_SEP = "<|TURN_SEP|>"
MSG_SEP = "<|MSG_SEP|>"


def parse_stored_conversation(conversation: str) -> tuple:
    """Parse a stored conversation back into original prompt and messages.

    Supports both new model-agnostic format and legacy Mistral format.
    Returns: (original_prompt, messages_list)
    """
    # Check for new format
    if TURN_SEP in conversation and MSG_SEP in conversation:
        original_prompt = ""
        messages = []

        if conversation.startswith("ORIGINAL_PROMPT:"):
            parts = conversation.split(TURN_SEP, 1)
            original_prompt = parts[0].replace("ORIGINAL_PROMPT:", "").strip()
            conv_part = parts[1] if len(parts) > 1 else ""
        else:
            conv_part = conversation

        turns = conv_part.split(TURN_SEP)
        for turn in turns:
            if MSG_SEP in turn:
                role, content = turn.split(MSG_SEP, 1)
                messages.append({"role": role.strip(), "content": content.strip()})

        return original_prompt, messages

    # Legacy Mistral format - use old parsing
    else:
        # Extract using legacy method
        conversation = extract_history(conversation)
        turns = create_turns(conversation)
        # Convert to messages format
        messages = []
        for i, turn in enumerate(turns):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn})
        # Try to extract original prompt from first user message
        original_prompt = turns[0] if turns else ""
        return original_prompt, messages


def clean_response(response: str) -> str:
    """Clean model response by removing stop tokens and artifacts.

    Handles common issues like:
    - Repeated stop tokens: </s>, [/S], <|endoftext|>, etc.
    - Leading/trailing artifacts: ], ", etc.
    - Excessive whitespace
    """
    if not response:
        return ""

    # Common stop tokens and artifacts to remove
    stop_patterns = [
        r'</s>',
        r'\[/S\]',
        r'<\|endoftext\|>',
        r'<\|end\|>',
        r'<\|eot_id\|>',
        r'<\|end_of_text\|>',
        r'\$\[S\]\$',
        r'\$\[E\]\$',
    ]

    cleaned = response

    # Remove repeated stop tokens
    for pattern in stop_patterns:
        cleaned = re.sub(f'({pattern}\\s*)+', '', cleaned, flags=re.IGNORECASE)

    # Remove leading/trailing brackets, quotes, and whitespace artifacts
    cleaned = cleaned.strip()
    cleaned = re.sub(r'^[\]\[\"\'\s]+', '', cleaned)
    cleaned = re.sub(r'[\]\[\"\'\s]+$', '', cleaned)
    cleaned = cleaned.strip()

    return cleaned


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info(f"Loading model {args.qa_model.shortname} conversations for response generation and win-rate computation for {args.split.name}...")
    random.seed(1)

    answer_model = VLLMInferenceModel(**args.answer_model.model_config)
    
    if not os.path.exists(f'{WINRATE_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}'):
        os.makedirs(f'{WINRATE_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}')
    
    
    # Load personas, prompts
    with open(f"{PERSONAS_PATH}/{VERSION_2_BSFT}/{args.split.name}.json", 'r') as f:
        personas = json.load(f)
    with open(f"{PERSONAS_PATH}/{VERSION_2_BSFT}/{args.split.name}_NAMES.json", 'r') as f:
        names = json.load(f)
    with open(f"{PROMPT_PATH}/{VERSION_2_BSFT}/{args.split.name}.json", "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
    with open(f"{GOLD_PATH}/{VERSION_2_BSFT}/{args.split.name}.json", "r") as f:
        gold = json.load(f)
        
    turns_conversations = []
    for i in range(1, args.MAX_TURNS + 1):
        turns_conversations.append(json.load(open(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}_turn-{i}.json", "r")))
    pooled_conversations = json.load(open(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}{f'_top-k-{args.k}' if args.k > 0 else ''}.json", "r"))

    # Use ALL available samples from each turn (no sampling)
    turn_sizes = [len(turns_conversations[i].keys()) for i in range(args.MAX_TURNS)]
    total_samples = sum(turn_sizes)
    logging.info(f"Using ALL available samples: {turn_sizes} per turn (total: {total_samples})")

    tokenizer = AutoTokenizer.from_pretrained(**args.qa_model.tokenizer_config)
    turns_1, turns_2, turns_3 = list(turns_conversations[0].keys()), list(turns_conversations[1].keys()), list(turns_conversations[2].keys())
    all_qa_responses = list()
    for t_num, group in enumerate([turns_1, turns_2, turns_3]):
        # group is a list of keys f'prompt-{i} persona-{j}' where prompt and persona can be used to index the prompts and personas list above
        
        conversations = [random.choice(pooled_conversations[key]) for key in group]
        
        group_answer_prompts = list()
        group_prompt_indices = [int(key[key.find('prompt-') + len('prompt-'):key.find('persona-')].strip()) for key in group]
        group_persona_indices = [int(key[key.find('persona-') + len('persona-'):].strip()) for key in group]
        for c_idx, conversation in enumerate(conversations):
            # Parse conversation (supports both new and legacy formats)
            original_prompt, messages = parse_stored_conversation(conversation)

            # If no original prompt found, use from prompts list
            if not original_prompt:
                original_prompt = prompts[group_prompt_indices[c_idx]]

            # Prepend user's name to the first message
            if messages and messages[0]["role"] == "user":
                messages[0]["content"] = f"My name is {names[group_persona_indices[c_idx]]}.\n\n{messages[0]['content']}"

            # Append original prompt to the last message to prompt for final answer
            if messages:
                messages[-1]["content"] += f"\n\nNow please provide a helpful answer to my original question: {original_prompt}"

            final_prompt = tokenizer.decode(tokenizer.apply_chat_template(messages, add_generation_prompt=True))
            group_answer_prompts.append(final_prompt)

            # Log first prompt of each turn for debugging
            if c_idx == 0:
                logging.info(f"=== Example prompt for turn {t_num + 1} ===")
                logging.info(f"Original prompt: {original_prompt[:100]}...")
                logging.info(f"Number of messages: {len(messages)}")
                logging.info(f"Messages structure: {[(m['role'], len(m['content'])) for m in messages]}")
                logging.info(f"Final prompt (first 1500 chars):\n{final_prompt[:1500]}")
                logging.info(f"=== End example prompt ===")

        group_answer_responses = answer_model.batch_prompt(group_answer_prompts, **args.answer_model.run.completion_config)

        # Clean responses and track empty ones
        cleaned_responses = []
        empty_count = 0
        for idx, resp in enumerate(group_answer_responses):
            cleaned = clean_response(resp)
            if not cleaned:
                empty_count += 1
                logging.warning(f"Empty response after cleaning for {group[idx]}, raw: {resp[:100] if resp else '(empty)'}...")
            cleaned_responses.append(cleaned)

        if empty_count > 0:
            logging.warning(f"Turn {t_num + 1}: {empty_count}/{len(group_answer_responses)} responses were empty after cleaning")

        with open(f'{WINRATE_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}_turn-{t_num + 1}_responses_zero_shot.json', 'w') as f:
            json.dump(dict(zip(group, cleaned_responses)), f)

        # Also save the full prompts for debugging/verification
        with open(f'{WINRATE_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}_turn-{t_num + 1}_prompts.json', 'w') as f:
            json.dump(dict(zip(group, group_answer_prompts)), f)
       
    
    
if __name__ == '__main__':
    main()
