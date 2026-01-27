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

from utils import *


# import models
from AG.models.huggingface.hf_inference_model import HFInferenceModel
from AG.models.openai.azure import AsyncAzureChatLLM
from AG.models.openai.gpt4 import GPT4Agent
from AG.models.vllm_models.inference_model import VLLMInferenceModel
from AG.models.openrouter.openrouter import AsyncOpenRouterChatLLM, OpenRouterAgent
from paths import *

# logging
logging.basicConfig(level=logging.INFO)

# Conversation separator - must match generate-qa.py
TURN_SEP = "<|TURN_SEP|>"
MSG_SEP = "<|MSG_SEP|>"


def format_conversation_for_storage(messages: list) -> str:
    """Store conversation in a model-agnostic format."""
    parts = []
    for msg in messages:
        parts.append(f"{msg['role']}{MSG_SEP}{msg['content']}")
    return TURN_SEP.join(parts)


def parse_stored_conversation(conversation: str) -> tuple:
    """Parse a stored conversation back into original prompt and messages."""
    original_prompt = ""
    messages = []

    if conversation.startswith("ORIGINAL_PROMPT:"):
        parts = conversation.split(TURN_SEP, 1)
        original_prompt = parts[0].replace("ORIGINAL_PROMPT:", "").strip()
        conv_part = parts[1] if len(parts) > 1 else ""
    else:
        # Legacy format - try to extract from conversation
        conv_part = conversation

    turns = conv_part.split(TURN_SEP)
    for turn in turns:
        if MSG_SEP in turn:
            role, content = turn.split(MSG_SEP, 1)
            messages.append({"role": role.strip(), "content": content.strip()})

    return original_prompt, messages


def format_history_for_human(messages: list) -> str:
    """Format conversation history for human role-player prompt."""
    history_parts = []
    for msg in messages:
        if msg["role"] == "user":
            history_parts.append(f"You: {msg['content']}")
        else:
            history_parts.append(f"AI Assistant: {msg['content']}")
    return "\n".join(history_parts)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info(f"Loading models for multi-turn dialogue for {args.split.name}...")
    random.seed(1)
    if args.turn.number > args.MAX_TURNS:
        logging.info(f"Cannot exceed {args.MAX_TURNS} turns per conversation. Exiting.")
        return
    
    with open(f'{PROMPT_PATH}/{VERSION_2_BSFT}/{args.split.name}.json', "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
    # Load personas
    with open(f'{PERSONAS_PATH}/{VERSION_2_BSFT}/{args.split.name}.json', "r") as f:
        personas = json.load(f)
    # Load names
    with open(f'{PERSONAS_PATH}/{VERSION_2_BSFT}/{args.split.name}_NAMES.json', "r") as f:
        names = json.load(f)
    
    if not os.path.exists(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}"):
        os.makedirs(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}")

    pulled_conversations = json.load(open(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}.json", 'r'))
    output_conversations = defaultdict(list)

    # Detect model type and initialize appropriate model
    is_openrouter = args.human_model.model_type.lower() == "openrouter"

    if is_openrouter:
        logging.info("Using OpenRouter API for human model...")
        llm = AsyncOpenRouterChatLLM(**(args.human_model.model_config.get('openrouter_api') or {}))
        human_model = OpenRouterAgent(llm=llm, **args.human_model.run.completion_config)
    else:
        logging.info("Using vLLM for human model...")
        human_model = VLLMInferenceModel(**args.human_model.model_config)
    for j, persona in enumerate(personas):
        logging.info(f"Beginning simulations for persona {j}...")
        
        # get list of all prompts for that persona in order
        # get all keys from pulled_conversation with 
        raw_conversation_keys = [key if key.strip().endswith(f'persona-{j}') else None for key in pulled_conversations.keys()]
        conversation_keys = list()
        for key in raw_conversation_keys:
            if key is not None:
                conversation_keys.append(key)
        prompt_keys = [int(key[key.find('prompt-') + len('prompt-') : key.find('persona')].strip()) for key in conversation_keys]
        interest_prompts = [prompts[prompt_id] for prompt_id in prompt_keys]
        conversations = [pulled_conversations[f'prompt-{right_key} persona-{j}'] for right_key in prompt_keys] 
        flattened_prompt_ids = flatten_list([[prompt_id] * args.qa_model.run.initial_completion_config.num_return_sequences for prompt_id in prompt_keys])
        flattened_prompts = flatten_list([[prompt] * args.qa_model.run.initial_completion_config.num_return_sequences for prompt in interest_prompts])

        flattened_conversations = flatten_list(conversations)

        prompt_batches = list(batch_list(flattened_prompts, args.human_model.run.batch_size))

        conversation_batches = list(batch_list(flattened_conversations, args.human_model.run.batch_size))  # Create batches of prompts        
        final_conversations = list()

        for batch_index, conversation_batch in enumerate(conversation_batches):
            logging.info(f"Running batch {batch_index} of {len(conversation_batches)}...")

            # Parse stored conversations to extract messages and original prompts
            parsed_data = [parse_stored_conversation(c) for c in conversation_batch]
            original_prompts_batch = [p[0] for p in parsed_data]
            messages_batch = [p[1] for p in parsed_data]

            # Format history for human role-player
            histories = [format_history_for_human(msgs) for msgs in messages_batch]

            # Get the last assistant message (the question to answer)
            last_questions = []
            for msgs in messages_batch:
                last_q = ""
                for msg in reversed(msgs):
                    if msg["role"] == "assistant":
                        last_q = msg["content"]
                        break
                last_questions.append(last_q)

            if is_openrouter:
                # OpenRouter uses separate system message and user messages
                system_message = HUMAN_SYS_MSGS[args.HUMAN_SYS_PROMPT_IDX]
                user_messages = [
                    HUMAN_PROMPTS[args.HUMAN_PROMPT_IDX].format(persona, prompt, history)
                    for prompt, history in zip(prompt_batches[batch_index], histories)
                ]
                human_responses = human_model.batch_prompt(system_message, user_messages)
                human_responses = flatten_list(human_responses)  # Flatten list of lists
            else:
                # vLLM - use tokenizer's chat template if available
                roleplay_prompts = [
                    f"{BOS_TOKEN}{B_INST} {HUMAN_SYS_MSGS[args.HUMAN_SYS_PROMPT_IDX]}\n\n{HUMAN_PROMPTS[args.HUMAN_PROMPT_IDX].format(persona, prompt, history)}{E_INST}"
                    for prompt, history in zip(prompt_batches[batch_index], histories)
                ]
                human_responses = human_model.batch_prompt(roleplay_prompts, **args.human_model.run.completion_config)

            # Append human responses to conversations in model-agnostic format
            updated_conversations = []
            for orig_prompt, msgs, human_response in zip(original_prompts_batch, messages_batch, human_responses):
                # Add human response as a user message
                updated_msgs = msgs + [{"role": "user", "content": human_response}]
                stored_conv = f"ORIGINAL_PROMPT: {orig_prompt}{TURN_SEP}{format_conversation_for_storage(updated_msgs)}"
                updated_conversations.append(stored_conv)

            final_conversations.extend(updated_conversations)
            
        
        final_conversations = chunk_list(final_conversations, args.qa_model.run.initial_completion_config.num_return_sequences)
        final_prompt_ids = chunk_list(flattened_prompt_ids, args.qa_model.run.initial_completion_config.num_return_sequences)
        for i, sublist in enumerate(final_conversations):
            pair_key = f"prompt-{prompt_keys[i]} persona-{j}"
            output_conversations[pair_key].extend(sublist)

    if args.turn.number != args.MAX_TURNS:
        # sample int over uniform [int(args.turn.number), args.MAX_TURNS] inclusive
        samples = [random.randint(int(args.turn.number), args.MAX_TURNS) for key in output_conversations.keys()]
        # create mask 1 if int == args.turn.number else 0
        mask = [1 if sample == args.turn.number else 0 for sample in samples]
        finished_keys, unfinished_keys = [key for key, mask_value in zip(output_conversations.keys(), mask) if mask_value == 1], [key for key, mask_value in zip(output_conversations.keys(), mask) if mask_value == 0]
        # Creating finished and unfinished dictionaries from those keys
        finished, unfinished = {key: output_conversations[key] for key in finished_keys}, {key: output_conversations[key] for key in unfinished_keys}
    
        print('Finished: ', finished.keys())
        print('Continuing', unfinished.keys())
        if not os.path.exists(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}"):
            os.makedirs(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}")

        with open(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}.json", 'w') as f:
            json.dump(unfinished, f)
        with open(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}_turn-{args.turn.number}.json", 'w') as f:
            json.dump(finished, f)
    else:
        with open(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}_turn-{args.turn.number}.json", 'w') as f:
            json.dump(output_conversations, f)
        



if __name__ == '__main__':
    main()

