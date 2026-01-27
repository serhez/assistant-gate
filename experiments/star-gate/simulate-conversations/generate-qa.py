## GENERATE QA:

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
from paths import *


# import models
from AG.models.huggingface.hf_inference_model import HFInferenceModel
from AG.models.openai.azure import AsyncAzureChatLLM
from AG.models.openai.gpt4 import GPT4Agent
from AG.models.vllm_models.inference_model import VLLMInferenceModel


# logging
logging.basicConfig(level=logging.INFO)


# Conversation separator - used to delimit turns in stored conversations
# This is model-agnostic and used for parsing later
TURN_SEP = "<|TURN_SEP|>"
MSG_SEP = "<|MSG_SEP|>"


def format_prompt_for_model(tokenizer, system_msg: str, user_msg: str) -> str:
    """Format a prompt using the model's native chat template."""
    messages = [{"role": "user", "content": f"{system_msg}\n\n{user_msg}"}]
    return tokenizer.decode(tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False))


def format_conversation_for_storage(messages: list) -> str:
    """Store conversation in a model-agnostic format.

    Format: role<|MSG_SEP|>content<|TURN_SEP|>role<|MSG_SEP|>content...
    """
    parts = []
    for msg in messages:
        parts.append(f"{msg['role']}{MSG_SEP}{msg['content']}")
    return TURN_SEP.join(parts)


def parse_stored_conversation(conversation: str) -> list:
    """Parse a stored conversation back into messages."""
    messages = []
    turns = conversation.split(TURN_SEP)
    for turn in turns:
        if MSG_SEP in turn:
            role, content = turn.split(MSG_SEP, 1)
            messages.append({"role": role.strip(), "content": content.strip()})
    return messages


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info(f"Loading models for multi-turn dialogue for {args.split.name}...")
    random.seed(1)
    
    if args.turn.number > args.MAX_TURNS:
        logging.info(f"Cannot exceed {args.MAX_TURNS} turns per conversation. Exiting.")
        return
    
    # Load prompts
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
    
    
    if os.path.exists(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}.json"):
        qa_model = VLLMInferenceModel(**args.qa_model.model_config)
        # Load tokenizer for proper chat template formatting
        tokenizer = AutoTokenizer.from_pretrained(
            args.qa_model.model_config.model,
            token=os.getenv("HF_TOKEN")
        )

        pulled_conversations = json.load(open(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}.json", 'r'))
        output_conversations = defaultdict(list)
        logging.info(f"Continuing from existing conversations: {len(pulled_conversations.keys())} keys")

        for j, persona in enumerate(personas):
            logging.info(f"Beginning simulations for persona {j}...")

            raw_conversation_keys = [key if key.strip().endswith(f'persona-{j}') else None for key in pulled_conversations.keys()]
            conversation_keys = [key for key in raw_conversation_keys if key is not None]

            prompt_keys = [int(key[key.find('prompt-') + len('prompt-') : key.find('persona')].strip()) for key in conversation_keys]
            interest_prompts = [prompts[prompt_id] for prompt_id in prompt_keys]
            conversations = [pulled_conversations[f'prompt-{right_key} persona-{j}'] for right_key in prompt_keys]

            flattened_prompt_ids = flatten_list([[prompt_id] * args.qa_model.run.initial_completion_config.num_return_sequences for prompt_id in prompt_keys])
            flattened_prompts = flatten_list([[prompt] * args.qa_model.run.initial_completion_config.num_return_sequences for prompt in interest_prompts])
            flattened_conversations = flatten_list(conversations)

            conversation_batches = list(batch_list(flattened_conversations, args.qa_model.run.batch_size))
            prompt_id_batches = list(batch_list(flattened_prompt_ids, args.qa_model.run.batch_size))

            final_conversations = list()
            for batch_index, conversation_batch in enumerate(conversation_batches):
                logging.info(f"Running batch {batch_index} of {len(conversation_batches)}...")

                # Parse stored conversations and format for model
                formatted_prompts = []
                parsed_messages_list = []
                original_prompts_list = []
                for conv in conversation_batch:
                    # Extract original prompt and messages from stored format
                    if conv.startswith("ORIGINAL_PROMPT:"):
                        parts = conv.split(TURN_SEP, 1)
                        original_prompt = parts[0].replace("ORIGINAL_PROMPT:", "").strip()
                        messages = parse_stored_conversation(parts[1]) if len(parts) > 1 else []
                    else:
                        # Legacy format fallback - try to parse as before
                        original_prompt = ""
                        messages = parse_stored_conversation(conv)

                    original_prompts_list.append(original_prompt)

                    # Add reminder to ask ONE question for turns 2-3
                    # (Turn 1 has the instruction in QA_PROMPTS, but subsequent turns need reinforcement)
                    if messages and messages[-1]["role"] == "user":
                        messages[-1]["content"] += "\n\n[Remember: Ask exactly ONE follow-up question about a DIFFERENT aspect. Do not ask multiple questions.]"

                    parsed_messages_list.append(messages)

                    # Format for model using chat template
                    formatted_prompt = tokenizer.decode(
                        tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
                    )
                    formatted_prompts.append(formatted_prompt)

                qa_responses = qa_model.batch_prompt(formatted_prompts, **args.qa_model.run.completion_config)

                # Append responses and store in new format
                for idx, (conv, qa_response, messages, orig_prompt) in enumerate(
                    zip(conversation_batch, qa_responses, parsed_messages_list, original_prompts_list)
                ):
                    # Add new assistant response to messages
                    updated_messages = messages + [{"role": "assistant", "content": qa_response}]
                    stored_conv = f"ORIGINAL_PROMPT: {orig_prompt}{TURN_SEP}{format_conversation_for_storage(updated_messages)}"
                    final_conversations.append(stored_conv)

            final_conversations = chunk_list(final_conversations, args.qa_model.run.initial_completion_config.num_return_sequences)
            for i, sublist in enumerate(final_conversations):
                pair_key = f"prompt-{prompt_keys[i]} persona-{j}"
                output_conversations[pair_key].extend(sublist)

        with open(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}.json", 'w') as f:
            json.dump(output_conversations, f)

    else:
        qa_model = VLLMInferenceModel(**args.qa_model.model_config)
        # Load tokenizer for proper chat template formatting
        tokenizer = AutoTokenizer.from_pretrained(
            args.qa_model.model_config.model,
            token=os.getenv("HF_TOKEN")
        )

        output_conversations = defaultdict(list)
        for j, persona in enumerate(personas):
            logging.info(f"Beginning simulations for persona {j}...")
            prompt_batches = list(batch_list(prompts, args.qa_model.run.batch_size))  # Create batches of prompts
            for batch_index, prompt_batch in enumerate(prompt_batches):
                logging.info(f"Running batch {batch_index} of {len(prompt_batches)}...")

                # Format prompts using the model's native chat template
                system_content = QA_PROMPTS[args.QA_PROMPT_IDX]
                initial_messages = []
                initial_prompts = []
                for prompt in prompt_batch:
                    user_content = system_content.format(names[j], prompt)
                    messages = [{"role": "user", "content": user_content}]
                    formatted_prompt = tokenizer.decode(
                        tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
                    )
                    initial_prompts.append(formatted_prompt)
                    initial_messages.append({"user_prompt": prompt, "messages": messages})

                # QA Model initial turn
                qa_responses = qa_model.batch_prompt(initial_prompts, **args.qa_model.run.initial_completion_config)

                conversations = list()
                for i, sublist in enumerate(chunk_list(qa_responses, args.qa_model.run.initial_completion_config.num_return_sequences)):
                    for qa_response in sublist:
                        # Store conversation in model-agnostic format
                        conv_messages = [
                            {"role": "user", "content": initial_messages[i]["messages"][0]["content"]},
                            {"role": "assistant", "content": qa_response}
                        ]
                        # Include original prompt for later reference
                        stored_conv = f"ORIGINAL_PROMPT: {initial_messages[i]['user_prompt']}{TURN_SEP}{format_conversation_for_storage(conv_messages)}"
                        conversations.append(stored_conv)

                final_conversations = chunk_list(conversations, args.qa_model.run.initial_completion_config.num_return_sequences)
                for i, sublist in enumerate(final_conversations):
                    pair_key = f"prompt-{i} persona-{j}"
                    output_conversations[pair_key].extend(sublist)

        if not os.path.exists(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}"):
            os.makedirs(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}")
        with open(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}.json", 'w') as f:
            json.dump(output_conversations, f)


if __name__ == '__main__':
    main()


