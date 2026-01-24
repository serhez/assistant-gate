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
from transformers import AutoTokenizer

from paths import *
from utils import SINGLE_RATER_SYS_PROMPT, SINGLE_RATER_PROMPT


# import models
from AG.models.openai.azure import AsyncAzureChatLLM
from AG.models.openai.gpt4 import GPT4Agent
from AG.models.openrouter.openrouter import AsyncOpenRouterChatLLM, OpenRouterAgent


# logging
logging.basicConfig(level=logging.INFO)


def extract_score(response: str) -> int:
    """Extract numeric score from 'Final Score: X' in response."""
    match = re.search(r'Final Score:\s*(\d+)', response)
    if match:
        score = int(match.group(1))
        return min(max(score, 1), 10)  # Clamp to 1-10
    return None


def extract_reasoning(response: str) -> str:
    """Extract reasoning from response."""
    match = re.search(r'Reasoning:\s*(.+?)(?=Final Score:|$)', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info(f"Rating responses for model {args.qa_model.shortname} on {args.split.name} split...")
    random.seed(1)

    # Load rating model (supports both Azure OpenAI and OpenRouter)
    is_openrouter = "openrouter" in args.rating_model.model_type.lower()
    if is_openrouter:
        rating_llm = AsyncOpenRouterChatLLM(**args.rating_model.model_config.get('openrouter_api', {}))
        rating_model = OpenRouterAgent(llm=rating_llm, **args.rating_model.run.completion_config)
    else:
        rating_llm = AsyncAzureChatLLM(**args.rating_model.model_config.azure_api)
        rating_model = GPT4Agent(llm=rating_llm, **args.rating_model.run.completion_config)

    # Create output directory for single model
    output_dir = f'{WINRATE_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load personas, prompts
    with open(f"{PERSONAS_PATH}/{VERSION_2_BSFT}/{args.split.name}.json", 'r') as f:
        personas = json.load(f)
    with open(f"{PROMPT_PATH}/{VERSION_2_BSFT}/{args.split.name}.json", "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]

    # Load responses for the single model
    qa_responses = []
    for i in range(1, args.MAX_TURNS + 1):
        response_file = f'{WINRATE_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}_turn-{i}_responses_zero_shot.json'
        qa_responses.append(json.load(open(response_file, 'r')))

    # Sample keys for each turn
    all_keys = list(qa_responses[0].keys())
    samples_per_turn = args.n // 3
    turn_keys = [
        random.sample(all_keys, samples_per_turn),
        random.sample(all_keys, samples_per_turn),
        random.sample(all_keys, samples_per_turn)
    ]

    # Track all scores for aggregate statistics
    all_scores = []
    turn_scores = {1: [], 2: [], 3: []}

    for t_num, group in enumerate(turn_keys):
        turn = t_num + 1
        logging.info(f"Processing turn {turn}...")

        # Extract prompt and persona indices from keys
        group_prompt_indices = [int(key[key.find('prompt-') + len('prompt-'):key.find('persona-')].strip()) for key in group]
        group_persona_indices = [int(key[key.find('persona-') + len('persona-'):].strip()) for key in group]

        group_prompts = [prompts[idx] for idx in group_prompt_indices]
        group_personas = [personas[idx] for idx in group_persona_indices]
        group_responses = [qa_responses[t_num][key] for key in group]

        # Create rating prompts using single-model template
        rating_prompts = [
            SINGLE_RATER_PROMPT.format(persona, prompt, response)
            for persona, prompt, response in zip(group_personas, group_prompts, group_responses)
        ]

        # Get ratings from model
        logging.info(f"Sending {len(rating_prompts)} prompts to rating model...")
        rating_messages = rating_model.batch_prompt(
            system_message=SINGLE_RATER_SYS_PROMPT,
            messages=rating_prompts
        )
        rating_messages = [msg[0] for msg in rating_messages]

        # Parse scores and reasoning
        turn_ratings = {}
        for key, raw_response in zip(group, rating_messages):
            score = extract_score(raw_response)
            reasoning = extract_reasoning(raw_response)

            if score is not None:
                turn_ratings[key] = {
                    "score": score,
                    "reasoning": reasoning
                }
                all_scores.append(score)
                turn_scores[turn].append(score)
            else:
                logging.warning(f"Could not extract score for {key}, raw response: {raw_response[:200]}")
                turn_ratings[key] = {
                    "score": None,
                    "reasoning": reasoning,
                    "raw_response": raw_response
                }

        # Save per-turn ratings
        turn_output_file = f'{output_dir}/{args.split.name}_turn-{turn}_ratings.json'
        with open(turn_output_file, 'w') as f:
            json.dump(turn_ratings, f, indent=2)
        logging.info(f"Saved turn {turn} ratings to {turn_output_file}")

    # Compute aggregate statistics
    summary = {
        "model": args.qa_model.shortname,
        "split": args.split.name,
        "overall_mean": float(np.mean(all_scores)) if all_scores else None,
        "overall_std": float(np.std(all_scores)) if all_scores else None,
        "turn_1_mean": float(np.mean(turn_scores[1])) if turn_scores[1] else None,
        "turn_2_mean": float(np.mean(turn_scores[2])) if turn_scores[2] else None,
        "turn_3_mean": float(np.mean(turn_scores[3])) if turn_scores[3] else None,
        "turn_1_std": float(np.std(turn_scores[1])) if turn_scores[1] else None,
        "turn_2_std": float(np.std(turn_scores[2])) if turn_scores[2] else None,
        "turn_3_std": float(np.std(turn_scores[3])) if turn_scores[3] else None,
        "n_samples": len(all_scores),
        "n_turn_1": len(turn_scores[1]),
        "n_turn_2": len(turn_scores[2]),
        "n_turn_3": len(turn_scores[3])
    }

    # Save summary
    summary_file = f'{output_dir}/{args.split.name}_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"=" * 50)
    logging.info(f"RATING SUMMARY for {args.qa_model.shortname}")
    logging.info(f"=" * 50)
    logging.info(f"Overall Mean: {summary['overall_mean']:.2f} (std: {summary['overall_std']:.2f})")
    logging.info(f"Turn 1 Mean:  {summary['turn_1_mean']:.2f} (n={summary['n_turn_1']})")
    logging.info(f"Turn 2 Mean:  {summary['turn_2_mean']:.2f} (n={summary['n_turn_2']})")
    logging.info(f"Turn 3 Mean:  {summary['turn_3_mean']:.2f} (n={summary['n_turn_3']})")
    logging.info(f"Total Samples: {summary['n_samples']}")
    logging.info(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass
