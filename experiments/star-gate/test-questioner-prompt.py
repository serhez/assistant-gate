#!/usr/bin/env python3
"""
Quick test script for iterating on questioner prompts.

Usage:
    python test-questioner-prompt.py

This script uses the EXACT same prompt construction as generate-qa.py.
To test different prompts, modify QA_PROMPTS[QA_PROMPT_IDX] in
simulate-conversations/utils.py, or change QA_PROMPT_IDX below.
"""

import os
import sys

# Add parent directory to path so we can import from simulate-conversations
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simulate-conversations'))

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils import QA_PROMPTS

# =============================================================================
# CONFIGURATION - Modify these as needed
# =============================================================================

MODEL_ID = "iaa01/qwen3-4b-elicit-pos-ckpt56"
TENSOR_PARALLEL_SIZE = 1  # Adjust based on your GPU setup

# Sampling parameters (must match 00-setup-custom-model-configs.sh)
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
MAX_TOKENS = 700  # Matches experiment config

# Which QA prompt to use (must match simulate-conversations/conf/config.yaml)
QA_PROMPT_IDX = 13

# Sample user requests to test with (actual samples from experiments)
USER_REQUESTS = [
    "Is there a chart I can follow to keep track of the US Presidential election.",
    "What is the most efficient way to clean a bathroom.",
    "Can you recommend me a good book to read.",
]

# Sample user name (matches experiment format)
USER_NAME = "Alex"

# =============================================================================
# SCRIPT - No need to modify below
# =============================================================================

def main():
    print("=" * 60)
    print("Loading model:", MODEL_ID)
    print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=os.getenv("HF_TOKEN")
    )

    # Load model
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="auto",
    )

    # Sampling params
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
    )

    # Get the prompt template (same as generate-qa.py)
    qa_prompt_template = QA_PROMPTS[QA_PROMPT_IDX]

    print("\n" + "=" * 60)
    print(f"PROMPT TEMPLATE (QA_PROMPTS[{QA_PROMPT_IDX}]):")
    print("=" * 60)
    print(qa_prompt_template.format("[NAME]", "[REQUEST]"))
    print("=" * 60 + "\n")

    # Format prompts using chat template (EXACTLY like generate-qa.py lines 189-198)
    formatted_prompts = []
    for request in USER_REQUESTS:
        # This matches: user_content = system_content.format(names[j], prompt)
        user_content = qa_prompt_template.format(USER_NAME, request)
        messages = [{"role": "user", "content": user_content}]
        formatted = tokenizer.decode(
            tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
        )
        formatted_prompts.append(formatted)

    # Show first formatted prompt for verification
    print("=" * 60)
    print("FIRST FORMATTED PROMPT (as sent to model):")
    print("=" * 60)
    print(formatted_prompts[0])
    print("=" * 60 + "\n")

    # Generate
    print("Generating responses...\n")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Display results
    for i, (request, output) in enumerate(zip(USER_REQUESTS, outputs)):
        response = output.outputs[0].text.strip()
        question_marks = response.count("?")

        print("=" * 60)
        print(f"TEST {i+1}")
        print("=" * 60)
        print(f"USER REQUEST: {request}")
        print("-" * 40)
        print(f"MODEL RESPONSE:\n{response}")
        print("-" * 40)
        print(f"QUESTION MARKS: {question_marks} {'[OK]' if question_marks == 1 else '[FAIL - multiple questions!]'}")
        print()

if __name__ == "__main__":
    main()
