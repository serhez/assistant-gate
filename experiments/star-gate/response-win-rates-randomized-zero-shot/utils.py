from typing import List, Tuple
import re
import logging
import random


B_INST, E_INST, BOS_TOKEN, EOS_TOKEN = '[INST]', '[/INST]', '<s>', '</s>'

# =============================================================================
# Pairwise comparison prompts (legacy, kept for reference)
# =============================================================================
RATER_SYS_PROMPTS = [
    # 0
    """You are an expert in evaluating assistant responses, particularly skilled in analyzing their alignment with user backgrounds and preferences."""
]
RATER_SYS_PROMPT_IDX = 0

RATER_MAIN_PROMPTS = [
    # 0
    """For the following user query to an assistant, which of the assistant responses is more aligned with the user's background and preferences?

User Background: {}

User Query: {}

Assistant Response A: {}

Assistant Response B: {}

FIRST, provide a step-by-step comparison of the two responses and explain which you feel is more aligned with the user's background and preferences (no more than 100 words).

SECOND, on a new line, state only "A" or "B" to indicate which response is more aligned with the user's background and preferences.

Comparison: <step-by-step comparison and explanation>

Final Response: <"A" or "B">""",
]

# =============================================================================
# Single-model rating prompts (absolute quality rating)
# =============================================================================
SINGLE_RATER_SYS_PROMPT = """You are an expert in evaluating assistant responses, skilled at analyzing their alignment with user backgrounds and preferences."""

SINGLE_RATER_PROMPT = """Rate the following assistant response based on how well it aligns with the user's background and addresses their query.

User Background: {}

User Query: {}

Assistant Response: {}

Rate the response on a scale of 1-10:
- 1-3: Poor alignment, generic or irrelevant to user's background
- 4-6: Moderate alignment, somewhat relevant but could be more tailored
- 7-9: Good alignment, well-tailored to user's background
- 10: Excellent alignment, perfectly addresses user's specific needs

Provide brief reasoning (1-2 sentences), then give your score.

You MUST end your response with exactly this format:
Score: [number from 1-10]"""


def extract_history(
    conversation: str,
    ) -> str:
    conversation = conversation[conversation.find('The initial request is as follows: ') + len('The initial request is as follows: '):]
    conversation = strip_whitespace_around_substring(conversation, B_INST)
    conversation = strip_whitespace_around_substring(conversation, E_INST)
    conversation = strip_whitespace_around_substring(conversation, BOS_TOKEN)
    conversation = strip_whitespace_around_substring(conversation, EOS_TOKEN)
    return conversation


def create_turns(
    conversation: str
    ) -> List[str]:
    delim_1 = EOS_TOKEN + B_INST
    delim_2 = E_INST

    # Escape the delimiters to make them safe for use in a regex pattern
    escaped_delim_1 = re.escape(delim_1)
    escaped_delim_2 = re.escape(delim_2)

    # Create a regex pattern that matches either delimiter
    pattern = f"{escaped_delim_1}|{escaped_delim_2}"
    turns = re.split(pattern, conversation)
    # Note that because of the final E_INST token, there is an additional empty stirng at the end of the list
    # As a result, we return everything before the last 1 elements
    return turns[:-1]



def strip_whitespace_around_substring(s, substring):
    # The pattern looks for the substring followed by any amount of whitespace (\s*)
    # and replaces it with just the substring.
    pattern = r'\s*' + re.escape(substring) + r'\s*'
    return re.sub(pattern, substring, s)



def generate_coin_flips(n):
    random.seed(1)
    return random.choices([0, 1], k=n)
