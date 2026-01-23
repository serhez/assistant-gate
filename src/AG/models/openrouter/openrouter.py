from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import asyncio
import time
import os

from openai import AsyncOpenAI

import logging


logging.basicConfig(level=logging.INFO)


# Default cost per token (fallback if model not in list)
# These are approximate costs for common models on OpenRouter
# See https://openrouter.ai/docs#models for up-to-date pricing
MODEL_COST_PER_INPUT = {
    'openai/gpt-4': 3e-05,
    'openai/gpt-4-turbo': 1e-05,
    'openai/gpt-4o': 2.5e-06,
    'openai/gpt-4o-mini': 1.5e-07,
    'openai/gpt-3.5-turbo': 5e-07,
    'anthropic/claude-3-opus': 1.5e-05,
    'anthropic/claude-3-sonnet': 3e-06,
    'anthropic/claude-3-haiku': 2.5e-07,
    'anthropic/claude-3.5-sonnet': 3e-06,
    'deepseek/deepseek-chat': 2.7e-07,
    'deepseek/deepseek-v3.2': 2.5e-07,
    'meta-llama/llama-3-70b-instruct': 5.9e-07,
    'meta-llama/llama-3-8b-instruct': 5.5e-08,
    'mistralai/mistral-7b-instruct': 6e-08,
    'mistralai/mixtral-8x7b-instruct': 2.4e-07,
    'google/gemini-pro': 1.25e-07,
    'google/gemini-pro-1.5': 1.25e-06,
}

MODEL_COST_PER_OUTPUT = {
    'openai/gpt-4': 6e-05,
    'openai/gpt-4-turbo': 3e-05,
    'openai/gpt-4o': 1e-05,
    'openai/gpt-4o-mini': 6e-07,
    'openai/gpt-3.5-turbo': 1.5e-06,
    'anthropic/claude-3-opus': 7.5e-05,
    'anthropic/claude-3-sonnet': 1.5e-05,
    'anthropic/claude-3-haiku': 1.25e-06,
    'anthropic/claude-3.5-sonnet': 1.5e-05,
    'deepseek/deepseek-chat': 1.1e-06,
    'deepseek/deepseek-v3.2': 3.8e-07,
    'meta-llama/llama-3-70b-instruct': 7.9e-07,
    'meta-llama/llama-3-8b-instruct': 5.5e-08,
    'mistralai/mistral-7b-instruct': 6e-08,
    'mistralai/mixtral-8x7b-instruct': 2.4e-07,
    'google/gemini-pro': 3.75e-07,
    'google/gemini-pro-1.5': 5e-06,
}

# Default fallback costs (conservative estimate)
DEFAULT_COST_PER_INPUT = 1e-05
DEFAULT_COST_PER_OUTPUT = 3e-05


class AsyncOpenRouterChatLLM:
    """
    Wrapper for an (Async) OpenRouter Chat Model.

    OpenRouter provides a unified API for accessing multiple LLM providers
    including OpenAI, Anthropic, Meta, Google, Mistral, and more.

    See https://openrouter.ai/docs for documentation.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
    ):
        """
        Initializes AsyncOpenAI client configured for OpenRouter.

        Args:
            api_key: OpenRouter API key. If not provided, reads from OPENROUTER_API_KEY env var.
            base_url: OpenRouter API base URL (default: https://openrouter.ai/api/v1)
            site_url: Optional URL of your site (for OpenRouter rankings)
            site_name: Optional name of your site (for OpenRouter rankings)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Build default headers for OpenRouter
        default_headers = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if site_name:
            default_headers["X-Title"] = site_name

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            default_headers=default_headers if default_headers else None,
        )

    @property
    def llm_type(self):
        return "AsyncOpenRouter"

    async def __call__(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ):
        """
        Make an async API call to OpenRouter.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional arguments passed to the API (model, temperature, etc.)

        Returns:
            OpenAI ChatCompletion response object
        """
        return await self.client.chat.completions.create(
            messages=messages,
            **kwargs
        )


class OpenRouterAgent:
    """
    OpenRouter LLM wrapper for async API calls with cost tracking.

    This is analogous to GPT4Agent but works with any model available on OpenRouter.
    """
    def __init__(
        self,
        llm: AsyncOpenRouterChatLLM,
        model: str = "openai/gpt-4",
        **completion_config,
    ) -> None:
        """
        Initialize the OpenRouter agent.

        Args:
            llm: An AsyncOpenRouterChatLLM instance
            model: The model identifier (e.g., 'openai/gpt-4', 'anthropic/claude-3-opus')
            **completion_config: Additional config passed to API calls (temperature, max_tokens, etc.)
        """
        self.llm = llm
        self.model = model
        self.completion_config = completion_config
        self.completion_config['model'] = model
        self.all_responses = []
        self.total_inference_cost = 0

    def calc_cost(
        self,
        response,
    ) -> float:
        """
        Calculate the cost of an API response.

        Args:
            response: The response from the API.

        Returns:
            float: The cost of the response in USD.
        """
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        input_cost = MODEL_COST_PER_INPUT.get(self.model, DEFAULT_COST_PER_INPUT)
        output_cost = MODEL_COST_PER_OUTPUT.get(self.model, DEFAULT_COST_PER_OUTPUT)

        cost = (input_cost * input_tokens) + (output_cost * output_tokens)
        return cost

    def get_prompt(
        self,
        system_message: str,
        user_message: str,
    ) -> List[Dict[str, str]]:
        """
        Get the (zero shot) prompt for the (chat) model.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        return messages

    async def get_response(
        self,
        messages: List[Dict[str, str]],
    ) -> Any:
        """
        Get the response from the model.
        """
        return await self.llm(messages=messages, **self.completion_config)

    async def run(
        self,
        system_message: str,
        message: str,
    ) -> Dict[str, Any]:
        """
        Runs the OpenRouter Agent.

        Args:
            system_message (str): The system message to use
            message (str): The user message to use

        Returns:
            A list of response strings, or fallback response on failure.
        """
        messages = self.get_prompt(system_message=system_message, user_message=message)
        success = False

        for i in range(10):
            if i < 9:
                try:
                    response = await self.get_response(messages=messages)
                    success = True
                    break
                except Exception as e:
                    logging.warning(f"API call attempt {i+1} failed: {e}")
                    time.sleep(1)
            else:
                logging.error("API call failed after 10 attempts.")

        if success:
            cost = self.calc_cost(response=response)
            logging.info(f"Cost for running {self.model}: {cost}")

            full_response = {
                'response': response,
                'response_str': [r.message.content for r in response.choices],
                'cost': cost
            }
            # Update total cost and store response
            self.total_inference_cost += cost
            self.all_responses.append(full_response)

            return full_response['response_str']

        else:
            return "Final Response: C"

    async def batch_prompt_sync(
        self,
        system_message: str,
        messages: List[str],
    ) -> List[str]:
        """
        Handles async API calls for batch prompting.

        Args:
            system_message (str): The system message to use
            messages (List[str]): A list of user messages

        Returns:
            A list of responses from the model for each message
        """
        responses = [self.run(system_message, message) for message in messages]
        return await asyncio.gather(*responses)

    def batch_prompt(
        self,
        system_message: str,
        messages: List[str],
    ) -> List[str]:
        """
        Synchronous wrapper for batch_prompt.

        Args:
            system_message (str): The system message to use
            messages (List[str]): A list of user messages

        Returns:
            A list of responses from the model for each message
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("Loop is already running.")
        return loop.run_until_complete(self.batch_prompt_sync(system_message, messages))
