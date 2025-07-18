import logging
import time

import requests

# REWRITTEN: Import the new client and response model for the Completion API
from openai import AsyncOpenAI
from openai.types import Completion
from tenacity import retry, stop_after_attempt, wait_fixed

from evals.apis.inference.openai.base import OpenAIModel
from evals.apis.inference.openai.utils import count_tokens, price_per_token
from evals.data_models.inference import LLMResponse
from evals.data_models.messages import Prompt
from evals.utils import COMPLETION_MODELS

LOGGER = logging.getLogger(__name__)

# REWRITTEN: Instantiate the asynchronous client.
client = AsyncOpenAI(api_key="DummyKey")


class OpenAICompletionModel(OpenAIModel):
    def _assert_valid_id(self, model_id: str):
        assert model_id in COMPLETION_MODELS, f"Invalid model id: {model_id}"

    @retry(stop=stop_after_attempt(8), wait=wait_fixed(2))
    async def _get_dummy_response_header(self, model_id: str):
        url = "https://api.openai.com/v1/completions"
        headers = {
            "Content-Type": "application/json",
            # REWRITTEN: Use the client's api_key attribute.
            "Authorization": f"Bearer {client.api_key}",
            "OpenAI-Organization": self.organization,
        }
        data = {"model": model_id, "prompt": "a", "max_tokens": 1}
        response = requests.post(url, headers=headers, json=data)
        if "x-ratelimit-limit-tokens" not in response.headers:
            raise RuntimeError("Failed to get dummy response header")
        return response.headers

    @staticmethod
    def _count_prompt_token_capacity(prompt: Prompt, **kwargs) -> int:
        # No changes needed here as it doesn't use the openai library directly.
        max_tokens = kwargs.get("max_tokens", 15)
        n = kwargs.get("n", 1)
        completion_tokens = n * max_tokens

        prompt_tokens = count_tokens(str(prompt))
        return prompt_tokens + completion_tokens

    async def _make_api_call(self, prompt: Prompt, model_id, start_time, **params) -> list[LLMResponse]:
        LOGGER.debug(f"Making {model_id} call with {self.organization}")
        prompt_file = self.create_prompt_history_file(str(prompt), model_id, self.prompt_history_dir)
        api_start = time.time()

        # REWRITTEN: Use the new client syntax for the legacy Completion API call.
        api_response: Completion = await client.completions.create(
            prompt=str(prompt), model=model_id, organization=self.organization, **params
        )

        api_duration = time.time() - api_start
        duration = time.time() - start_time
        context_token_cost, completion_token_cost = price_per_token(model_id)

        # REWRITTEN: `api_response.usage` can be None. Handle this case.
        context_cost = 0
        if api_response.usage:
            context_cost = api_response.usage.prompt_tokens * context_token_cost

        # REWRITTEN: The response structure is very similar, so only minor changes are needed here.
        # The dot notation access (`choice.text`, `choice.finish_reason`) remains the same.
        responses = [
            LLMResponse(
                model_id=model_id,
                completion=choice.text,
                stop_reason=choice.finish_reason,
                api_duration=api_duration,
                duration=duration,
                cost=context_cost + count_tokens(choice.text) * completion_token_cost,
                # This access pattern for logprobs is still correct in v1.x for this endpoint.
                logprobs=choice.logprobs.top_logprobs if choice.logprobs is not None else None,
            )
            for choice in api_response.choices
        ]
        self.add_response_to_prompt_file(prompt_file, responses)
        return responses
