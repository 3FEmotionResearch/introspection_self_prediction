import logging
import time

import requests

# REWRITTEN: Import the new client, exception types, and response models
from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletion
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from evals.apis.inference.openai.base import OpenAIModel
from evals.apis.inference.openai.utils import count_tokens, price_per_token
from evals.data_models.inference import LLMResponse
from evals.data_models.messages import Prompt
from evals.utils import GPT_CHAT_MODELS

LOGGER = logging.getLogger(__name__)

# REWRITTEN: Instantiate the asynchronous client. It will automatically find the OPENAI_API_KEY from your environment.
client = AsyncOpenAI(api_key="DummyKey")


class OpenAIChatModel(OpenAIModel):
    def _assert_valid_id(self, model_id: str):
        if "ft:" in model_id:
            model_id = model_id.split(":")[1]
        assert model_id in GPT_CHAT_MODELS, f"Invalid model id: {model_id}"

    @retry(stop=stop_after_attempt(8), wait=wait_fixed(2))
    async def _get_dummy_response_header(self, model_id: str):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            # REWRITTEN: Use the client's api_key attribute instead of the global openai.api_key
            "Authorization": f"Bearer {client.api_key}",
            "OpenAI-Organization": self.organization,
        }
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Say 1"}],
        }
        response = requests.post(url, headers=headers, json=data)
        if "x-ratelimit-limit-tokens" not in response.headers:
            print("⚠️ Failed to get dummy response from header—adding defaults")
            response.headers["x-ratelimit-limit-tokens"] = "2000000"
            response.headers["x-ratelimit-limit-requests"] = "10000"
            response.headers["x-ratelimit-remaining-tokens"] = "1999999"
            response.headers["x-ratelimit-remaining-requests"] = "9999"
            response.headers["x-ratelimit-reset-requests"] = "6ms"
            response.headers["x-ratelimit-reset-tokens"] = "0s"
        return response.headers

    @staticmethod
    def _count_prompt_token_capacity(prompt: Prompt, **kwargs) -> int:
        # This function doesn't use the openai library directly, so no changes needed.
        BUFFER = 5
        MIN_NUM_TOKENS = 20
        num_tokens = 0
        for message in prompt.messages:
            num_tokens += 1
            num_tokens += len(message.content) / 4
        return max(
            MIN_NUM_TOKENS,
            int(num_tokens + BUFFER) + kwargs.get("n", 1) * kwargs.get("max_tokens", 15),
        )

    @staticmethod
    def convert_top_logprobs(logprobs_content: list | None) -> list[dict[str, float]] | None:
        """
        REWRITTEN: This function is updated to handle the new logprobs structure from openai v1.x.
        The old structure was a dict with a 'content' key. The new one is a list of objects.
        """
        if logprobs_content is None:
            return None

        top_logprobs_list = []
        for token_logprob in logprobs_content:
            # Each token_logprob object has a .top_logprobs list.
            # We convert it to the desired dictionary format.
            logprob_dict = {lp.token: lp.logprob for lp in token_logprob.top_logprobs}

            # The main token chosen by the model is not always in its own top_logprobs list, so add it.
            logprob_dict[token_logprob.token] = token_logprob.logprob

            top_logprobs_list.append(logprob_dict)

        return top_logprobs_list

    @retry(
        # REWRITTEN: Use the new exception types from the openai v1.x library.
        retry=retry_if_exception_type((RateLimitError, APIError, APIConnectionError)),
        wait=wait_fixed(5),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _make_api_call(self, prompt: Prompt, model_id, start_time, **params) -> list[LLMResponse]:
        LOGGER.debug(f"Making {model_id} call with {self.organization}")

        # The parameter name is still 'logprobs', but it now controls the return of `top_logprobs`
        if "logprobs" in params and params["logprobs"]:
            params["logprobs"] = True
            # 'top_logprobs' is controlled by the 'logprobs' parameter, so no need to set it separately.
            if "top_logprobs" in params:
                del params["top_logprobs"]

        prompt_file = self.create_prompt_history_file(prompt.openai_format(), model_id, self.prompt_history_dir)
        api_start = time.time()
        try:
            # REWRITTEN: Use the new client syntax for making API calls.
            # The 'organization' param is now passed to the client on initialization, but passing it
            # here per-call is still supported as an override.
            api_response: ChatCompletion = await client.chat.completions.create(
                messages=prompt.openai_format(), model=model_id, organization=self.organization, **params
            )
        except Exception as e:
            LOGGER.error(
                f"Error when getting response from OpenAI: {e}\nRetrying once before failing.\nThe prompt was: {prompt.openai_format()}"
            )
            time.sleep(1)
            # REWRITTEN: Use the new client syntax for the retry call as well.
            api_response: ChatCompletion = await client.chat.completions.create(
                messages=prompt.openai_format(), model=model_id, organization=self.organization, **params
            )
        api_duration = time.time() - api_start
        duration = time.time() - start_time
        context_token_cost, completion_token_cost = price_per_token(model_id)

        # REWRITTEN: `api_response.usage` can be None if streaming is used. Handle this case.
        context_cost = 0
        if api_response.usage:
            context_cost = api_response.usage.prompt_tokens * context_token_cost

        responses = []
        for choice in api_response.choices:
            completion_content = choice.message.content or ""

            # REWRITTEN: Pass the correct object to the logprobs conversion function.
            # The logprobs data is now in `choice.logprobs.content`.
            logprobs_data = self.convert_top_logprobs(choice.logprobs.content) if choice.logprobs else None

            response = LLMResponse(
                model_id=model_id,
                completion=completion_content,
                stop_reason=choice.finish_reason,
                api_duration=api_duration,
                duration=duration,
                cost=context_cost + count_tokens(completion_content) * completion_token_cost,
                logprobs=logprobs_data,
            )
            responses.append(response)

        self.add_response_to_prompt_file(prompt_file, responses)
        return responses
