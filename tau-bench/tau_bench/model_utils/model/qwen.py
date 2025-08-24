import os

from tau_bench.model_utils.api.datapoint import Datapoint
from tau_bench.model_utils.model.chat import ChatModel, Message
from tau_bench.model_utils.model.completion import approx_cost_for_datapoint, approx_prompt_str
from tau_bench.model_utils.model.general_model import wrap_temperature
from tau_bench.model_utils.model.utils import approx_num_tokens

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # If python-dotenv is not installed, skip loading .env

DEFAULT_QWEN_MODEL = "togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
API_KEY_ENV_VAR = "QWEN_API_KEY"
BASE_URL_ENV_VAR = "QWEN_BASE_URL"

PRICE_PER_INPUT_TOKEN_MAP = {
    "togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8": 0.0006 / 1000,  # $0.02 per 1K tokens
    "togetherai/Qwen/Qwen3-235B-A22B-FP8": 0.0006 / 1000,
    "togetherai/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8": 0.0006 / 1000
    # "qwen/qwen-max-longcontext": 0.02 / 1000,
    # "qwen/qwen-plus": 0.01 / 1000,  # $0.01 per 1K tokens
    # "qwen/qwen-turbo": 0.002 / 1000,  # $0.002 per 1K tokens
    # "qwen/qwen2.5-72b-instruct": 0.0,  # Local model
    # "qwen/qwen2.5-32b-instruct": 0.0,  # Local model
    # "qwen/qwen2.5-14b-instruct": 0.0,  # Local model
    # "qwen/qwen2.5-7b-instruct": 0.0,   # Local model
    # "qwen/qwen2.5-3b-instruct": 0.0,   # Local model
    # "Qwen/Qwen2.5-7B-Instruct": 0.0,   # Local model
    # "Qwen/Qwen2.5-14B-Instruct": 0.0,  # Local model
    # "Qwen/Qwen2.5-32B-Instruct": 0.0,  # Local model
    # "Qwen/Qwen2.5-72B-Instruct": 0.0,  # Local model
}
INPUT_PRICE_PER_TOKEN_FALLBACK = 0.01 / 1000

CAPABILITY_SCORE_MAP = {
    "togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8": 0.8,  # $0.02 per 1K tokens
    "togetherai/Qwen/Qwen3-235B-A22B-FP8": 0.8,
    "togetherai/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8": 0.85
    # "qwen/qwen-turbo": 0.7,
    # "qwen/qwen2.5-72b-instruct": 0.85,
    # "qwen/qwen2.5-32b-instruct": 0.8,
    # "qwen/qwen2.5-14b-instruct": 0.75,
    # "qwen/qwen2.5-7b-instruct": 0.7,
    # "qwen/qwen2.5-3b-instruct": 0.6,
    # "Qwen/Qwen2.5-7B-Instruct": 0.7,
    # "Qwen/Qwen2.5-14B-Instruct": 0.75,
    # "Qwen/Qwen2.5-32B-Instruct": 0.8,
    # "Qwen/Qwen2.5-72B-Instruct": 0.85,
}
CAPABILITY_SCORE_FALLBACK = 0.7

# TODO: implement
LATENCY_MS_PER_OUTPUT_TOKEN_MAP = {}
# TODO: implement
LATENCY_MS_PER_OUTPUT_TOKEN_FALLBACK = 0.0

MAX_CONTEXT_LENGTH_MAP = {
    "togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8": 256000,  # $0.02 per 1K tokens
    "togetherai/Qwen/Qwen3-235B-A22B-FP8": 256000,
    "togetherai/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8": 256000
    # "qwen/qwen-turbo": 128000,
    # "qwen/qwen2.5-72b-instruct": 131072,
    # "qwen/qwen2.5-32b-instruct": 131072,
    # "qwen/qwen2.5-14b-instruct": 131072,
    # "qwen/qwen2.5-7b-instruct": 131072,
    # "qwen/qwen2.5-3b-instruct": 131072,
    # "Qwen/Qwen2.5-7B-Instruct": 131072,
    # "Qwen/Qwen2.5-14B-Instruct": 131072,
    # "Qwen/Qwen2.5-32B-Instruct": 131072,
    # "Qwen/Qwen2.5-72B-Instruct": 131072,
}
MAX_CONTEXT_LENGTH_FALLBACK = 128000


class QwenModel(ChatModel):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        from openai import AsyncOpenAI, OpenAI

        if model is None:
            self.model = DEFAULT_QWEN_MODEL
        else:
            self.model = model

        # Load API key and base URL from environment if not provided
        # This will work with .env if python-dotenv is installed and load_dotenv() is called above
        if api_key is None:
            api_key = os.getenv(API_KEY_ENV_VAR)
        if not api_key:
            raise ValueError(f"{API_KEY_ENV_VAR} environment variable is not set or empty")

        if base_url is None:
            base_url = os.getenv(BASE_URL_ENV_VAR)
        if not base_url:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.temperature = temperature

    def generate_message(
        self,
        messages: list[Message],
        force_json: bool,
        temperature: float | None = None,
    ) -> Message:
        if temperature is None:
            temperature = self.temperature
        msgs = self.build_generate_message_state(messages)
        res = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=wrap_temperature(temperature),
            response_format={"type": "json_object" if force_json else "text"},
            max_tokens=4096,  # Add max_tokens parameter
        )
        return self.handle_generate_message_response(
            prompt=msgs, content=res.choices[0].message.content, force_json=force_json
        )

    def get_approx_cost(self, dp: Datapoint) -> float:
        cost_per_token = PRICE_PER_INPUT_TOKEN_MAP.get(self.model, INPUT_PRICE_PER_TOKEN_FALLBACK)
        return approx_cost_for_datapoint(dp=dp, price_per_input_token=cost_per_token)

    def get_latency(self, dp: Datapoint) -> float:
        latency_per_output_token = LATENCY_MS_PER_OUTPUT_TOKEN_MAP.get(
            self.model, LATENCY_MS_PER_OUTPUT_TOKEN_FALLBACK
        )
        return approx_cost_for_datapoint(dp=dp, price_per_input_token=latency_per_output_token)

    def get_capability(self) -> float:
        return CAPABILITY_SCORE_MAP.get(self.model, CAPABILITY_SCORE_FALLBACK)

    def supports_dp(self, dp: Datapoint) -> bool:
        prompt = approx_prompt_str(dp)
        return approx_num_tokens(prompt) <= MAX_CONTEXT_LENGTH_MAP.get(
            self.model, MAX_CONTEXT_LENGTH_FALLBACK
        ) 