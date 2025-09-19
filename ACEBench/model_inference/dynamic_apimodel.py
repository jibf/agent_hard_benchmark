import os
from model_inference.apimodel_inference import APIModelInference


def create_dynamic_apimodel_class(api_key=None, base_url=None):
    """
    Factory function to create a dynamic APIModelInference class with custom API key and base URL.
    This allows ACEBench to work with arbitrary API providers using environment variables.
    """

    class DynamicAPIModelInference(APIModelInference):
        def __init__(self, model_name, model_path=None, temperature=0.001, top_p=1, max_tokens=1000, max_dialog_turns=40, user_model="gpt-4o", language="zh"):
            # Don't call super().__init__() to avoid the original API key/URL logic
            self.model_name = model_name
            self.temperature = temperature
            self.top_p = top_p
            self.max_tokens = max_tokens
            self.max_dialog_turns = max_dialog_turns
            self.language = language
            self.user_model = user_model

            # Use provided API key/URL or fall back to environment variables
            self._dynamic_api_key = api_key or os.getenv("API_KEY")
            self._dynamic_base_url = base_url or os.getenv("BASE_URL")

            if not self._dynamic_api_key:
                raise ValueError("No API key found. Please set API_KEY environment variable.")
            if not self._dynamic_base_url:
                raise ValueError("No base URL found. Please set BASE_URL environment variable.")

            # Create OpenAI client with dynamic settings
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self._dynamic_base_url,
                api_key=self._dynamic_api_key
            )

    return DynamicAPIModelInference


def create_dynamic_apimodel_config(model_name, api_key=None, base_url=None):
    """
    Create a dynamic model configuration for ACEBench.
    Returns the dynamic class that can be used in the inference_map.
    """
    return create_dynamic_apimodel_class(api_key, base_url)