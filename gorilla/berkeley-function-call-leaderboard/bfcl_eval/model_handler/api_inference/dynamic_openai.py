"""
Dynamic OpenAI-compatible model handler.
Based on ToolSandbox's dynamic model creation approach.
"""
import os
from typing import Optional

from bfcl_eval.constants.model_config import ModelConfig
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from openai import OpenAI


def create_dynamic_handler_class(api_key=None, base_url=None, is_fc_model=False):
    """
    Create a dynamic handler class with pre-configured API settings.

    Args:
        api_key: API key for the model service
        base_url: Base URL for the API endpoint
        is_fc_model: Whether this is a function calling model

    Returns:
        A handler class that can be instantiated with standard (model_name, temperature) signature
    """
    class DynamicOpenAIHandler(OpenAICompletionsHandler):
        def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
            # Store dynamic configuration
            self._dynamic_api_key = api_key or os.getenv("API_KEY")
            self._dynamic_base_url = base_url or os.getenv("BASE_URL")
            self._is_fc_model_dynamic = is_fc_model

            # Temporarily set OPENAI_ environment variables for parent init
            old_api_key = os.environ.get("OPENAI_API_KEY")
            old_base_url = os.environ.get("OPENAI_BASE_URL")

            if self._dynamic_api_key:
                os.environ["OPENAI_API_KEY"] = self._dynamic_api_key
            if self._dynamic_base_url:
                os.environ["OPENAI_BASE_URL"] = self._dynamic_base_url

            try:
                # Call parent init
                super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
            finally:
                # Restore original environment variables
                if old_api_key is not None:
                    os.environ["OPENAI_API_KEY"] = old_api_key
                elif "OPENAI_API_KEY" in os.environ:
                    del os.environ["OPENAI_API_KEY"]

                if old_base_url is not None:
                    os.environ["OPENAI_BASE_URL"] = old_base_url
                elif "OPENAI_BASE_URL" in os.environ:
                    del os.environ["OPENAI_BASE_URL"]

            # Set the is_fc_model property
            self.is_fc_model = self._is_fc_model_dynamic

    return DynamicOpenAIHandler


def create_dynamic_model_config(
    model_name: str,
    display_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    is_fc_model: bool = False,
    org: str = "Custom",
    license: str = "Unknown"
):
    """
    Create a dynamic model configuration.

    Args:
        model_name: The model name to use
        display_name: Display name for the leaderboard (defaults to model_name)
        api_key: API key for the model service
        base_url: Base URL for the API endpoint
        is_fc_model: Whether this is a function calling model
        org: Organization name
        license: License information

    Returns:
        A ModelConfig object for the dynamic model
    """
    # Create a dynamic handler class with pre-configured API settings
    handler_class = create_dynamic_handler_class(
        api_key=api_key,
        base_url=base_url,
        is_fc_model=is_fc_model
    )

    return ModelConfig(
        model_name=model_name,
        display_name=display_name or f"{model_name} (Dynamic)",
        url=base_url or "Custom API",
        org=org,
        license=license,
        model_handler=handler_class,
        input_price=None,  # Unknown for custom models
        output_price=None,  # Unknown for custom models
        is_fc_model=is_fc_model,
        underscore_to_dot=False,
    )