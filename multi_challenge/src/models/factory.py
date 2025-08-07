from src.models.base import ModelProvider
from src.models.huggingface import HuggingFaceModel
from src.models.openai import OpenAIModel
from typing import Dict, Type


class ModelFactory:
    """Factory to create model providers based on the user input."""

    # Registry of available model providers
    providers: Dict[str, Type[ModelProvider]] = {
        'huggingface': HuggingFaceModel,
        'openai': OpenAIModel,
    }

    @classmethod
    def register_provider(cls, name: str, provider: Type[ModelProvider]):
        """Register a new model provider."""
        cls.providers[name] = provider

    @classmethod
    def get_provider(cls, provider_name: str, **kwargs) -> ModelProvider:
        """Create an instance of the requested model provider."""
        if provider_name not in cls.providers:
            raise ValueError(f"Model provider '{provider_name}' is not supported.")

        # Initialize the provider with the provided keyword arguments
        return cls.providers[provider_name](**kwargs)