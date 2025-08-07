from abc import ABC, abstractmethod

class ModelProvider(ABC):
    """Abstract base class for all model providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates a response for the given prompt."""
        pass