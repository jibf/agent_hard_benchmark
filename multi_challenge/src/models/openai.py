from openai import OpenAI
import os
from typing import Any, Dict
from src.models.base import ModelProvider

class OpenAIModel(ModelProvider):
    """OpenAI model provider that uses GPT-4 for evaluation."""

    def __init__(self, model: str, temp: float, response_format: Any = None):
        """Initialize OpenAI API with the environment variable and other necessary parameters."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the .env file.")
        openai_client_config = {
            "api_key": api_key,
        }
        if os.getenv("OPENAI_BASE_URL"):
            openai_client_config["base_url"] = os.getenv("OPENAI_BASE_URL")

        self.client = OpenAI(**openai_client_config)

        self.model = model
        self.temp = float(temp)
        self.response_format = response_format or False

    def generate(self, prompt:Any):
        """Generate a response using the OpenAI GPT-4 model."""
        if type(prompt) == str:
            prompt = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) and 'role' in item and item['role'] in ['user', 'assistant'] for item in prompt):
            pass 
        else:
            raise ValueError("Prompt must be a string or a list of dictionaries with 'role' keys as 'user' or 'assistant'.")
        
        if self.response_format:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=prompt,
                temperature=self.temp,
                response_format=self.response_format
            )
            return response.choices[0].message.parsed
        else:
            response = self.client.chat.completions.create(
                model = self.model,
                messages = prompt,
                temperature = self.temp,
                max_tokens = 4096,
            )
            return response.choices[0].message.content