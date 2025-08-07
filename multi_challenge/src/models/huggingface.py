from transformers import pipeline
from src.models.base import ModelProvider
import torch
import os
from typing import List, Dict

class HuggingFaceModel(ModelProvider):
    def __init__(self, model_path: str, temp: float, top_p: float):
        self.temp = temp
        self.top_p = top_p
        api_key = os.getenv("HUGGINGFACE_TOKEN")
        if not api_key:
            raise ValueError("HUGGINGFACE_TOKEN is not set in the .env file.")
        self.generator = pipeline(
            'text-generation',
            model=model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            batch_size=8
        )
    def generate(self, chat: List[Dict]) -> str:
        """Generate a response from the model."""
        response = self.generator(
            chat,
            max_new_tokens=2000,
            temperature=self.temp,
            top_p=self.top_p,
            do_sample=True  # Required when using temperature/top_p
        )
        return response[0]['generated_text'][-1]['content']
