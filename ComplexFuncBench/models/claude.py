from typing import Any
import os
from anthropic import Anthropic
import copy
import json
import sys
from urllib.parse import unquote
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.prompts import SimpleTemplatePrompt
from utils.utils import *

class ClaudeModel:
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.client = Anthropic(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )

    def __call__(self, prefix, prompt: SimpleTemplatePrompt, **kwargs: Any):
        filled_prompt = prompt(**kwargs)
        prediction = self._predict(prefix, filled_prompt)
        return prediction

    @retry(max_attempts=10, delay=60)
    def _predict(self, prefix, query):
        try:
            completion = self.client.messages.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
            )
            return completion.content[0].text
        except Exception as e:
            print(f"Exception: {e}")
            return None

class FunctionCallClaude(ClaudeModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    @retry(max_attempts=10, delay=60)
    def __call__(self, messages, tools=None, **kwargs: Any):
        if "function_call" not in json.dumps(messages, ensure_ascii=False):
            self.messages = copy.deepcopy(messages)
        try:
            # Enable thinking for thinking models
            thinking_enabled = "thinking" in self.model_name.lower()
            create_params = {
                "model": self.model_name,
                "messages": self.messages,
                "temperature": 1.0 if thinking_enabled else 0.0,
                "tools": tools,
                "max_tokens": 2048
            }
            
            if thinking_enabled:
                create_params["thinking"] = {"enabled": True}
            
            print(f"Debug - create_params: {create_params}")  # Debug line
            response = self.client.messages.create(**create_params)
            return response
        except Exception as e:
            print(f"Exception: {e}")
            return None

if __name__ == "__main__":
    model = ClaudeModel("claude-3-5-sonnet-20240620")
    response_message = model._predict("You are a helpful assistant.", query="What is the capital of France?")
    print(response_message)