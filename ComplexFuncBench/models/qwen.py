import os
import copy
from typing import Any, Dict
from utils.utils import *
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

"""
You can also deploy Qwen2.5 via vLLM, please enable the auto-tool-choice. 
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
```
Note: Tool support has been available in vllm since v0.6.0. Be sure to install a version that supports tool use.
Reference: https://qwen.readthedocs.io/en/latest/framework/function_call.html#vllm
"""

class QwenModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.messages = []
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )

    @retry(max_attempts=5, delay=20)
    def __call__(self, messages, tools=None, **kwargs: Any):
        if "function_call" not in json.dumps(messages, ensure_ascii=False):
            self.messages = copy.deepcopy(messages)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature=0.0,
                tools=tools,
                tool_choice="auto",
                max_tokens=2048
            )
            return completion.model_dump()['choices'][0]['message']
        except Exception as e:
            print(f"Exception: {e}")
            return None