from typing import Any, Dict
import os
from openai import OpenAI
import json
import sys
import copy
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.prompts import SimpleTemplatePrompt
from utils.utils import *


class LlamaModel:
    def __init__(self, url, model_name):
        super().__init__()
        self.temperature = 0.95
        self.model_name = model_name
        self.url = url
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=self.url)
        
        self.messages = []

    def _format_prompt(self, messages, function):
        formatted_prompt = "<|begin_of_text|>"

        system_message = ""
        remaining_messages = messages
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"].strip()
            remaining_messages = messages[1:]

        formatted_prompt += "<|start_header_id|>system<|end_header_id|>\n\n"
        formatted_prompt += "Cutting Knowledge Date: December 2023\n"
        formatted_prompt += "Today Date: 23 Jul 2024\n\n"
        formatted_prompt += "When you receive a tool call response, use the output to format an answer to the orginal user question.\n\n"
        formatted_prompt += "You are a helpful assistant with tool calling capabilities."
        formatted_prompt += system_message + "<|eot_id|>\n"

        # Llama pass in custom tools in first user message
        is_first_user_message = True
        for message in remaining_messages:
            if message["role"] == "user" and is_first_user_message:
                is_first_user_message = False
                formatted_prompt += "<|start_header_id|>user<|end_header_id|>\n\n"
                formatted_prompt += "Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\n"
                formatted_prompt += 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.\n\n'
                for func in function:
                    formatted_prompt += json.dumps(func, indent=4) + "\n\n"
                formatted_prompt += f"Question: {message['content'].strip()}<|eot_id|>"

            elif message["role"] == "tool":
                formatted_prompt += "<|start_header_id|>ipython<|end_header_id|>\n\n"
                if isinstance(message["content"], (dict, list)):
                    formatted_prompt += json.dumps(message["content"])
                else:
                    formatted_prompt += message["content"]
                formatted_prompt += "<|eot_id|>"

            else:
                formatted_prompt += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content'].strip()}<|eot_id|>"

        formatted_prompt += "\n<|start_header_id|>assistant<|end_header_id|>\n\n"

        return formatted_prompt
    

    @retry(max_attempts=5)
    def __call__(self, messages, tools=None, **kwargs: Any):
        if "function_call" not in json.dumps(messages, ensure_ascii=False):
            self.messages = copy.deepcopy(messages)
        prompt = self._format_prompt(self.messages, tools)
        try:
            completion = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                temperature=0.0,
                max_tokens=4096
            )
            return completion.choices[0].text
        except Exception as e:
            print(f"Exception: {e}")
            return None
