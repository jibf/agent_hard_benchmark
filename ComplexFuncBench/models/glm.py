import requests
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json
import random
import re
import requests
import ast
import os
import datetime
import copy
from zhipuai import ZhipuAI

from utils.utils import *

from openai import OpenAI
Message = dict[str, str]  # keys role, content
MessageList = list[Message]


class GLMAPIModel():
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.message = []
        self.client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
    
    @retry(max_attempts=10)
    def __call__(self, messages, tools=None, **kwargs: Any):
        if "function_call" not in json.dumps(messages, ensure_ascii=False):
            self.messages = copy.deepcopy(messages)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature=0.0,
                stream=False,
                do_sample=False,
                tools=tools,
                tool_choice="auto",
                max_tokens=2048
            )
            return completion.choices[0]
        
        except Exception as e:
            print(f"Exception: {e}")
            return None


class GLMVllmModel():
    def __init__(self, url, model_name):
        super().__init__()
        self.model_name = model_name
        self.message = []
        self.url = url

        self.client = OpenAI(
            api_key="EMPTY",
            base_url=self.url
        )

    def build_system_prompt(self, functions=None, current_time: Optional[float] = None):
        if functions is None:
            functions = []
        value = "你是一个名为 ChatGLM 的人工智能助手。你是基于智谱 AI 公司训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。"
        _date_prompt = "当前日期: %Y-%m-%d"
        if current_time is not None:
            value += "\n\n" + datetime.datetime.fromtimestamp(current_time).strftime(_date_prompt)
        if len(functions) > 0:
            value += "\n\n# 可用工具"
            contents = []
            for function in functions:
                content = f"\n\n## {function['name']}\n\n{json.dumps(function, ensure_ascii=False, indent=4)}"
                content += "\n在调用上述函数时，请使用 Json 格式表示调用的参数。"
                contents.append(content)
            random.shuffle(contents)
            value += "".join(contents)
        return value
    
    def build_single_message(self, role, metadata, message, add_dummy_prefix=False):
        assert role in ["system", "user", "assistant", "observation"], role
        role_tokens = f"<|{role}|>" + f"{metadata if metadata is not None else ''}\n"
        message_tokens = message
        tokens = role_tokens + message_tokens
        return tokens

    def get_full_prompt(self, messages):
        prompt = ""
        for item in messages:
            content = item["content"]
            prompt += self.build_single_message(item["role"], item.get("metadata", ""), content,
                                add_dummy_prefix=False)
        prompt += " <|assistant|>"
        return prompt

    def process_single_call(self, text):
        name, arguments = text.split("\n")
        arguments = json.loads(arguments)
        return {"name": name, "arguments": arguments}
    
    def get_standard_messages(self, messages, tools):
        system_prompt = self.build_system_prompt(functions=tools)
        messages.insert(0, {"role": "system", "content": system_prompt})
        new_messages = []
        for message in messages:
            if message['role'] == "assistant" and "function_call" in message:
                for call in message['function_call']:
                    value = json.dumps(call["arguments"], ensure_ascii=False)
                    new_messages.append({"role": "assistant", "metadata": call['name'], "content": value})
            else:
                new_messages.append(message)
        
        return new_messages
    
    @retry(max_attempts=5)
    def __call__(self, messages, tools=None, **kwargs: Any):
        generated_result = []
        function_calls = []
        messages = self.get_standard_messages(messages, tools)

        while True:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=self.get_full_prompt(messages),
                temperature=0.0,
                max_tokens=2048
            )

            try:
                function_call = self.process_single_call(response.choices[0].text)
                function_calls.append(function_call)
                messages.append({
                    "role": "assistant", "metadata": function_call['name'], 
                    "content": json.dumps(function_call["arguments"], ensure_ascii=False)})
            except:
                single_message = {"role": "assistant", "content": response.choices[0].text.strip()}
                generated_result.append(single_message)
                messages.append(single_message)
            
            try:
                if response.choices[0].stop_reason == 151338:
                    generated_result.append({"role": "assistant", "function_call": function_calls})
                    return generated_result
            
                elif response.choices[0].stop_reason == 151336:
                    return generated_result
            except Exception as e:
                print(e)
                return None


if __name__ =="__main__":
    model = GLMAPIModel("glm-4-alltools")

