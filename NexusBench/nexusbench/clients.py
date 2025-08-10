from typing import Optional

from nexusflowai import NexusflowAI

from openai import OpenAI

from anthropic import Anthropic

from mistralai.client import MistralClient

from nexusbench.utils import handle_exceptions


class BaseClient:
    def __init__(self, api_key, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.client = self.create_client()

    def create_client(self):
        raise NotImplementedError("Subclasses must implement create_client method")

    @handle_exceptions
    def get_completion(self, prompt, model=None, contextual_history=None):
        raise NotImplementedError("Subclasses must implement get_completion method")

    def get_client_params(self, base_url_key: str = "base_url"):
        params = {"api_key": self.api_key}
        if self.base_url is not None:
            params[base_url_key] = self.base_url

        return params


class NexusflowAIFCClient(BaseClient):
    def create_client(self):
        return NexusflowAI(**self.get_client_params())

    @handle_exceptions
    def get_completion(
        self, prompt, model="nexus-tool-use-20240816", contextual_history=None
    ):
        self.client: NexusflowAI
        # Create base parameters
        params = {
            "model": model,
            "messages": prompt["messages"],
            "tools": prompt["tools"],
            "max_tokens": 2048,
            "temperature": 0.0,
        }
        
        # Try to add parallel_tool_calls, but handle models that don't support it
        try:
            params["parallel_tool_calls"] = False
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            if "parallel_tool_calls" in str(e) and "not supported" in str(e).lower():
                # Remove parallel_tool_calls parameter for unsupported models
                del params["parallel_tool_calls"]
                response = self.client.chat.completions.create(**params)
            else:
                # Re-raise if it's a different error
                raise e

        return response


class NexusflowAICompletionsClient(NexusflowAIFCClient):
    @handle_exceptions
    def get_completion(
        self, prompt, model="nexus-tool-use-20240816", contextual_history=None
    ):
        self.client: NexusflowAI
        # Create base parameters
        params = {
            "model": model,
            "messages": prompt["messages"],
            "tools": prompt["tools"],
            "max_tokens": 2048,
            "temperature": 0.0,
        }
        
        # Try to add parallel_tool_calls, but handle models that don't support it
        try:
            params["parallel_tool_calls"] = False
            response = self.client.completions.create_with_tools(**params)
        except Exception as e:
            if "parallel_tool_calls" in str(e) and "not supported" in str(e).lower():
                # Remove parallel_tool_calls parameter for unsupported models
                del params["parallel_tool_calls"]
                response = self.client.completions.create_with_tools(**params)
            else:
                # Re-raise if it's a different error
                raise e

        return response


class QwenFCClient(BaseClient):
    def __init__(self, api_key, base_url, model):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = self.create_client()

    def create_client(self):
        from qwen_agent.llm import get_chat_model

        llm = get_chat_model(
            {
                "model": self.model,
                "model_server": self.base_url,
                "api_key": self.api_key,
            }
        )
        return llm

    @handle_exceptions
    def get_completion(
        self,
        prompt,
        model="Nexusflow/Qwen-2.5-72B-Instruct",
        contextual_history=None,
    ):
        for responses in self.client.chat(
            messages=prompt["messages"],
            functions=prompt["tools"],
            extra_generate_cfg=dict(
                parallel_function_calls=False, temperature=0.0, max_tokens=1024
            ),
        ):
            pass

        return responses[0]


class OpenAIFCClient(BaseClient):
    def create_client(self):
        return OpenAI(**self.get_client_params())

    @handle_exceptions
    def get_completion(
        self, prompt, model="gpt-4-0125-preview", contextual_history=None
    ):
        is_claude = "claude" in model.lower() or "anthropic/" in model.lower()

        if is_claude:
            # Claude models expect `tool_choice` to be a dict
            params = {
                "model": model,
                "messages": prompt["messages"],
                "tools": prompt["tools"],
                "tool_choice": {"type": "auto"},
                "max_tokens": 2048,
                "temperature": 1.0 if "thinking-on-10k" in model.lower() else 0.0,
            }
            response = self.client.chat.completions.create(**params)
        else:
            # Create base parameters
            params = {
                "model": model,
                "messages": prompt["messages"],
                "tools": prompt["tools"],
                "tool_choice": "auto",
                "max_tokens": 2048,
                "temperature": 0.0,
            }
            
            # Try to add parallel_tool_calls, but handle models that don't support it
            try:
                params["parallel_tool_calls"] = False
                response = self.client.chat.completions.create(**params)
            except Exception as e:
                if "parallel_tool_calls" in str(e) and "not supported" in str(e).lower():
                    # Remove parallel_tool_calls parameter for unsupported models
                    del params["parallel_tool_calls"]
                    response = self.client.chat.completions.create(**params)
                else:
                    # Re-raise if it's a different error
                    raise e
        
        return response


class MistralFCClient(BaseClient):
    def create_client(self):
        return MistralClient(**self.get_client_params("endpoint"))

    @handle_exceptions
    def get_completion(
        self, prompt, model="mistral-large-2407", contextual_history=None
    ):
        # Create base parameters
        params = {
            "model": model,
            "messages": prompt["messages"],
            "tools": prompt["tools"],
            "tool_choice": "required",
            "max_tokens": 2048,
            "temperature": 0.0,
        }
        
        # Try to add parallel_tool_calls, but handle models that don't support it
        try:
            params["parallel_tool_calls"] = False
            response = self.client.chat(**params)
        except Exception as e:
            if "parallel_tool_calls" in str(e) and "not supported" in str(e).lower():
                # Remove parallel_tool_calls parameter for unsupported models
                del params["parallel_tool_calls"]
                response = self.client.chat(**params)
            else:
                # Re-raise if it's a different error
                raise e
        
        return response


class AnthropicFCClient(BaseClient):
    def create_client(self):
        return Anthropic(**self.get_client_params())

    @handle_exceptions
    def get_completion(
        self, prompt, model="claude-3-5-sonnet-20240620", contextual_history=None
    ):
        response = self.client.messages.create(
            model=model,
            messages=prompt["messages"],
            tools=prompt["tools"],
            system=prompt["system"],
            max_tokens=2048,
            tool_choice={"type": "auto"},
            temperature=0.0,
        )

        return response
