# Copyright Sierra

import json
import os
import re
from litellm import completion
from typing import List, Optional, Dict, Any

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME

from dotenv import load_dotenv
load_dotenv()

# Global flag to control thinking models
ALLOW_SONNET_THINKING = True

class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        base_url: Optional[str] = None,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        
        # Check if this is a custom API model (contains slash)
        self.is_custom_api = any(prefix in model for prefix in [
            "anthropic/", "deepseek-ai/", "openai/", "google/", "togetherai/", "xai/", "Qwen"
        ])

        # Handle thinking models
        if 'anthropic' in self.model and 'thinking-on' in self.model:
            self.temperature = 1.0
            if not ALLOW_SONNET_THINKING:
                raise ValueError("Sonnet thinking is disabled")
        
        if self.is_custom_api:
            if 'local' in self.model or 'huggingface' in self.model:
                # Use provided base_url if available and model contains 'huggingface'
                if 'huggingface' in self.model and base_url is not None:
                    self.base_url = base_url
                # elif "8B" in self.model:
                #     self.base_url = os.getenv("SGLANG_API_BASE_1")
                # else:
                #     self.base_url = os.getenv("SGLANG_API_BASE_2")
                self.api_key = "dummy-key"
            else:
                # Handle different API base URLs for thinking vs non-thinking models
                if "anthropic" in self.model:
                    if "thinking-on" in self.model:
                        self.base_url = os.getenv("CLAUDE_THINKING_API_BASE")
                    else:
                        self.base_url = os.getenv("OPENAI_API_BASE")
                else:
                    self.base_url = os.getenv("OPENAI_API_BASE")
                self.api_key = os.getenv("OPENAI_API_KEY")

    # def _is_custom_api_model(self, model: str) -> bool:
    #     """
    #     Check if the model name indicates a custom API (contains slash).
    #     """
    #     return any(prefix in model for prefix in [
    #         "anthropic/", "deepseek-ai/", "openai/", "google/", "togetherai/", "xai/", "huggingface"
    #     ])


    # def _prepare_messages_for_thinking_model(self, obs: str) -> List[Dict[str, Any]]:
    #     """
    #     Prepare messages for thinking models with proper format.
    #     """
    #     return [
    #         {"role": "system", "content": self.wiki},
    #         {
    #             "role": "user", 
    #             "content": [
    #                 {"type": "text", "text": obs},
    #                 {
    #                     "type": "thinking",
    #                     "thinking": {
    #                         "content": ""
    #                     },
    #                     "signature": "thinking-on"
    #                 }
    #             ]
    #         },
    #     ]

    # def _prepare_thinking_user_message(self, obs: str) -> Dict[str, Any]:
    #     """
    #     Prepare a user message for thinking models.
    #     """
    #     return {
    #         "role": "user", 
    #         "content": [
    #             {"type": "text", "text": obs},
    #             {
    #                 "type": "thinking",
    #                 "thinking": {
    #                     "content": ""
    #                 },
    #                 "signature": "thinking-on"
    #             }
    #         ]
    #     }

    def _prepare_completion_kwargs(self) -> Dict[str, Any]:
        """
        Prepare completion kwargs based on model type.
        """
        kwargs = {
            "temperature": self.temperature,
        }
        
        if self.is_custom_api:
            kwargs["custom_llm_provider"] = "openai"
            kwargs["api_key"] = self.api_key
            kwargs["api_base"] = self.base_url
            # if "huggingface" in self.model:
            #     kwargs["api_key"] = "dummy-key"
            #     # kwargs["api_base"] = self.base_url
            # else:
            #     # kwargs["api_key"] = self.api_key
            #     # kwargs["api_base"] = self.base_url

            if "anthropic" in self.model:
                if "thinking-on" in self.model:
                    # kwargs["api_base"] = os.getenv("CLAUDE_THINKING_API_BASE")
                    kwargs["max_tokens"] = 16384
                    kwargs["temperature"] = 1.0
                    # Remove thinking parameter as it's not needed for the updated API
                    if "thinking" in kwargs:
                        kwargs.pop("thinking")
                else:
                    kwargs["api_base"] = os.getenv("OPENAI_API_BASE")
                    kwargs["max_tokens"] = 8192
                
                # Remove seed parameter for Anthropic models accessed via custom API
                if "seed" in kwargs:
                    kwargs.pop("seed")
                
                # Remove other parameters that Anthropic models don't support
                unsupported_params = ["frequency_penalty", "presence_penalty", "logit_bias"]
                for param in unsupported_params:
                    if param in kwargs:
                        kwargs.pop(param)
            
            if "openai" in self.model:
                kwargs["model"] = "openai/" + self.model
        
        return kwargs

    # def _prepare_messages_for_completion(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #     """
    #     Prepare messages for completion, including special handling for Claude thinking models.
    #     """
    #     # For Claude thinking models, append a user message if there are tool calls
    #     # This is required by the Claude thinking model API
    #     if "anthropic" in self.model and "thinking-on" in self.model and self.tools_info:
    #         # Check if the last message is a tool message (indicating tool call results)
    #         if messages and messages[-1].get("role") == "tool":
    #             messages.append({
    #                 "role": "user",
    #                 "content": "The tool has finished running. Please continue your reasoning"
    #             })
        
    #     return messages

    # def _prepare_tool_choice(self) -> Optional[Dict[str, Any]]:
    #     """
    #     Prepare tool_choice parameter based on model type.
    #     """
    #     if not self.tools_info:
    #         return None
        
    #     # Claude models expect tool_choice as a dictionary, not a string
    #     if "anthropic" in self.model:
    #         return {"type": "auto"}
    #     else:
    #         return "auto"

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30, progress_bar=None
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        
        # Initialize messages based on model type
        # if 'thinking-on' in self.model:
        #     messages = self._prepare_messages_for_thinking_model(obs)
        # else:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]

        for _ in range(max_num_steps):
            # Prepare completion kwargs
            kwargs = self._prepare_completion_kwargs()
            
            # Add tools and tool_choice
            if self.tools_info:
                kwargs["tools"] = self.tools_info
                # tool_choice = self._prepare_tool_choice()
                    
                if "anthropic" in self.model:
                    kwargs["tool_choice"] = {"type": "auto"}
                else:
                    kwargs["tool_choice"] = "auto"

            # Prepare messages for completion
            # completion_messages = self._prepare_messages_for_completion(messages.copy())

            # Make the completion call
            if self.is_custom_api:
                if 'anthropic' in self.model and 'thinking-on' in self.model:
                    if messages[-1]["role"] != "user":
                        messages.append({"role": "user", "content": "The tool has finished running. Please continue your reasoning."})
                    # print(self.api_key, self.base_url)
                    # print(messages)
                    # print(kwargs)
                    res = completion(
                        messages=messages,
                        model=self.model,
                        **kwargs
                    )
                    # print(res)
                    # print(**kwargs)

                # elif 'openai' in self.model:
                #     res = completion(
                #         messages=completion_messages,
                #         model="openai/" + self.model,
                #         **kwargs
                #     )
                else:
                    res = completion(
                        messages=messages,
                        model=self.model,
                        **kwargs
                    )
            else:
                res = completion(
                    messages=messages,
                    model=self.model,
                    custom_llm_provider=self.provider,
                    tools=self.tools_info,
                    temperature=self.temperature,
                )

            next_message = res.choices[0].message.model_dump()
            
            # Filter out reasoning_content for Claude thinking-on models
            if 'anthropic' in self.model and 'thinking-on' in self.model:
                if 'reasoning_content' in next_message:
                    next_message.pop('reasoning_content')
            
            # Handle case where custom API doesn't return cost information
            if hasattr(res, '_hidden_params') and res._hidden_params and "response_cost" in res._hidden_params:
                cost = res._hidden_params["response_cost"]
                if cost is not None:
                    total_cost += cost
            
            action = message_to_action(next_message)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            
            # Update progress bar if provided
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'action': action.name,
                    'reward': f"{reward:.2f}",
                    'done': env_response.done
                })
            
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                # if 'thinking-on' in self.model:
                #     # For thinking-on models, wrap the observation in the required format
                #     messages.extend(
                #         [
                #             next_message,
                #             self._prepare_thinking_user_message(env_response.observation),
                #         ]
                #     )
                # else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            
            if env_response.done:
                break
                
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )


def message_to_action(
    message: Dict[str, Any],
) -> Action:
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})
