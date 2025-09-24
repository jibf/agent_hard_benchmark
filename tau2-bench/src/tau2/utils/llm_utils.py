import json
import re
import os
from typing import Any, Optional

import litellm
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger

from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# litellm._turn_on_debug()
# litellm._turn_on_debug()  # Uncomment for debugging LiteLLM issues

if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

litellm.drop_params = True

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


# ALLOW_SONNET_THINKING = False
ALLOW_SONNET_THINKING = True

if not ALLOW_SONNET_THINKING:
    logger.warning("Sonnet thinking is disabled")


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def _is_custom_api_model(model: str) -> bool:
    """
    Check if the model name indicates a custom API (contains slash).
    e.g: "openai/gpt-4.1" -> True
    """
    # return "/" in model
    return any(prefix in model for prefix in [
            "anthropic/", "deepseek-ai/", "openai/", "google/", "togetherai/", "xai/", "huggingface"
        ])


def _get_custom_api_client():
    """
    Get OpenAI client for custom API using environment variables.
    """
    from openai import OpenAI
    
    base_url = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not base_url:
        raise ValueError("OPENAI_API_BASE environment variable is required for custom API models")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for custom API models")
    
    return OpenAI(base_url=base_url, api_key=api_key)


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.warning(f"Could not calculate cost for model {response.model}: {e}")
        # Return a default cost for custom models
        return 0.0001
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    """
    Get the usage of the response from the litellm completion.
    """
    try:
        usage = response.usage
        if usage is None:
            return None
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    except Exception as e:
        # logger.warning(f"Could not get usage for model {response.model}: {e}")
        return None


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert litellm messages to tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        base_url: Base URL for custom API endpoints.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    max_retries = kwargs.get("num_retries", DEFAULT_MAX_RETRIES)
    grok_retry_count = 0
    max_grok_retries = 3  # Maximum retries for Grok empty content issues
    
    while True:
        try:
            return _generate_single_attempt(
                model, messages, tools, tool_choice, base_url, **kwargs
            )
        except ValueError as e:
            # Check if this is a Grok empty content error
            if "grok" in model and "must have either content or tool calls" in str(e):
                grok_retry_count += 1
                if grok_retry_count <= max_grok_retries:
                    logger.warning(f"Grok returned empty content, retrying ({grok_retry_count}/{max_grok_retries})")
                    continue
                else:
                    logger.error(f"Grok failed after {max_grok_retries} retries due to empty content")
                    raise e
            else:
                # Re-raise non-Grok validation errors
                raise e


def _generate_single_attempt(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model (single attempt).
    """
    # print(model)
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}
    if _is_custom_api_model(model):
        kwargs["custom_llm_provider"] = "openai"
        if "huggingface" in model:
            kwargs["api_key"] = "dummy-key"
            # Use provided base_url if available, otherwise fall back to environment variable
            if base_url is not None:
                kwargs["api_base"] = base_url
            else:
                kwargs["api_base"] = os.getenv("HUGGINGFACE_API_BASE")
        else:
            kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
            # Use provided base_url if available, otherwise fall back to environment variable
            if base_url is not None:
                kwargs["api_base"] = base_url
            else:
                kwargs["api_base"] = os.getenv("OPENAI_API_BASE")

        if "anthropic" in model:
            if "thinking-on" in model:
                kwargs["api_base"] = os.getenv("CLAUDE_THINKING_API_BASE")
                kwargs["max_tokens"] = 16384
                kwargs["temperature"] = 1.0
                # Remove thinking parameter as it's not needed for the updated API
                if "thinking" in kwargs:
                    kwargs.pop("thinking")
            else:
                kwargs["max_tokens"] = 8192
            
            # Remove seed parameter for Anthropic models accessed via custom API
            if "seed" in kwargs:
                logger.info(f"Removing seed parameter for custom API Anthropic model {model} as it's not supported")
                kwargs.pop("seed")
            
            # Remove other parameters that Anthropic models don't support
            unsupported_params = ["frequency_penalty", "presence_penalty", "logit_bias"]
            for param in unsupported_params:
                if param in kwargs:
                    logger.info(f"Removing {param} parameter for custom API Anthropic model {model} as it's not supported")
                    kwargs.pop(param)
        if "openai" in model:
            model = "openai/" + model
        
        # messages=messages,
        #                 model="openai/" + self.model,  # Your custom API should handle different model families
        #                 api_key=self.api_key,  # Your custom API key
        #                 api_base=self.base_url,  # Your custom base URL
        #                 custom_llm_provider="openai",  # Tell LiteLLM to use OpenAI format
        #                 tools=self.tools_info,
        #                 temperature=self.temperature,
    litellm_messages = to_litellm_messages(messages)
    
    # For Claude thinking models, append a user message if there are tool calls
    # This is required by the Claude thinking model API
    if "anthropic" in model and "thinking-on" in model and tools:
        # Check if the last message is a tool message (indicating tool call results)
        if messages and isinstance(messages[-1], ToolMessage):
            litellm_messages.append({
                "role": "user",
                "content": "The tool has finished running. Please continue your reasoning"
            })
    
    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = "auto"
    
    # Add tool_choice to kwargs if tools are provided
    if tools and tool_choice is not None:
        # Claude models expect tool_choice as a dictionary, not a string
        if "anthropic" in model:
            kwargs["tool_choice"] = {"type": tool_choice}
        else:
            kwargs["tool_choice"] = tool_choice
    
    try:
        # print(litellm_messages)
        # print(tools)
        # print(kwargs)
        response = completion(
            model=model,
            messages=litellm_messages,
            tools=tools,
            **kwargs,
        )
    except Exception as e:
        logger.error(e)
        raise e
    cost = get_response_cost(response)
    usage = get_response_usage(response)
    response = response.choices[0]
    try:
        finish_reason = response.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response.message.content
    
    # Strip thinking tags for models that output them
    # Check if content contains thinking tags and strip them regardless of model
    if "huggingface" in model or "Qwen" in model or "DeepSeek-R1" in model or (content and "<think>" in content):
        
        if "claude" not in model and content and ("<think>" in content or "</think>" in content):
            match = re.search(r'</think>\s*(.*)$', content, re.DOTALL)
            if match:
                stripped_content = match.group(1).strip()
                # # Only use stripped content if it's not empty
                # if stripped_content:
                content = stripped_content

    if "grok" in model:
        logger.debug(f"Grok model {model} - original content: {repr(content)}")
        logger.debug(f"Grok model {model} - content type: {type(content)}")
        
        # Handle Grok's JSON string format
        if isinstance(content, str):
            try:
                parsed_content = json.loads(content)
                logger.debug(f"Grok model {model} - parsed JSON: {parsed_content}")
                if isinstance(parsed_content, dict) and "message" in parsed_content:
                    extracted_content = parsed_content["message"]
                    logger.debug(f"Grok model {model} - extracted message: {repr(extracted_content)}")
                    # Only use extracted content if it's not None and not empty
                    if extracted_content is not None and str(extracted_content).strip():
                        content = extracted_content
                        logger.debug(f"Grok model {model} - using extracted content: {repr(content)}")
                    else:
                        logger.warning(f"Grok model {model} - extracted message is empty or None: {repr(extracted_content)}")
            except json.JSONDecodeError as e:
                logger.warning(f"Grok model {model} - JSON decode error: {e}, keeping original content")
                # If JSON parsing fails, keep the original content
                pass
        elif isinstance(content, dict) and "message" in content:
            extracted_content = content["message"]
            logger.debug(f"Grok model {model} - extracted message from dict: {repr(extracted_content)}")
            # Only use extracted content if it's not None and not empty
            if extracted_content is not None and str(extracted_content).strip():
                content = extracted_content
                logger.debug(f"Grok model {model} - using extracted content from dict: {repr(content)}")
            else:
                logger.warning(f"Grok model {model} - extracted message from dict is empty or None: {repr(extracted_content)}")
        
    
    tool_calls = response.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        for tool_call in tool_calls
    ]
    tool_calls = tool_calls or None

    # Ensure we have valid content or tool calls for Grok models
    if "grok" in model:
        logger.debug(f"Grok model {model} - final content before validation: {repr(content)}")
        logger.debug(f"Grok model {model} - tool_calls: {tool_calls}")
        # If content is None, empty, or whitespace-only, and no tool calls, provide fallback
        if (content is None or (isinstance(content, str) and not content.strip())) and not tool_calls:
            logger.warning(f"Grok model {model} returned empty content and no tool calls, providing fallback message")
            content = "I apologize, but I couldn't generate a proper response. Please try again."
            logger.debug(f"Grok model {model} - using fallback content: {repr(content)}")
    
    # Debug logging to understand what's happening
    # print(f"Model {model} response - content: {repr(content)}, tool_calls: {tool_calls}")
    
    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict() if hasattr(response, 'to_dict') else response.model_dump(),
    )
    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage
