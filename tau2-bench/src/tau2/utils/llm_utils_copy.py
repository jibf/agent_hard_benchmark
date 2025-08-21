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
        logger.error(e)
        return 0.0
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
        logger.error(e)
        return None


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert litellm messages to tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        if message["role"] in ignore_roles:
            continue
        if message["role"] == "system":
            tau2_messages.append(SystemMessage(content=message["content"]))
        elif message["role"] == "user":
            tau2_messages.append(UserMessage(content=message["content"]))
        elif message["role"] == "assistant":
            tau2_messages.append(AssistantMessage(content=message["content"]))
        elif message["role"] == "tool":
            tau2_messages.append(ToolMessage(content=message["content"]))
        else:
            logger.warning(f"Unknown message role: {message['role']}")
    return tau2_messages


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert tau2 messages to litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
        elif isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            message_dict = {"role": "assistant", "content": message.content}
            if message.tool_calls:
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
                message_dict["tool_calls"] = tool_calls
            litellm_messages.append(message_dict)
        elif isinstance(message, ToolMessage):
            message_dict = {"role": "tool", "content": message.content}
            if hasattr(message, 'id'):
                message_dict["tool_call_id"] = message.id
            litellm_messages.append(message_dict)
        else:
            logger.warning(f"Unknown message type: {type(message)}")
    return litellm_messages


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}
    
    litellm_messages = to_litellm_messages(messages)
    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = "auto"
    
    # Check if this is a custom API model
    if _is_custom_api_model(model):
        try:
            # Use direct OpenAI client for custom API
            client = _get_custom_api_client()
            
            # Use the full model name (e.g., "openai/gpt-4.1") for custom APIs
            # since the API expects the full model ID
            
            # Prepare the request
            request_kwargs = {
                "model": model,  # Use the full model name
                "messages": litellm_messages,
                "temperature": kwargs.get("temperature", 0.0),
            }
            
            if tools:
                request_kwargs["tools"] = tools
                request_kwargs["tool_choice"] = tool_choice
            
            response = client.chat.completions.create(**request_kwargs)
            
            # For custom API, we'll use the response directly without converting to LiteLLM format
            # since the response structure is already compatible
            
            # Estimate cost for custom API (you may want to adjust this)
            cost = 0.0001  # Rough estimate
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            
            # Use the response directly for the rest of the processing
            # The response structure should be compatible with what the rest of the code expects
            
        except Exception as e:
            logger.error(f"Custom API error: {e}")
            raise e
    else:
        # Use litellm for standard APIs
        try:
            response = completion(
                model=model,
                messages=litellm_messages,
                tools=tools,
                tool_choice=tool_choice,
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
    tool_calls = response.message.tool_calls or []
    
    # Convert tool calls to the expected format
    converted_tool_calls = []
    for tool_call in tool_calls:
        try:
            # Handle both OpenAI and LiteLLM tool call formats
            if hasattr(tool_call, 'function'):
                # OpenAI format
                function = tool_call.function
                arguments = json.loads(function.arguments) if hasattr(function, 'arguments') else {}
                converted_tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=function.name,
                        arguments=arguments,
                    )
                )
            elif hasattr(tool_call, 'name'):
                # LiteLLM format
                converted_tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                    )
                )
            else:
                # Fallback for other formats
                logger.warning(f"Unknown tool call format: {tool_call}")
        except Exception as e:
            logger.error(f"Error converting tool call: {e}")
            continue
    
    tool_calls = converted_tool_calls or None

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
    agent_tokens = 0
    user_tokens = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is not None:
            if isinstance(message, AssistantMessage):
                agent_tokens += message.usage.get("completion_tokens", 0)
            elif isinstance(message, UserMessage):
                user_tokens += message.usage.get("prompt_tokens", 0)
        else:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
    return {"agent_tokens": agent_tokens, "user_tokens": user_tokens}
