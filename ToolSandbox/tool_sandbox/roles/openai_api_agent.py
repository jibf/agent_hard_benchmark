# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Agent role for any model that conforms to OpenAI tool use API"""

from typing import Any, Iterable, List, Literal, Optional, Union, cast

from openai import NOT_GIVEN, NotGiven, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from requests.exceptions import HTTPError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from tool_sandbox.common.execution_context import RoleType, get_current_context
from tool_sandbox.common.message_conversion import (
    Message,
    openai_tool_call_to_python_code,
    to_openai_messages,
)
from tool_sandbox.common.tool_conversion import convert_to_openai_tools
from tool_sandbox.common.utils import all_logging_disabled
from tool_sandbox.roles.base_role import BaseRole


class OpenAIAPIAgent(BaseRole):
    """Agent role for any model that conforms to OpenAI tool use API"""

    role_type: RoleType = RoleType.AGENT
    model_name: str

    def __init__(self) -> None:
        # We set the `base_url` explicitly here to avoid picking up the
        # `OPENAI_BASE_URL` environment variable that may be set for serving models as
        # OpenAI API compatible servers.
        # self.openai_client: OpenAI = OpenAI(base_url="https://api.openai.com/v1")
        # self.openai_client: OpenAI = OpenAI(api_key="sk-sgl-MH7bEVVJlBp3RT_P5cPQ6-KfC1qJElBRCfTDHy40Ue4", base_url="http://5.78.122.79:10000/v1", )
        self.openai_client: OpenAI = OpenAI(base_url="http://5.78.122.79:10000/v1", )
        # import pdb; pdb.set_trace()

    def respond(self, ending_index: Optional[int] = None) -> None:
        """Reads a List of messages and attempt to respond with a Message

        Specifically, interprets system, user, execution environment messages and sends out NL response to user, or
        code snippet to execution environment.

        Message comes from current context, the last k messages should be directed to this role type
        Response are written to current context as well. n new messages, addressed to appropriate recipient
        k != n when dealing with parallel function call and responses. Parallel function call are expanded into
        individual messages, parallel function call responses are combined as 1 OpenAI API request

        Args:
            ending_index:   Optional index. Will respond to message located at ending_index instead of most recent one
                            if provided. Utility for processing system message, which could contain multiple entries
                            before each was responded to

        Raises:
            KeyError:   When the last message is not directed to this role
        """
        messages: List[Message] = self.get_messages(ending_index=ending_index)
        response_messages: List[Message] = []
        self.messages_validation(messages=messages)
        # Keeps only relevant messages
        messages = self.filter_messages(messages=messages)
        # Does not respond to System
        if messages[-1].sender == RoleType.SYSTEM:
            return
        # Get OpenAI tools if most recent message is from user
        available_tools = self.get_available_tools()
        available_tool_names = set(available_tools.keys())
        openai_tools = (
            convert_to_openai_tools(available_tools)
            if messages[-1].sender == RoleType.USER
            or messages[-1].sender == RoleType.EXECUTION_ENVIRONMENT
            else NOT_GIVEN
        )
        # We need a cast here since `convert_to_openai_tool` returns a plain dict, but
        # `ChatCompletionToolParam` is a `TypedDict`.
        openai_tools = cast(
            Union[Iterable[ChatCompletionToolParam], NotGiven],
            openai_tools,
        )
        # Convert to OpenAI messages.
        current_context = get_current_context()
        openai_messages, _ = to_openai_messages(messages)
        # Call model
        response = self.model_inference(
            openai_messages=openai_messages, openai_tools=openai_tools
        )
        # Parse response
        openai_response_message = response.choices[0].message
        # Message contains no tool call, aka addressed to user
        if openai_response_message.tool_calls is None:
            assert openai_response_message.content is not None
            response_messages = [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content=openai_response_message.content,
                )
            ]
        else:
            assert openai_tools is not NOT_GIVEN
            for tool_call in openai_response_message.tool_calls:
                tool_call.id = tool_call.id.replace("-", "_")
                # The response contains the agent facing tool name so we need to get
                # the execution facing tool name when creating the Python code.
                execution_facing_tool_name = (
                    current_context.get_execution_facing_tool_name(
                        tool_call.function.name
                    )
                )
                response_messages.append(
                    Message(
                        sender=self.role_type,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=openai_tool_call_to_python_code(
                            tool_call,
                            available_tool_names,
                            execution_facing_tool_name=execution_facing_tool_name,
                        ),
                        openai_tool_call_id=tool_call.id,
                        openai_function_name=tool_call.function.name,
                    )
                )
        self.add_messages(response_messages)

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(HTTPError),
    )
    def model_inference(
        self,
        openai_messages: list[
            dict[
                Literal["role", "content", "tool_call_id", "name", "tool_calls"],
                Any,
            ]
        ],
        openai_tools: Union[Iterable[ChatCompletionToolParam], NotGiven],
    ) -> ChatCompletion:
        """Run OpenAI model inference

        Args:
            openai_messages:    List of OpenAI API format messages
            openai_tools:       List of OpenAI API format tools definition

        Returns:
            OpenAI API chat completion object
        """
        with all_logging_disabled():
            return self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=cast(list[ChatCompletionMessageParam], openai_messages),
                tools=openai_tools,
            )


class GPT_4_0125_Agent(OpenAIAPIAgent):
    model_name = "gpt-4-0125-preview"


class GPT_3_5_0125_Agent(OpenAIAPIAgent):
    model_name = "gpt-3.5-turbo-0125"


class GPT_4_o_2024_05_13_Agent(OpenAIAPIAgent):
    model_name = "gpt-4o-2024-05-13"

class GPT_4_o_2024_08_06_Agent(OpenAIAPIAgent):
    model_name = "openai/gpt-4o-20240806"

class GPT_4_o_mini_Agent(OpenAIAPIAgent):
    model_name = "openai/gpt-4o-mini"

class GPT_o_3_high_Agent(OpenAIAPIAgent):
    model_name = "openai/o3-high"

class GPT_o_4_mini_high_Agent(OpenAIAPIAgent):
    model_name = "openai/o4-mini-high"

class GPT_4_1_Agent(OpenAIAPIAgent):
    model_name = "openai/gpt-4.1"

# Qwen-235B models from Together AI
class Qwen_235B_Agent(OpenAIAPIAgent):
    model_name = "togetherai/Qwen/Qwen3-235B-A22B-FP8"

class Qwen_235B_Instruct_Agent(OpenAIAPIAgent):
    model_name = "togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

class Qwen_235B_Thinking_Agent(OpenAIAPIAgent):
    model_name = "togetherai/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"

class Claude_4_Sonnet_Agent(OpenAIAPIAgent):
    model_name = "anthropic/claude-4-sonnet-thinking-off"

class Claude_4_Sonnet_Thinking_Agent(OpenAIAPIAgent):
    model_name = "anthropic/claude-4-sonnet-thinking-on-10k"

class DeepSeek_V3_Agent(OpenAIAPIAgent):
    model_name = "deepseek-ai/DeepSeek-V3-0324"

class DeepSeek_R1_Agent(OpenAIAPIAgent):
    model_name = "deepseek-ai/DeepSeek-R1-0528"

class Grok_4_Agent(OpenAIAPIAgent):
    model_name = "xai/grok-4"

class Kimi_K2_Agent(OpenAIAPIAgent):
    model_name = "togetherai/moonshotai/Kimi-K2-Instruct"