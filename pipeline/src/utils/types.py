from enum import Enum
from pydantic import BaseModel
from typing import Optional

class Benchmark(Enum):
    TAU_BENCH = "tau_bench"
    TAU2_BENCH = "tau2_bench"
    ACE_BENCH = "ace_bench"
    NEXUS_BENCH = "nexus_bench"
    TOOL_SANDBOX = "tool_sandbox"
    COMPLEX_FUNC_BENCH = "complex_func_bench"
    DRAFTER_BENCH = "drafter_bench"
    BFCLV2 = "bfcl_v2"
    BFCLV3 = "bfcl_v3"
    MULTI_CHALLENGE = "multi_challenge"


# Base class
class FormattedQuestion(BaseModel):
    benchmark: Benchmark                    # Benchmark that the question belongs to. example: Benchmark.TAU_BENCH
    question_id: str                        # A *unique* ID of the question. example: "Flight-83" (NOT just "83", to make this serve as a unique identifier within the benchmark!)
    instruction: str                        # The user's instruction (request) to the agent. example: "I want to rent a car for a self-driving trip starting tomorrow. Could you provide me with the ratings of the vehicle suppliers?"
                                            # For benchmarks that uses LLMs to simulate user (e.g., Tau-Bench), the user's instruction might be multi-turn would be different in every run. 
                                            # This case, use this field as a place for the prompt that initializes the user model. example: "Your user id is mia_li_3668. You want to fly from New York to Seattle on May 20..."
    available_function_list: list           # List of functions schemas that are available to the agent. This corresponds to the `tools` property in the OpenAI API endpoint. You can refer to https://platform.openai.com/docs/guides/function-calling.
    gt_conv_traj: list                      # Ground-truth conversation trajectory (if provided). If the benchmark does not provide one, leave this as an empty list.
    meta: Optional[dict] = None             # Other information of the question in dictionary type.



# TODO: Add fields so that all the necessary information for assessing each question is contained. Please try to keep variable names consistent to other benchmarks.

class ComplexFuncBenchQuestion(FormattedQuestion):  # CFB requires no additional field
    pass

class TauBenchQuestion(FormattedQuestion):
    agent_system_prompt: str                # the system prompt used to initialize the agent. e.g., "You are a specialized retail agent. Your task is to..."
    user_context: str                       # information of the user and order/reservation details


class Tau2BenchQuestion(FormattedQuestion):
    pass


class AceBenchQuestion(FormattedQuestion):
    pass


class NexusBenchQuestion(FormattedQuestion):
    pass


class ToolSandboxQuestion(FormattedQuestion):
    pass





class DrafterBenchQuestion(FormattedQuestion):
    pass


class BfclV2Question(FormattedQuestion):
    pass


class BfclV3Question(FormattedQuestion):
    pass


class MultiChallengeQuestion(FormattedQuestion):
    pass